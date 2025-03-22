import concurrent.futures
import glob
import json
import os
import re
import threading
import requests
import traceback
from typing import Annotated, List, Generator, Optional

# ======== FastAPI Imports ========
from fastapi import HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse, RedirectResponse
import httpx
from loguru import logger

# ======== Lepton AI Imports ========
import leptonai
from leptonai import Client
from leptonai.photon import Photon, StaticFiles
from leptonai.photon.types import to_bool
from leptonai.api.v0.workspace import WorkspaceInfoLocalRecord
from leptonai.util import tool

# ======== KV Imports ========
import shelve

# ======== Search Engine Functions ========
from langchain_community.tools import DuckDuckGoSearchResults

# ======== Prompt Texts ========
import nettyPrompts
from arag.arag import get_rag_context
from arag.route import questionRouting

################################################################################
# Constant values for the RAG model.
################################################################################
from dotenv import load_dotenv
load_dotenv(override = True) # Load the .env file
logger.info(f"Loaded .env file successfully.")

# Specify the number of references from the search engine you want to use.
# 8 is usually a good number.
REFERENCE_COUNT = 8
PROD_MODE = False

# A set of stop words to use - this is not a complete set, and you may want to
# add more given your observation.
if os.environ.get("LLM_USE_CUSTOM_SERVER"):
    # For custom models, we will use a different set of stop words.
    stop_words = [
        # "<|im_end|>",
        "[End]",
        # "[end]",
        "\nReferences:\n",
        "\nSources:\n",
        "End.",
    ]
else:
    # OpenAI API only supports the four stop words below.
    stop_words = [
        "<|im_end|>",
        "[End]",
        "[end]",
        "\nReferences:\n",
        "\nSources:\n",
        "End.",
    ]

# ======== Search Engine (ARAG) Functions ========

def search_with_duckduckgo(query: str):
    """
    Search with duckduckgo and return the contexts.
    """
    search = DuckDuckGoSearchResults(output_format = "list", num_results = REFERENCE_COUNT)
    results = search.invoke(query)
    # Convert the search results to the same format as bing/google
    contexts = []
    for result in results:
        contexts.append({
            "name": result["title"],
            "url": result["link"],
            "snippet": result["snippet"]
        })
    return contexts

def search_with_adaptiveRAG(query: str):
    """
    Search with the adaptive RAG model.
    """
    return get_rag_context(query)

# ======== Photon Class ========

class RAG(Photon):
    """
    Retrieval-Augmented Generation Demo from Lepton AI.

    This is a minimal example to show how to build a RAG engine with Lepton AI.
    It uses search engine to obtain results based on user queries, and then uses
    LLM models to generate the answer as well as related questions. The results
    are then stored in a KV so that it can be retrieved later.
    """

    requirement_dependency = [
        "openai",  # for openai client usage.
    ]

    extra_files = glob.glob("ui/**/*", recursive=True)

    deployment_template = {
        # All actual computations are carried out via remote apis, so
        # we will use a cpu.small instance which is already enough for most of
        # the work.
        "resource_shape": "cpu.small",
        # You most likely don't need to change this.
        "env": {
            # RAG Backend: LEPTON or DUCKDUCKGO
            "BACKEND": "DUCKDUCKGO",
            # Specify the LLM model you are going to use.
            "LLM_MODEL": "mixtral-8x7b",
            # KV name (SQLite database name) to store the search results.
            "KV_NAME": "netty-chat.kv",
            # If set to true, will generate related questions. Otherwise, will not.
            "RELATED_QUESTIONS": "true",
            # On the lepton platform, allow web access when you are logged in.
            "LEPTON_ENABLE_AUTH_BY_COOKIE": "true",
        },
        # Secrets you need to have: search api subscription key, and lepton
        # workspace token to query lepton's llama models.
        "secret": [
            # You need to specify the workspace token to query lepton's LLM models.
            "LEPTON_WORKSPACE_TOKEN",
        ],
    }

    # It's just a bunch of api calls, so our own deployment can be made massively
    # concurrent.
    handler_max_concurrency = 16

    def local_client(self, force_openai=False):
        """
        Gets a thread-local client, so in case openai clients are not thread safe,
        each thread will have its own client.
        """
        import openai

        thread_local = threading.local()
        try:
            return thread_local.client
        except AttributeError:
            if not os.environ.get("LLM_USE_CUSTOM_SERVER"):
                logger.info("Using Lepton LLM model.")
                thread_local.client = openai.OpenAI(
                    base_url=f"https://{self.model}.lepton.run/api/v1/",
                    api_key=os.environ.get("LEPTON_WORKSPACE_TOKEN")
                    or WorkspaceInfoLocalRecord.get_current_workspace_token(),
                    # We will set the connect timeout to be 10 seconds, and read/write
                    # timeout to be 120 seconds, in case the inference server is
                    # overloaded.
                    timeout=httpx.Timeout(connect=10, read=120, write=120, pool=10),
                )
            else:
                # Check if needed to force openai custom server.
                if force_openai:
                    logger.info(f"[FORCE OPENAI SERVER] Using custom LLM model. Remote URL: {os.environ['LLM_REMOTE_OPENAI_URL']}")
                    thread_local.client = openai.OpenAI(
                        base_url=os.environ["LLM_REMOTE_OPENAI_URL"],
                        api_key=os.environ.get("LLM_REMOTE_OPENAI_API_KEY"),
                        timeout=httpx.Timeout(connect=10, read=120, write=120, pool=10),
                    )
                else:
                    logger.info(f"Using custom LLM model. Remote URL: {os.environ['LLM_REMOTE_URL']}")
                    thread_local.client = openai.OpenAI(
                        base_url=os.environ["LLM_REMOTE_URL"],
                        api_key=os.environ.get("LLM_REMOTE_API_KEY"),
                        timeout=httpx.Timeout(connect=10, read=120, write=120, pool=10),
                    )

            return thread_local.client

    def init(self):
        """
        Initializes photon configs.
        """
        # First, log in to the workspace.
        leptonai.api.v0.workspace.login()
        self.backend = os.environ["BACKEND"].upper()
        if self.backend == "LEPTON":
            logger.info("Using Lepton search API.")
            self.leptonsearch_client = Client(
                "https://search-api.lepton.run/",
                token=os.environ.get("LEPTON_WORKSPACE_TOKEN")
                or WorkspaceInfoLocalRecord.get_current_workspace_token(),
                stream=True,
                timeout=httpx.Timeout(connect=10, read=120, write=120, pool=10),
            )
        elif self.backend == "DUCKDUCKGO":
            logger.info("Using DuckDuckGo search API.")
            self.search_function = lambda query: search_with_duckduckgo(
                query,
            )
        elif self.backend == "VECTORSTORE":
            logger.info("Using vectorstore search API.")
            self.search_function = lambda query: search_with_adaptiveRAG(
                query,
            )
        else:
            raise RuntimeError(f"Unknown backend. Please check the environment variable. self.backend: {self.backend}")
        # Set the LLM model to use.
        if os.environ.get("LLM_USE_CUSTOM_SERVER"):
            self.model = os.environ["LLM_REMOTE_MODEL"]
        else:
            self.model = os.environ["LLM_MODEL"]
        # An executor to carry out async tasks, such as uploading to KV.
        self.executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=self.handler_max_concurrency * 2
        )
        # Create the KV to store the search results.
        logger.info("Creating KV. May take a while for the first time.")
        self.kv = os.environ["KV_NAME"]
        with shelve.open(self.kv) as db:
            logger.info(f"KV created/loaded. Current number of keys: {len(db)}")
        # whether we should generate related questions.
        self.should_do_related_questions = to_bool(os.environ["RELATED_QUESTIONS"])

    def get_related_questions(self, query, contexts):
        """
        Gets related questions based on the query and context.
        """

        def ask_related_questions(
            questions: Annotated[
                List[str],
                [(
                    "question",
                    Annotated[
                        str, "related question to the original question and context."
                    ],
                )],
            ]
        ):
            """
            ask further questions that are related to the input and output.
            """
            pass

        try:
            response = self.local_client(force_openai=True).chat.completions.create(
                # Use self.model if it doesn't include deepseek (DeepSeek does not support tool calls)
                model=self.model if "deepseek" not in self.model.lower() else "gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": nettyPrompts._more_questions_prompt.format(
                            context="\n\n".join([c["snippet"] for c in contexts])
                        ),
                    },
                    {
                        "role": "user",
                        "content": query,
                    },
                ],
                tools=[{
                    "type": "function",
                    "function": tool.get_tools_spec(ask_related_questions),
                }],
                max_tokens=512,
            )
            related = response.choices[0].message.tool_calls[0].function.arguments
            # Chain-of-Thought Patching: Remove <think> and </think> tags with everything in between.
            # related = re.sub(r"<think>.*?</think>", "", related)
            if isinstance(related, str):
                related = json.loads(related)
            logger.trace(f"Related questions: {related}")
            return related["questions"]
        except TypeError as e:
            # No related questions found.
            logger.info(f"No related questions found: {e}")
            logger.info(f"Related Questions Response: {response}")
            return []
        except Exception as e:
            # For any exceptions, we will just return an empty list.
            logger.error(
                "Encountered error while generating related questions:"
                f" {e}\n{traceback.format_exc()}"
            )
            return []

    def _raw_stream_response(
        self, contexts, llm_response, related_questions_future
    ) -> Generator[str, None, None]:
        """
        A generator that yields the raw stream response. You do not need to call
        this directly. Instead, use the stream_and_upload_to_kv which will also
        upload the response to KV.
        """
        # First, yield the contexts.
        yield json.dumps(contexts)
        yield "\n\n__LLM_RESPONSE__\n\n"
        # Second, yield the llm response.
        if not contexts:
            # Prepend a warning to the user
            yield (
                "(The search engine returned nothing for this query. Please take the"
                " answer with a grain of salt.)\n\n"
            )
        for chunk in llm_response:
            if chunk.choices:
                yield chunk.choices[0].delta.content or ""
        # Third, yield the related questions. If any error happens, we will just
        # return an empty list.
        if related_questions_future is not None:
            related_questions = related_questions_future.result()
            try:
                result = json.dumps(related_questions)
            except Exception as e:
                logger.error(f"encountered error: {e}\n{traceback.format_exc()}")
                result = "[]"
            yield "\n\n__RELATED_QUESTIONS__\n\n"
            yield result

    def stream_and_upload_to_kv(
        self, contexts, llm_response, related_questions_future, search_uuid
    ) -> Generator[str, None, None]:
        """
        Streams the result and uploads to KV.
        """
        # First, stream and yield the results.
        all_yielded_results = []
        for result in self._raw_stream_response(
            contexts, llm_response, related_questions_future
        ):
            all_yielded_results.append(result)
            yield result
        # Second, store the results in the SQLite database as KV.
        with shelve.open(self.kv) as db:
            db[search_uuid] = all_yielded_results

    @Photon.handler(method="POST", path="/query")
    def query_function(
        self,
        query: str,
        search_uuid: str,
        generate_related_questions: Optional[bool] = True,
    ) -> StreamingResponse:
        """
        Query the search engine and returns the response.

        The query can have the following fields:
            - query: the user query.
            - search_uuid: a uuid that is used to store or retrieve the search result. If
                the uuid does not exist, generate and write to the kv. If the kv
                fails, we generate regardless, in favor of availability. If the uuid
                exists, return the stored result.
            - generate_related_questions: if set to false, will not generate related
                questions. Otherwise, will depend on the environment variable
                RELATED_QUESTIONS. Default: true.
        """
        # Note that, if uuid exists, we don't check if the stored query is the same
        # as the current query, and simply return the stored result. This is to enable
        # the user to share a searched link to others and have others see the same result.

        # ======== KV Storage ========
        if search_uuid:
            try:
                with shelve.open(self.kv) as db:
                    if search_uuid in db:
                        return StreamingResponse(
                            content=db[search_uuid], media_type="text/html"
                        )
                    else:
                        logger.info(f"search_uuid {search_uuid} not found in KV. Generating...")
            except Exception as e:
                logger.error(f"encountered error: {e}\n{traceback.format_exc()}")
                # If the KV fails, we will generate regardless, in favor of availability.

        else:
            raise HTTPException(status_code=400, detail="search_uuid must be provided.")

        # ======== Search Engine Query ========
        contexts = []
        # if questionRouting(query) == "vectorstore":
        contexts = search_with_adaptiveRAG(query)
        logger.info(f"Got {len(contexts)} contexts from the ARAG.")
        search_number = REFERENCE_COUNT - len(contexts)
        # else:
        if self.backend == "LEPTON":
            # delegate to the lepton search api.
            result = self.leptonsearch_client.query(
                query=query,
                search_uuid=search_uuid,
                generate_related_questions=generate_related_questions,
            )
            return StreamingResponse(content=result, media_type="text/html")

        # First, do a search query.
        try:
            query = query or nettyPrompts._default_query
            # Basic attack protection: remove "[INST]" or "[/INST]" from the query
            query = re.sub(r"\[/?INST\]", "", query)
            searched_array = self.search_function(query)
            appended = 0
            for i in searched_array:
                if appended < search_number:
                    appended += 1
                    contexts.append(i)
        except Exception as e:
            logger.error(f"encountered error: {e}\n{traceback.format_exc()}")
            
        # DEBUG: print the contexts.
        logger.debug(f"Contexts: \n{json.dumps(contexts, sort_keys = True, indent = 4)}")

        system_prompt = nettyPrompts._rag_query_text.format(
            context="\n\n".join(
                [f"[[citation:{i+1}]] {c['snippet']}" for i, c in enumerate(contexts)]
            )
        )
        force_gpt = True # Force GPT-4o-mini model
        try:
            client = self.local_client(force_openai=force_gpt)
            llm_response = client.chat.completions.create(
                model=self.model if not force_gpt else "gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": query},
                ],
                max_tokens=1024,
                stop=stop_words,
                stream=True,
                temperature=0.9,
            )
            if self.should_do_related_questions and generate_related_questions:
                # While the answer is being generated, we can start generating
                # related questions as a future.
                related_questions_future = self.executor.submit(
                    self.get_related_questions, query, contexts
                )
            else:
                related_questions_future = None
        except Exception as e:
            logger.error(f"encountered error: {e}\n{traceback.format_exc()}")
            return HTMLResponse("Internal server error.", 503)
        
        if not force_gpt:
            try:
                # Collect the complete streamed answer into a list.
                complete_generation = list(self._raw_stream_response(
                    contexts, llm_response, related_questions_future
                ))
                logger.info("Collected complete generation output.")

                # Combine the list into a single string to pass to the diagram-generation LLM.
                complete_answer_text = "".join(complete_generation)
                
                logger.debug(f"Complete answer text: {complete_answer_text}")
                
                graph_response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "Draw a diagram (e.g., a flowchart or mindmap) based on the generated answer. "
                                "The diagram should be written in Mermaid syntax and enclosed in a markdown code block. "
                                "If the answer is not suitable for diagramming, you may output nothing."
                            )
                        },
                        {
                            "role": "user",
                            "content": f"Here is the generated answer:\n\n{complete_answer_text}\n\nQuery: {query}",
                        },
                    ],
                    max_tokens=1024,
                    stop=stop_words,
                    stream=False,
                    temperature=0.9,
                )
                
                llm_response = [complete_generation, graph_response]
            except Exception as e:
                logger.error(f"encountered error: {e}\n{traceback.format_exc()}")
                return HTMLResponse("Internal server error.", 503)
            
        return StreamingResponse(
            self.stream_and_upload_to_kv(
                contexts, llm_response, related_questions_future, search_uuid
            ),
            media_type="text/html",
        )

    @Photon.handler(mount=True)
    def ui(self):
        return StaticFiles(directory="ui")

    @Photon.handler(mount=True)
    def localData(self):
        return StaticFiles(directory="localData")

    @Photon.handler(method="GET", path="/")
    def index(self) -> RedirectResponse:
        """
        Redirects "/" to the ui page.
        """
        return RedirectResponse(url="/ui/index.html")


if __name__ == "__main__":
    rag = RAG()
    port = 8080 if not PROD_MODE else 80
    rag.launch(host = "0.0.0.0", port = port)
