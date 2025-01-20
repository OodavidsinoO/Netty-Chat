from dotenv import load_dotenv
load_dotenv(override = True) 

import os
from langchain_openai import ChatOpenAI

from typing import Literal

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

chatLLM = ChatOpenAI(
    api_key = os.environ.get('LLM_REMOTE_API_KEY'),
    model = os.environ.get('LLM_REMOTE_MODEL'),
    base_url = os.environ.get('LLM_REMOTE_URL'),
    temperature = 0,
)

class routeQuery(BaseModel):
    """Route a user query to the most relevant datasource."""
    datasource: Literal["vectorstore", "websearch"] = Field(
        ...,
        description = "Given a user question choose to route it to web search or a vectorstore.",
    )

structuredLLMRouter = chatLLM.with_structured_output(routeQuery)

systemPrompt = """You are an expert at routing a user question to a vectorstore or web search.
The vectorstore contains documents related to computer networks, programming, and cybersecurity.
Use the vectorstore for questions on these topics. Use websearch for everything else."""
routePrompt = ChatPromptTemplate.from_messages(
    [
        ("system", systemPrompt),
        ("human", "{question}"),
    ]
)

questionRouter = routePrompt | structuredLLMRouter

def questionRouting(query: str):
    routing_decision = questionRouter.invoke({"question": query})
    return routing_decision.datasource
