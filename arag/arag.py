from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import re, os
import torch

from dotenv import load_dotenv
load_dotenv(override = True) 

PATH_MODEL_CACHE = "./arag/modelCache"
PATH_VECTOR_DB = "./arag/chromaVectorStore"
EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"

MAGIC_NUMBER = 0.75
# Cosine distance threshold, if content below this threshold,
# it is considered as similar and therefore used as context

GPU_available = torch.cuda.is_available()

# Check if GPU is available
if GPU_available:
    print("[INFO] GPU is available")
else:
    print("[INFO] GPU is not available. Using CPU instead")

embeddingModel = HuggingFaceEmbeddings(
    model_name = EMBEDDING_MODEL,
    model_kwargs = { "device": "cuda" if GPU_available else "cpu", "trust_remote_code": True },
    encode_kwargs = { "normalize_embeddings": True },
    cache_folder = PATH_MODEL_CACHE,
)

chromaVectorStore = Chroma(
    collection_name = "nettyRAG",
    embedding_function = embeddingModel,
    persist_directory = PATH_VECTOR_DB
)
chromaVectorStoreRetriever = chromaVectorStore.as_retriever()

def get_rag_context(query: str):
    retrieved_docs = chromaVectorStore.similarity_search_with_score(query)

    context = [{
        'name': f"Page {doc[0].metadata['page'] + 1}, {os.path.basename(doc[0].metadata['file_path'])}",
        'snippet': doc[0].page_content,
        'url': re.sub(r'\.\.', '', re.sub(r"\\?\\", "/", doc[0].metadata['file_path'])) + f"#page={doc[0].metadata['page'] + 1}"
    } for doc in retrieved_docs if doc[-1] < MAGIC_NUMBER]
    
    unique_context = {entry['snippet']: entry for entry in context}.values()
    unique_context_list = list(unique_context)

    return unique_context_list
