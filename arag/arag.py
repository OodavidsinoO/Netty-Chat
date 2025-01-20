from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import re, os
import torch

from dotenv import load_dotenv
load_dotenv(override = True) 

PATH_MODEL_CACHE = "./arag/modelCache"
PATH_VECTOR_DB = "./arag/chromaVectorStore"
EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
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
    retrieved_docs = chromaVectorStoreRetriever.invoke(query, k=3)

    context = [{
        'name': f"Page {doc.metadata['page'] + 1}, {os.path.basename(doc.metadata['file_path'])}",
        'snippet': doc.page_content,
        'url': 'http://localhost:8000' + re.sub(r'\.\.', '', re.sub(r"\\?\\", "/", doc.metadata['file_path']))
    } for doc in retrieved_docs]
    
    # Notice: Links to the actual files are not working in the current implementation
    # This is because the file paths are not accessible from the current environment
    # The links are generated based on the file paths in the original environment
    
    # TODO: Setup the file server for the actual files to be accessible
    
    unique_context = {entry['snippet']: entry for entry in context}.values()
    unique_context_list = list(unique_context)

    return unique_context_list
