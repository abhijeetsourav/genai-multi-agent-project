# vectorstore.py
# This file can be replaced if you change RAG later

from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings

EMBEDDINGS = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

VECTORSTORE = Chroma(
    persist_directory="./chroma_reviews", embedding_function=EMBEDDINGS
)


def retrieve_reviews(query: str, top_k: int = 8):
    docs = VECTORSTORE.similarity_search(query, k=top_k)
    return docs
