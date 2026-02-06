"""
Vector Store Management for Document Embeddings
"""
from typing import List, Dict, Any, Optional
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.schema import Document
import logging

logger = logging.getLogger(__name__)


class VectorStoreManager:
    """Manage document embeddings and retrieval."""
    
    def __init__(
        self,
        persist_directory: str = "./data/embeddings",
        embedding_model: str = "text-embedding-ada-002"
    ):
        self.persist_directory = persist_directory
        self.embeddings = OpenAIEmbeddings(model=embedding_model)
        self.vectorstore = None
        
    def create_vectorstore(
        self,
        documents: List[Dict[str, Any]]
    ) -> Chroma:
        """Create vector store from documents."""
        try:
            langchain_docs = self._prepare_documents(documents)
            
            self.vectorstore = Chroma.from_documents(
                documents=langchain_docs,
                embedding=self.embeddings,
                persist_directory=self.persist_directory
            )
            
            logger.info(f"Created vector store with {len(langchain_docs)} documents")
            return self.vectorstore
            
        except Exception as e:
            logger.error(f"Error creating vector store: {e}")
            raise
    
    def _prepare_documents(
        self,
        documents: List[Dict[str, Any]]
    ) -> List[Document]:
        """Convert raw documents to LangChain Documents."""
        langchain_docs = []
        
        for idx, doc in enumerate(documents):
            if not doc.get("text") or not doc.get("id"):
                logger.warning(f"Skipping invalid document at index {idx}")
                continue
                
            langchain_docs.append(
                Document(
                    page_content=doc["text"],
                    metadata={
                        "doc_id": doc["id"],
                        "source": doc.get("source", "unknown"),
                        "index": idx
                    }
                )
            )
        
        return langchain_docs
    
    def similarity_search(
        self,
        query: str,
        k: int = 4
    ) -> List[Document]:
        """Search for similar documents."""
        if not self.vectorstore:
            raise ValueError("Vector store not initialized")
        
        return self.vectorstore.similarity_search(query, k=k)
