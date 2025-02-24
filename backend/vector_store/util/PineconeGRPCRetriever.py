from langchain.schema import BaseRetriever, Document
from typing import List, Callable, Any
import logging

logger = logging.getLogger(__name__)


class PineconeGRPCRetriever(BaseRetriever):
    grpc_index: Any
    embedding_model: Callable[[str], List[float]]
    text_key: str = "page_content"
    tags: List[str] = []
    metadata: dict = {}

    def __init__(self, grpc_index, embedding_model, text_key: str = "page_content", tags: List[str] = None, metadata: dict = None):
        object.__setattr__(self, "grpc_index", grpc_index)
        object.__setattr__(self, "embedding_model", embedding_model)
        object.__setattr__(self, "text_key", text_key)
        object.__setattr__(self, "tags", tags or [])
        object.__setattr__(self, "metadata", metadata or {})

    def _get_relevant_documents(self, query: str) -> List[Document]:
        """Get relevant documents for a query."""
        query_embedding = self.embedding_model(query)
        logger.info(f"PineconeGRPCRetriever - Query embedding dimension: {len(query_embedding)}")
        
        response = self.grpc_index.query(
            vector=query_embedding,
            top_k=5,
            include_metadata=True
        )
        documents = [
            Document(
                page_content=match.metadata.get(self.text_key, ""),
                metadata=match.metadata
            )
            for match in response.matches
        ]
        return documents

    async def _aget_relevant_documents(self, query: str) -> List[Document]:
        return self._get_relevant_documents(query)