from abc import ABC, abstractmethod
from typing import List, Dict, Any

class BaseVectorStore(ABC):
    """
    Abstract base class for vector stores.
    """
    @abstractmethod
    def index_embeddings(
        self, 
        embeddings: Dict[str, List[Any]], 
        texts_dict: Dict[str, List[str]], 
        metadata_dict: Dict[str, List[Dict[str, Any]]] = None, 
        batch_size: int = 245,
        index_type: str = "IVF_FLAT"
    ) -> List[Dict[str, Any]]:
        """
        Index embeddings into the vector store.
        
        Args:
            embeddings: Dictionary of embeddings by document
            texts_dict: Dictionary of text content by document
            metadata_dict: Optional dictionary of metadata by document
            batch_size: Size of batches for indexing (used by Pinecone)
            index_type: Type of index to use (used by TileDB, default: "IVF_FLAT")
            
        Returns:
            List[Dict[str, Any]]: List of dictionaries containing indexing stats
            Each dict should have at least:
                - file_name: str
                - indexing_time: float
                - dimension: int
        """
        pass

    @abstractmethod
    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        embedding_model: Any = None,
        **kwargs
    ) -> tuple[str, float]:
        """
        Abstract method to retrieve the most similar embeddings based on a query.

        Args:
            query (str): The query string
            top_k (int): Number of results to return
            embedding_model: Model to use for query embedding
            **kwargs: Additional arguments

        Returns:
            tuple[str, float]: (response, retrieval_time)
        """
        pass
