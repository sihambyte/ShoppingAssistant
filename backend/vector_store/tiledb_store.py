import os
import shutil
import time
from typing import List, Dict, Any, Callable, override
from langchain.schema import Document
from langchain_community.vectorstores import TileDB
from langchain.chains import ConversationalRetrievalChain
from langchain_openai import ChatOpenAI
from .base_store import BaseVectorStore
from backend.embedding import EmbeddingModelWrapper
import numpy as np

class TileDBStore(BaseVectorStore):
    def __init__(self, index_uri_base: str, embedding_model: Any = None):
        """
        Initialize TileDB store with base URI for indexes and embedding model.

        Args:
            index_uri_base (str): Base URI for storing TileDB indexes
            embedding_model: Initial embedding model (will be updated per dimension)
        """
        self.index_uri_base = index_uri_base
        self.embedding_model = embedding_model
        self.db = None

    def get_embedding_model_for_dimension(self, dimension: int) -> Any:
        """Get the appropriate embedding model for a given dimension."""
        model_map = {
            384: "all-MiniLM-L6-v2",
            768: "all-distilroberta-v1",
            1024: "all-roberta-large-v1",
            1536: "text-embedding-ada-002"
        }
        return EmbeddingModelWrapper(model_map[dimension])

    @override
    def index_embeddings(
        self, 
        embeddings: Dict[str, List[Any]], 
        texts_dict: Dict[str, List[str]], 
        metadata_dict: Dict[str, List[Dict[str, Any]]] = None, 
        batch_size: int = 245,
        index_type: str = "IVF_FLAT"
    ) -> List[Dict[str, Any]]:
        """
        Index embeddings using TileDB.
        
        Args:
            embeddings: Dictionary of embeddings by document
            texts_dict: Dictionary of text content by document
            metadata_dict: Optional dictionary of metadata by document
            batch_size: Size of batches for indexing (unused)
            index_type: Type of index to use (default: "IVF_FLAT")
            
        Returns:
            List[Dict[str, Any]]: List of dictionaries containing indexing stats
        """
        indexing_times = []
        
        for doc_id, vectors in embeddings.items():
            print(f"Indexing embeddings for {doc_id}...")
            start_time = time.time()

            # Determine dimension from the first vector
            dimension = len(vectors[0])
            # Get appropriate embedding model for this dimension
            self.embedding_model = self.get_embedding_model_for_dimension(dimension)

            # Create index URI for this document and dimension
            index_uri = f"{self.index_uri_base}_{dimension}D_{doc_id}"
            if os.path.isdir(index_uri):
                shutil.rmtree(index_uri)

            # Create text-embedding pairs
            text_embedding_pairs = [(texts_dict[doc_id][i], vectors[i]) 
                                  for i in range(len(vectors))]

            # Create TileDB index with dimension-specific model
            self.db = TileDB.from_embeddings(
                text_embedding_pairs,
                self.embedding_model,
                index_uri=index_uri,
                index_type=index_type,
                allow_dangerous_deserialization=True,
                metadatas=metadata_dict[doc_id]
            )

            indexing_time = time.time() - start_time
            indexing_times.append({
                "file_name": doc_id,
                "indexing_time": indexing_time,
                "dimension": dimension
            })
            print(f"Indexing completed for {doc_id} in {indexing_time:.2f} seconds.")
        
        return indexing_times

    def ensure_db_loaded(self):
        """Ensure database is loaded before retrieval."""
        if self.db is None:
            self.db = TileDB.load(
                index_uri=self.index_uri_base,
                embedding=self.embedding_model,
                allow_dangerous_deserialization=True
            )

    @override
    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        embedding_model: Callable[[str], List[float]] = None,
        text_key: str = "page_content",
        llm: Any = None,
        **kwargs
    ) -> tuple[str, float]:
        """
        Retrieve response for query using TileDB.

        Args:
            query (str): Query string
            top_k (int): Number of results to return
            embedding_model: Optional embedding model override
            text_key (str): Key for text content in metadata
            llm: Optional language model override
            **kwargs: Additional arguments

        Returns:
            tuple[str, float]: (response, retrieval_time)
        """
        # Ensure database is loaded
        self.ensure_db_loaded()

        if embedding_model is None:
            embedding_model = self.embedding_model

        if llm is None:
            llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

        retriever = self.db.as_retriever(
            search_type="similarity",
            search_kwargs={"k": top_k}
        )

        private_chatgpt = ConversationalRetrievalChain.from_llm(llm, retriever=retriever)

        start_time = time.time()
        response = private_chatgpt.run({'question': query, 'chat_history': ''})
        retrieval_time = time.time() - start_time
        
        return response, retrieval_time
