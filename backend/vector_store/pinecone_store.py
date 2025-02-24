from abc import ABC
from typing import List, Dict, Any, Callable, override
from pinecone.grpc import PineconeGRPC, GRPCClientConfig
from .base_store import BaseVectorStore
from .util.PineconeGRPCRetriever import PineconeGRPCRetriever
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.chains import ConversationalRetrievalChain
import concurrent.futures
import math
import time
import numpy as np
from typing_extensions import override
from ..embedding import EmbeddingModelWrapper

class PineconeStore(BaseVectorStore, ABC):
    """
    Pinecone implementation of the BaseVectorStore abstract class.
    """
    def __init__(self, api_key: str, host: str = None, secure: bool = False):
        """
        Initialize Pinecone store.

        Args:
            api_key (str): API key for Pinecone
            host (str, optional): Host address for Pinecone index
            secure (bool, optional): Whether to use secure connection
        """
        self.pc = PineconeGRPC(api_key=api_key)
        self.index = self.pc.Index(host=host, grpc_config=GRPCClientConfig(secure=secure))

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
        Index embeddings into Pinecone.
        
        Args:
            embeddings: Dictionary of embeddings by document
            texts_dict: Dictionary of text content by document
            metadata_dict: Optional dictionary of metadata by document (unused)
            batch_size: Size of batches for indexing
            index_type: Type of index to use (unused)
            
        Returns:
            List[Dict[str, Any]]: List of dictionaries containing indexing stats
        """
        indexing_times = []

        for doc_id, embedding_list in embeddings.items():
            print(f"Indexing embeddings for {doc_id}...")
            start_time = time.time()

            # Split vectors into batches
            num_batches = math.ceil(len(embedding_list) / batch_size)
            batch_vectors_list = [
                [(f"{doc_id}-{j}", embedding_list[j], {"page_content": texts_dict[doc_id][j]})
                 for j in range(batch_num * batch_size, min((batch_num + 1) * batch_size, len(embedding_list)))]
                for batch_num in range(num_batches)
            ]

            # Perform parallel upserting
            with concurrent.futures.ThreadPoolExecutor() as executor:
                executor.map(lambda batch_vectors: self.index.upsert(vectors=batch_vectors), batch_vectors_list)

            end_time = time.time()
            indexing_time = end_time - start_time
            
            indexing_times.append({
                "file_name": doc_id,
                "indexing_time": indexing_time
            })
            print(f"Indexing completed for {doc_id} in {indexing_time:.2f} seconds.")

        return indexing_times

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
        """Single retrieval with timing."""
        
        if llm is None:
            llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

        # Create an instance of the custom retriever
        custom_retriever = PineconeGRPCRetriever(
            grpc_index=self.index,
            embedding_model=embedding_model,
            text_key=text_key,
        )

        # Create a conversational retrieval chain
        private_chatgpt = ConversationalRetrievalChain.from_llm(llm, custom_retriever)

        start_time = time.time()
        response = private_chatgpt.run({'question': query, 'chat_history': ''})
        retrieval_time = time.time() - start_time
        return response, retrieval_time