import time
import logging
from typing import Dict, List
from langchain_openai import OpenAIEmbeddings
from sentence_transformers import SentenceTransformer

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class EmbeddingModelWrapper:
    """Wrapper class to provide consistent embed_query interface."""
    def __init__(self, model_name: str):
        if model_name == "text-embedding-ada-002":
            self.model = OpenAIEmbeddings()
        else:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(model_name)
        self.model_name = model_name

    def embed_query(self, text: str) -> List[float]:
        """Embed a single query text."""
        if self.model_name == "text-embedding-ada-002":
            return self.model.embed_query(text)
        else:
            return self.model.encode([text])[0].tolist()

class EmbeddingGenerator:
    def __init__(self):
        """
        Initialize the embedding generator with supported models.
        """
        self.model_map = {
            384: "all-MiniLM-L6-v2",  # Sentence Transformer model
            768: "all-distilroberta-v1",  # Sentence Transformer model
            1024: "all-roberta-large-v1",  # Sentence Transformer model
            1536: "text-embedding-ada-002"  # OpenAI model
        }

    def generate_embeddings_multiple_datasets(self, texts_dict: Dict[str, List[str]]):
        """Generate embeddings in batches to avoid rate limits."""
        embeddings_dict = {}
        embedding_times = []

        for file_name, texts in texts_dict.items():
            logger.info(f"Generating embeddings for {file_name}...")
            try:
                t1 = time.time()
                embedding = OpenAIEmbeddings()
                
                # Process in smaller batches
                batch_size = 50  # Smaller batch size to stay well under limits
                text_embeddings = []
                
                total_batches = (len(texts) + batch_size - 1) // batch_size
                for i in range(0, len(texts), batch_size):
                    batch = texts[i:i + batch_size]
                    logger.info(f"Processing batch {(i//batch_size)+1}/{total_batches} for {file_name}")
                    batch_embeddings = embedding.embed_documents(batch)
                    text_embeddings.extend(batch_embeddings)
                    time.sleep(0.5)  # Longer delay between batches
                
                t2 = time.time()
                embedding_time = t2 - t1
                
                embeddings_dict[file_name] = text_embeddings
                embedding_times.append({
                    "file_name": file_name,
                    "embedding_time": embedding_time
                })

                logger.info(f"Embeddings generated for {file_name} in {embedding_time:.4f} seconds.")
                
            except Exception as e:
                logger.error(f"Error generating embeddings for {file_name}: {e}")
                raise

        return embeddings_dict, embedding_times

    def generate_embeddings_different_models(self, texts: List[str], metadata: List[Dict]):
        """
        Generate embeddings using multiple models with different dimensions.
        """
        embedding_dict = {}
        embedding_times = []

        for dimension, model_name in self.model_map.items():
            logger.info(f"Generating embeddings using model: {model_name} ({dimension}D)...")
            try:
                t1 = time.time()
                if dimension != 1536:
                    embedding_model = SentenceTransformer(model_name)
                    text_embeddings = embedding_model.encode(texts)
                else:
                    embedding_model = OpenAIEmbeddings()
                    text_embeddings = embedding_model.embed_documents(texts)
                t2 = time.time()

                embedding_time = t2 - t1
                logger.info(f"Embeddings generated in {embedding_time:.4f} seconds.")
                
                embedding_dict[dimension] = text_embeddings
                
                embedding_times.append({
                    "dimension": dimension,
                    "embedding_time": embedding_time,
                    "model_name": model_name
                })
            except Exception as e:
                logger.error(f"Error generating embeddings with model {model_name}: {e}")
                raise

        return embedding_dict, embedding_times

    def generate_embeddings_single_model(self, texts: List[str], metadata: List[Dict], model_type="openai"):
        """
        Generate embeddings for a single dataset using one model.
        """
        logger.info(f"Generating embeddings using model: {model_type}...")
        try:
            t1 = time.time()
            if model_type.lower() == "openai":
                embedding_model = OpenAIEmbeddings()
                embeddings = embedding_model.embed_documents(texts)
            elif model_type.lower() in self.model_map.values():
                embedding_model = SentenceTransformer(model_type)
                embeddings = embedding_model.encode(texts)
            else:
                raise ValueError(f"Unsupported model type: {model_type}.")
            t2 = time.time()

            embedding_time = t2 - t1
            
            logger.info(f"Embeddings generated in {embedding_time:.4f} seconds.")
            return embeddings, embedding_time
        except Exception as e:
            logger.error(f"Error generating embeddings with model {model_type}: {e}")
            raise