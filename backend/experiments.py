import logging
from typing import Dict, List, Any
from .preprocessing import DataPreprocessor
from .embedding import EmbeddingGenerator, EmbeddingModelWrapper
from .vector_store.pinecone_store import PineconeStore
from .vector_store.tiledb_store import TileDBStore
from .visualization.plotter import ExperimentPlotter
import time
import numpy as np
from pathlib import Path
import os
from dotenv import load_dotenv
from .vector_store.util.docker_compose_generator import write_docker_compose


# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def calculate_z_score(data: List[float]) -> List[float]:
    """Calculate the z-score for each data point."""
    mean = np.mean(data)
    std_dev = np.std(data)
    return [(x - mean) / std_dev for x in data]

def remove_outliers(data: List[float], z_scores: List[float], threshold: float = 3) -> List[float]:
    """Remove data points with z-scores above the threshold."""
    return [x for x, z in zip(data, z_scores) if abs(z) <= threshold]

class ExperimentRunner:
    def __init__(
        self,
        tiledb_store: TileDBStore,
        pinecone_store: PineconeStore,
        data_preprocessor: DataPreprocessor,
        embedding_generator: EmbeddingGenerator
    ):
        """Initialize the experiment runner."""
        self.pinecone_store = pinecone_store
        self.tiledb_store = tiledb_store
        self.preprocessor = data_preprocessor
        self.embedding_generator = embedding_generator
        self.questions = [
            "What is LangChain?",
            "Can you explain LangChain?",
            "Describe the functionality of LangChain.",
            "What are the main features of LangChain?",
            "How does LangChain work?",
        ]
        self.fixed_dimension = 1536

    def update_pinecone_store(self, host: str, api_key: str):
        """Update the Pinecone store with new connection details."""
        self.pinecone_store = PineconeStore(host=host, api_key=api_key)

    def setup_docker_environment(self, experiment_type: str, dimensions: List[int] = None, num_datasets: int = None):
        """Setup appropriate docker environment based on experiment type."""
        docker_compose_path = "./docker-compose.yaml"  # or wherever your file is located
        
        # Delete existing docker-compose file if it exists
        if os.path.exists(docker_compose_path):
            logger.info(f"Removing existing docker-compose file at {docker_compose_path}")
            os.remove(docker_compose_path)
        
        # Create new docker-compose file
        if experiment_type == "dimension":
            return write_docker_compose(dimensions)
        elif experiment_type == "scalability":
            return write_docker_compose([self.fixed_dimension] * num_datasets)
        elif experiment_type == "retrieval":
            return write_docker_compose([self.fixed_dimension])
        raise ValueError(f"Unknown experiment type: {experiment_type}")

    def measure_retrieval_times(self, single_question=True, questions=None, num_runs=25, threshold=3, dimension=None):
        """Measure retrieval times with multiple runs and outlier removal."""
        if dimension is None:
            dimension = self.fixed_dimension

        # Create embedding model for this dimension
        model_map = {
            384: "all-MiniLM-L6-v2",
            768: "all-distilroberta-v1",
            1024: "all-roberta-large-v1",
            1536: "text-embedding-ada-002"
        }
        embedding_model = EmbeddingModelWrapper(model_map[dimension])
        
        if single_question:
            questions = [self.questions[0]]
        elif questions is None:
            questions = self.questions

        results = {
            "pinecone": {},
            "tiledb": {},
        }

        for question in questions:
            logger.info(f"Testing retrieval for question: {question}")
            
            # Pass the same embedding model to both stores to ensure dimension matching
            pinecone_latencies = []
            pinecone_responses = []
            for _ in range(num_runs):
                response, latency = self.pinecone_store.retrieve(
                    question, 
                    embedding_model=embedding_model.embed_query
                )
                pinecone_latencies.append(latency)
                pinecone_responses.append(response)
            
            # Also update TileDB to use the same embedding model
            tiledb_latencies = []
            tiledb_responses = []
            for _ in range(num_runs):
                response, latency = self.tiledb_store.retrieve(
                    question,
                    embedding_model=embedding_model.embed_query
                )
                tiledb_latencies.append(latency)
                tiledb_responses.append(response)
            
            # Process results with outlier detection
            pinecone_stats = self.process_latencies(pinecone_latencies, threshold)
            results["pinecone"][question] = {
                **pinecone_stats,
                "responses": pinecone_responses
            }
            
            # Process TileDB results with outlier detection
            tiledb_stats = self.process_latencies(tiledb_latencies, threshold)
            results["tiledb"][question] = {
                **tiledb_stats,
                "responses": tiledb_responses
            }

        return results

    def run_dimension_experiment(self, data_path: str, dimensions: List[int] = [384, 768, 1024, 1536]):
        """
        Experiment 1: Compare performance across different vector dimensions.
        Uses a single question for consistent retrieval measurements.
        """
        logger.info("Starting dimension experiment...")
        results = {
            "pinecone": {
                "indexing_stats": {},
                "retrieval_stats": {}
            },
            "tiledb": {
                "indexing_stats": {},
                "retrieval_stats": {}
            },
            "embedding": {}
        }

        # Just pass the directory - preprocessor will handle getting the first file
        texts_dict, metadata_dict = self.preprocessor.preprocess(data_path)
        
        # Use the first document's data
        first_doc_id = list(texts_dict.keys())[0]
        logger.info(f"Using document for dimension experiment: {first_doc_id}")

        # Generate embeddings for this dimension
        embeddings_dict, embedding_times = self.embedding_generator.generate_embeddings_different_models(
            texts_dict[first_doc_id],
            metadata_dict[first_doc_id]
        )
        
        base_port = 5080
        for i, dimension in enumerate(dimensions):
            logger.info(f"Testing dimension: {dimension}")
            
            # Update Pinecone connection for this dimension's index
            port = base_port + i
            self.pinecone_store = PineconeStore(
                host=f"localhost:{port}",
                api_key="pclocal"
            )
            
            # Update TileDB's embedding model for this dimension
            model_map = {
                384: "all-MiniLM-L6-v2",
                768: "all-distilroberta-v1",
                1024: "all-roberta-large-v1",
                1536: "text-embedding-ada-002"
            }
            self.tiledb_store.embedding_model = EmbeddingModelWrapper(model_map[dimension])
            
            # Store embedding times 
            embedding_time_entry = next(
                (entry for entry in embedding_times if entry["dimension"] == dimension),
                None
            )
            if embedding_time_entry:
                results["embedding"][dimension] = embedding_time_entry["embedding_time"]
            else:
                logger.warning(f"No embedding time found for dimension {dimension}")
            
            # Test Pinecone indexing
            pinecone_times = self.pinecone_store.index_embeddings(
                {first_doc_id: embeddings_dict[dimension]},
                {first_doc_id: texts_dict[first_doc_id]}
            )
            # Extract indexing time from the stats
            indexing_time = next(
                (entry["indexing_time"] for entry in pinecone_times 
                 if entry["file_name"] == first_doc_id),
                None
            )
           
            results["pinecone"]["indexing_stats"][dimension] = indexing_time

            
            # Test TileDB indexing
            tiledb_times = self.tiledb_store.index_embeddings(
                {first_doc_id: embeddings_dict[dimension]},
                {first_doc_id: texts_dict[first_doc_id]},
                {first_doc_id: metadata_dict[first_doc_id]}
            )
            # Extract indexing time from the stats
            indexing_time = next(
                (entry["indexing_time"] for entry in tiledb_times 
                 if entry["file_name"] == first_doc_id),
                None
            )
    
            results["tiledb"]["indexing_stats"][dimension] = indexing_time

            
            # Add retrieval time measurements
            logger.info(f"Measuring retrieval times for dimension {dimension}")
            retrieval_results = self.measure_retrieval_times(
                single_question=True,
                dimension=dimension
            )
            
            # Store retrieval stats - just the mean latency
            first_question = self.questions[0]  # We use single_question=True
            results["pinecone"]["retrieval_stats"][dimension] = retrieval_results["pinecone"][first_question]["mean_latency"]
            results["tiledb"]["retrieval_stats"][dimension] = retrieval_results["tiledb"][first_question]["mean_latency"]
            
        return results

    def run_scalability_experiment(self, data_path: str, page_counts: List[int] = [50, 200, 500, 1000, 2000]):
        """
        Experiment 2: Test scalability with different dataset sizes.
        """
        logger.info("Starting scalability experiment...")
        results = {
            "pinecone": {
                "indexing_stats": {},
                "retrieval_stats": {}
            },
            "tiledb": {
                "indexing_stats": {},
                "retrieval_stats": {}
            },
            "embedding": {}
        }

        # Set TileDB's embedding model once since we use fixed dimension
        self.tiledb_store.embedding_model = EmbeddingModelWrapper("text-embedding-ada-002")

        base_port = 5080
        for i, page_count in enumerate(page_counts):
            logger.info(f"Testing with {page_count} pages...")
            
            # Update Pinecone connection for this dataset's index
            port = base_port + i
            self.pinecone_store = PineconeStore(
                host=f"localhost:{port}",
                api_key="pclocal"
            )
            
            # Preprocess subset of data
            texts_dict, metadata_dict = self.preprocessor.preprocess(data_path)
            
            # Generate embeddings
            embeddings_dict, embedding_times = self.embedding_generator.generate_embeddings_multiple_datasets(texts_dict)
            
            # Store embedding times - extract page count from filename and match
            embedding_time = next(
                (entry["embedding_time"] for entry in embedding_times 
                 if entry["file_name"] == f"copy{page_count}.pdf"),
                None
            )
            if embedding_time is not None:
                results["embedding"][page_count] = embedding_time
            else:
                logger.warning(f"No embedding time found for page count {page_count}")
            
            # Test Pinecone indexing
            pinecone_times = self.pinecone_store.index_embeddings(embeddings_dict, metadata_dict)
            # Extract indexing time from the stats
            indexing_time = next(
                (entry["indexing_time"] for entry in pinecone_times 
                 if entry["file_name"] == f"copy{page_count}.pdf"),
                None
            )
            results["pinecone"]["indexing_stats"][page_count] = indexing_time
            
            # Test TileDB indexing
            tiledb_times = self.tiledb_store.index_embeddings(embeddings_dict, texts_dict, metadata_dict)
            # Extract indexing time from the stats
            indexing_time = next(
                (entry["indexing_time"] for entry in tiledb_times 
                 if entry["file_name"] == f"copy{page_count}.pdf"),
                None
            )
            results["tiledb"]["indexing_stats"][page_count] = indexing_time
            
            # Add retrieval time measurements
            logger.info(f"Measuring retrieval times for {page_count} pages")
            retrieval_results = self.measure_retrieval_times(single_question=True)
            
            # Store retrieval stats - just the mean latency
            first_question = self.questions[0]  # We use single_question=True
            results["pinecone"]["retrieval_stats"][page_count] = retrieval_results["pinecone"][first_question]["mean_latency"]
            results["tiledb"]["retrieval_stats"][page_count] = retrieval_results["tiledb"][first_question]["mean_latency"]
            
        return results

    def process_latencies(self, latencies: List[float], threshold: float = 3) -> Dict[str, Any]:
        """Process latencies to remove outliers and calculate statistics."""
        z_scores = calculate_z_score(latencies)
        filtered_latencies = remove_outliers(latencies, z_scores, threshold)
        
        num_removed = len(latencies) - len(filtered_latencies)
        logger.info(f"Removed {num_removed} outlier(s) with z-score above {threshold}")
        
        return {
            "mean_latency": np.mean(filtered_latencies),
            "std_dev_latency": np.std(filtered_latencies),
            "num_runs_kept": len(filtered_latencies),
            "num_runs_removed": num_removed,
            "filtered_latencies": filtered_latencies,
            "raw_latencies": latencies
        }

    def run_retrieval_experiment(self, questions: List[str], num_runs: int = 25, threshold: float = 3):
        """
        Experiment 3: Dedicated retrieval performance experiment.
        Tests multiple questions while keeping other factors constant.
        """
        logger.info("Starting retrieval experiment...")
        retrieval_results = self.measure_retrieval_times(
            single_question=False, 
            questions=questions, 
            num_runs=num_runs, 
            threshold=threshold
        )
        
        # Extract just the mean latency values for each question
        results = {
            "pinecone": {},
            "tiledb": {}
        }
        for question in questions:
            results["pinecone"][question] = retrieval_results["pinecone"][question]["mean_latency"]
            results["tiledb"][question] = retrieval_results["tiledb"][question]["mean_latency"]
        
        return results

def main():
    # Load environment variables from config/.env
    env_path = Path(__file__).parent.parent / "config" / ".env"
    load_dotenv(env_path)
    
    # Set up data path - relative to project root
    project_root = Path(__file__).parent.parent
    data_path = project_root / "data" / "pdfs"
    
    if not data_path.exists():
        raise FileNotFoundError(f"Data directory not found at {data_path}")
    
    # Count PDF files in data directory
    pdf_files = list(data_path.glob("*.pdf"))
    num_datasets = len(pdf_files)
    
    # Initialize components
    preprocessor = DataPreprocessor()
    embedding_generator = EmbeddingGenerator()
    
    # Create experiment runner with minimal initialization
    runner = ExperimentRunner(
        tiledb_store=TileDBStore(index_uri_base="./tiledb_indexes"),
        pinecone_store=PineconeStore(
            api_key="pclocal",
            host="localhost:5080",  # Add initial host
            secure=False
        ),
        data_preprocessor=preprocessor,
        embedding_generator=embedding_generator
    )
    
    # Define experiment parameters
    dimensions_experiment = [384, 768, 1024, 1536]  # Different dimensions for dimension experiment
    fixed_dimension = 1536  # Fixed dimension for scalability experiment
    base_port = 5080
    
    # Choose which experiment to run
    run_dimension_experiment = False  # Set this based on what experiment you want to run
    run_scalability_experiment = False
    run_retrieval_experiment = True
    
    # Create plotter
    plotter = ExperimentPlotter(output_dir="experiment_plots", save_plots=False)
    
    results = {}
    
    if run_dimension_experiment:
        logger.info("Running dimension experiment...")
        docker_compose_path = runner.setup_docker_environment("dimension", dimensions=dimensions_experiment)
        logger.info(f"Generated docker-compose.yaml at {docker_compose_path}")
        
        input("Please run 'docker compose up -d' and press Enter when containers are ready...")
        
        dimension_results = runner.run_dimension_experiment(str(data_path))
        results["dimension"] = dimension_results
        plotter.plot_vs_dimensions(dimension_results)
        
        input("Experiment complete. Please run 'docker compose down' and press Enter to continue...")
    
    if run_scalability_experiment:
        logger.info("Running scalability experiment...")
        docker_compose_path = runner.setup_docker_environment("scalability", None, num_datasets)
        logger.info(f"Generated docker-compose.yaml for scalability experiment at {docker_compose_path}")
        
        input("Please run 'docker compose up -d' and press Enter when containers are ready...")
        logger.info("Starting scalability experiment...")
        
        scalability_results = runner.run_scalability_experiment(str(data_path))
        results["scalability"] = scalability_results
        plotter.plot_vs_dataset_size(scalability_results)
        
        input("Experiment complete. Please run 'docker compose down' and press Enter to continue...")
    
    if run_retrieval_experiment:
        logger.info("Running retrieval experiment...")
        docker_compose_path = runner.setup_docker_environment("retrieval", None, None)
        logger.info(f"Generated docker-compose.yaml for retrieval experiment at {docker_compose_path}")
        
        input("Please run 'docker compose up -d' and press Enter when containers are ready...")
        logger.info("Starting retrieval experiment...")
        
        retrieval_results = runner.run_retrieval_experiment(runner.questions)
        results["retrieval"] = retrieval_results
        plotter.plot_retrieval_comparison(retrieval_results)
        
        input("Experiment complete. Please run 'docker compose down' and press Enter to continue...")
    
    if len(results) > 1:
        logger.info("Creating combined plots...")
        required_keys = {"dimension", "scalability", "retrieval"}
        if not required_keys.issubset(results.keys()):
            logger.warning("Not all experiments were run. Skipping combined plots.")
        else:
            plotter.plot_all_metrics(results)

if __name__ == "__main__":
    main() 