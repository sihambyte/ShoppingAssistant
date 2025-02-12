import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Any
import logging
from pathlib import Path
import os

logger = logging.getLogger(__name__)

class ExperimentPlotter:
    def __init__(self, output_dir: str = "plots", save_plots: bool = True):
        """
        Initialize the plotter with an output directory.

        Args:
            output_dir (str): Directory to save plot files
            save_plots (bool): Whether to save plots to files
        """
        self.output_dir = output_dir
        self.save_plots = save_plots
        if save_plots:
            os.makedirs(output_dir, exist_ok=True)
        self.colors = {
            "pinecone": "#FF6B6B",  # Coral red
            "tiledb": "#4ECDC4"     # Turquoise
        }

    def plot_vs_dimensions(self, results: Dict):
        """Plot dimension experiment metrics in separate plots."""
        dimensions = sorted(list(results["embedding"].keys()))
        
        # Plot 1: Embedding times
        plt.figure(figsize=(12, 6))
        plt.plot(dimensions, 
                [results["embedding"][d] for d in dimensions],
                label="Embedding Time", marker='o')
        plt.xlabel("Dimensions")
        plt.ylabel("Time (seconds)")
        plt.title("Embedding Time vs Dimensions")
        plt.legend()
        plt.grid(True)
        if self.save_plots:
            plt.savefig(f"{self.output_dir}/dimension_embedding.png")
        plt.show()
        
        # Plot 2: Indexing times
        plt.figure(figsize=(12, 6))
        plt.plot(dimensions, 
                [results["pinecone"]["indexing_stats"][d] for d in dimensions],
                label="Pinecone", marker='s', color=self.colors["pinecone"])
        plt.plot(dimensions,
                [results["tiledb"]["indexing_stats"][d] for d in dimensions],
                label="TileDB", marker='^', color=self.colors["tiledb"])
        plt.xlabel("Dimensions")
        plt.ylabel("Time (seconds)")
        plt.title("Indexing Time vs Dimensions")
        plt.legend()
        plt.grid(True)
        if self.save_plots:
            plt.savefig(f"{self.output_dir}/dimension_indexing.png")
        plt.show()
        
        # Plot 3: Retrieval times
        plt.figure(figsize=(12, 6))
        plt.plot(dimensions,
                [results["pinecone"]["retrieval_stats"][d] for d in dimensions],
                label="Pinecone", marker='D', color=self.colors["pinecone"])
        plt.plot(dimensions,
                [results["tiledb"]["retrieval_stats"][d] for d in dimensions],
                label="TileDB", marker='v', color=self.colors["tiledb"])
        plt.xlabel("Dimensions")
        plt.ylabel("Time (seconds)")
        plt.title("Retrieval Time vs Dimensions")
        plt.legend()
        plt.grid(True)
        if self.save_plots:
            plt.savefig(f"{self.output_dir}/dimension_retrieval.png")
        plt.show()

    def plot_vs_dataset_size(self, results: Dict):
        """Plot all metrics vs dataset size on a single plot."""
        page_counts = sorted(list(results["embedding"].keys()))
        
        plt.figure(figsize=(12, 6))
        
        # Embedding times
        plt.plot(page_counts, 
                [results["embedding"][p] for p in page_counts],
                label="Embedding Time", marker='o')
        
        # Indexing times
        plt.plot(page_counts, 
                [results["pinecone"]["indexing_stats"][p] for p in page_counts],
                label="Pinecone Indexing", marker='s', color=self.colors["pinecone"])
        plt.plot(page_counts,
                [results["tiledb"]["indexing_stats"][p] for p in page_counts],
                label="TileDB Indexing", marker='^', color=self.colors["tiledb"])
        
        # Retrieval times
        plt.plot(page_counts,
                [results["pinecone"]["retrieval_stats"][p] for p in page_counts],
                label="Pinecone Retrieval", marker='D', linestyle='--', color=self.colors["pinecone"])
        plt.plot(page_counts,
                [results["tiledb"]["retrieval_stats"][p] for p in page_counts],
                label="TileDB Retrieval", marker='v', linestyle='--', color=self.colors["tiledb"])
        
        plt.xlabel("Dataset Size (pages)")
        plt.ylabel("Time (seconds)")
        plt.title("Performance vs Dataset Size")
        plt.legend()
        plt.grid(True)
        
        if self.save_plots:
            plt.savefig(f"{self.output_dir}/scalability_performance.png")
        plt.show()

    def plot_retrieval_comparison(self, results: Dict):
        """Plot retrieval performance comparison across questions."""
        questions = list(results["pinecone"].keys())
        
        # Get times for each store
        pinecone_times = [results["pinecone"][q] for q in questions]
        tiledb_times = [results["tiledb"][q] for q in questions]
        
        plt.figure(figsize=(12, 6))
        x = range(len(questions))
        
        # Plot bars
        plt.bar([i - 0.2 for i in x], pinecone_times, 0.4, 
                label="Pinecone", color=self.colors["pinecone"])
        plt.bar([i + 0.2 for i in x], tiledb_times, 0.4, 
                label="TileDB", color=self.colors["tiledb"])
        
        plt.xlabel("Questions")
        plt.ylabel("Retrieval Time (seconds)")
        plt.title("Retrieval Performance Comparison")
        plt.xticks(x, [f"Q{i+1}" for i in x], rotation=45)
        plt.legend()
        plt.grid(True)
        
        # Add value labels on bars
        for i, v in enumerate(pinecone_times):
            plt.text(i - 0.2, v, f'{v:.2f}s', ha='center', va='bottom')
        for i, v in enumerate(tiledb_times):
            plt.text(i + 0.2, v, f'{v:.2f}s', ha='center', va='bottom')
        
        plt.tight_layout()
        
        if self.save_plots:
            plt.savefig(f"{self.output_dir}/retrieval_comparison.png")
        plt.show()

    def plot_all_metrics(self, all_results: Dict[str, Any], save: bool = True):
        """Create separate plots for embedding, indexing, and retrieval metrics."""
        
        # Plot 1: Embedding Times
        plt.figure(figsize=(12, 8))
        dimensions = sorted(list(all_results["dimension"]["embedding"].keys()))
        plt.plot(dimensions, 
                [all_results["dimension"]["embedding"][d] for d in dimensions],
                label="Embedding Time", color="#2E86AB", marker='o')
        plt.title("Embedding Time vs Dimensions")
        plt.xlabel("Dimensions")
        plt.ylabel("Time (seconds)")
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        if save:
            plt.savefig(f"{self.output_dir}/embedding_times.png")
        plt.show()
        
        # Plot 2: Indexing Times
        plt.figure(figsize=(12, 8))
        page_counts = sorted(list(all_results["scalability"]["embedding"].keys()))
        for store in ["pinecone", "tiledb"]:
            plt.plot(page_counts, 
                    [all_results["scalability"][store]["indexing_stats"][p] for p in page_counts],
                    label=store.capitalize(),
                    color=self.colors[store],
                    marker='o')
        plt.title("Indexing Time vs Dataset Size")
        plt.xlabel("Dataset Size (pages)")
        plt.ylabel("Time (seconds)")
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        if save:
            plt.savefig(f"{self.output_dir}/indexing_times.png")
        plt.show()
        
        # Plot 3: Retrieval Times
        plt.figure(figsize=(12, 8))
        questions = list(all_results["retrieval"]["pinecone"].keys())
        x = np.arange(len(questions))
        width = 0.35
        
        plt.bar(x - width/2, 
               [all_results["retrieval"]["pinecone"][q] for q in questions],
               width, label="Pinecone", color=self.colors["pinecone"])
        plt.bar(x + width/2, 
               [all_results["retrieval"]["tiledb"][q] for q in questions],
               width, label="TileDB", color=self.colors["tiledb"])
        
        plt.title("Retrieval Time by Question")
        plt.xlabel("Questions")
        plt.ylabel("Time (seconds)")
        plt.xticks(x, [f"Q{i+1}" for i in range(len(questions))], rotation=45)
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Add value labels on bars
        for i, v in enumerate([all_results["retrieval"]["pinecone"][q] for q in questions]):
            plt.text(i - width/2, v, f'{v:.2f}s', ha='center', va='bottom')
        for i, v in enumerate([all_results["retrieval"]["tiledb"][q] for q in questions]):
            plt.text(i + width/2, v, f'{v:.2f}s', ha='center', va='bottom')
        
        plt.tight_layout()
        if save:
            plt.savefig(f"{self.output_dir}/retrieval_times.png")
        plt.show() 