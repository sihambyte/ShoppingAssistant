import os
import logging
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter, Language
from pathlib import Path
from typing import Tuple, Dict, List

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class DataPreprocessor:
    def __init__(self, chunk_size=1000, chunk_overlap=100):
        """
        Initialize the preprocessor with chunking parameters.

        Args:
            chunk_size (int): The size of each text chunk. Defaults to 1000.
            chunk_overlap (int): The overlap between consecutive chunks. Defaults to 100.
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def preprocess_pdfs(self, data_source_path: str) -> Tuple[Dict[str, List[str]], Dict[str, List[Dict]]]:
        """
        Process PDF files and return text chunks and metadata.

        Args:
            data_source_path (str): The path to the directory containing PDF files.

        Returns:
            texts_dict (Dict[str, List[str]]): A dictionary where keys are file names and values are lists of text chunks.
            metadata_dict (Dict[str, List[Dict[str, Any]]]): A dictionary where keys are file names and values are lists of metadata dictionaries.
        """
        documents_dict = {}
        texts_dict = {}
        metadata_dict = {}

        for file_name in os.listdir(data_source_path):
            if file_name.endswith(".pdf"):
                file_path = os.path.join(data_source_path, file_name)

                # Load and split documents
                loader = PyPDFLoader(file_path)
                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap
                )
                documents = loader.load()
                documents = splitter.split_documents(documents)
                documents = [d for d in documents if len(d.page_content) > 5]  # Filter out small chunks

                texts = [d.page_content for d in documents]
                metadata = [d.metadata for d in documents]

                # Store results
                documents_dict[file_name] = documents
                texts_dict[file_name] = texts
                metadata_dict[file_name] = metadata

                logger.info(f"Preprocessed {file_name}: {len(texts)} chunks.")

        return texts_dict, metadata_dict

    def preprocess_markdowns(self, data_source_path):
        """
        Preprocess markdown files from the specified path.
        Returns texts and metadata for all processed markdowns.

        Args:
            data_source_path (str): The path to the directory containing markdown files.

        Returns:
            texts (List[str]): A list of text chunks.
            metadatas (List[Dict[str, Any]]): A list of metadata dictionaries.
        """
        # Create loader and splitter
        loader = GenericLoader.from_filesystem(
            data_source_path, glob="**/*", suffixes=[".mdx"], parser=TextParser()
        )
        splitter = RecursiveCharacterTextSplitter.from_language(
            language=Language.MARKDOWN, chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap
        )

        # Load and split documents
        documents = loader.load()
        logger.info(f"Number of raw documents loaded: {len(documents)}")

        documents = splitter.split_documents(documents)
        documents = [d for d in documents if len(d.page_content) > 5]  # Filter out small chunks

        texts = [d.page_content for d in documents]
        metadatas = [d.metadata for d in documents]
        logger.info(f"Number of document chunks: {len(texts)}")

        return texts, metadatas

    def preprocess(self, data_source_path, file_type="pdf"):
        """
        Preprocess files based on their type ('pdf' or 'markdown').
        Returns texts and metadata.

        Args:
            data_source_path (str): The path to the directory containing files.
            file_type (str): The type of files to preprocess ('pdf' or 'markdown'). Defaults to 'pdf'.

        Returns:
            texts (Dict[str, List[str]] or List[str]): A dictionary or list of text chunks.
            metadata (Dict[str, List[Dict[str, Any]]] or List[Dict[str, Any]]): A dictionary or list of metadata dictionaries.

        Raises:
            ValueError: If the file type is unsupported.
        """
        if file_type.lower() == "pdf":
            return self.preprocess_pdfs(data_source_path)
        elif file_type.lower() == "markdown":
            return self.preprocess_markdowns(data_source_path)
        else:
            logger.error("Unsupported file type. Choose 'pdf' or 'markdown'.")
            raise ValueError("Unsupported file type. Choose 'pdf' or 'markdown'.")