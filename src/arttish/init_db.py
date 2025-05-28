from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import LlamaCppEmbeddings
from langchain_chroma import Chroma
from model import model_path
from chromadb import EphemeralClient

import os
from uuid import uuid4

llama_embeddings = LlamaCppEmbeddings(
    model_path=model_path,
    n_ctx=32768,
    n_gpu_layers=40,
    n_batch=512,
    verbose=False,
)

db = Chroma("LessonsCollection", embedding_function=llama_embeddings, client=EphemeralClient())


def parse_file_name(file_path):
    base_name = os.path.basename(file_path)
    return os.path.splitext(base_name)[0]


def add_to_existing_chroma(
        directory_path: str,
        chroma_client: Chroma,
        embeddings,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        glob_pattern: str = "**/*.txt"
):
    """
    Add documents to an existing Chroma collection using Llama.cpp embeddings

    Args:
        directory_path: Path to directory containing documents
        chroma_client: Pre-configured Chroma client instance
        embeddings: Initialized LlamaCppEmbeddings instance
        chunk_size: Size of text chunks (default: 1000)
        chunk_overlap: Overlap between chunks (default: 200)
        glob_pattern: File pattern to match (default: .txt files)
    """
    # Load documents with specified pattern
    loader = DirectoryLoader(directory_path, glob=glob_pattern)
    documents = loader.load()

    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    chunks = text_splitter.split_documents(documents)
    ids = [str(uuid4()) for _ in range(len(chunks))]

    # Add documents to existing collection
    return chroma_client.add_documents(
        documents=chunks,
        embedding=embeddings,
        ids=ids
    )


if __name__ == "__main__":
    print(add_to_existing_chroma(directory_path='./text_dir', chroma_client=db, embeddings=llama_embeddings))
