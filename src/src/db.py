from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import LlamaCppEmbeddings
from langchain_chroma import Chroma
from chromadb import HttpClient

from src.model import model_path

from uuid import uuid4

embeddings = LlamaCppEmbeddings(
    model_path=model_path,
    n_ctx=32768,
    n_gpu_layers=40,
    n_batch=512,
    verbose=False
)

db = Chroma(
    collection_name="Etu-smartchat",
    embedding_function=embeddings,
    client=HttpClient(host='localhost', port=8000)
)


def add_docs_from_dir(
        directory_path: str,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        glob_pattern: str = "**/*.txt"
):
    loader = DirectoryLoader(directory_path, glob=glob_pattern)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    chunks = text_splitter.split_documents(documents)
    ids = [str(uuid4()) for _ in range(len(chunks))]

    return db.add_documents(
        documents=chunks,
        ids=ids
    )
