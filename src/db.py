from os import environ
from uuid import uuid4

from chromadb import HttpClient
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

collection = environ.get("CHROMA_COLLECTION")
embeddings = HuggingFaceEmbeddings(model_name=environ.get("MODEL_EMB"))
client = HttpClient(host=environ.get("CHROMA_HOST", 'localhost'), port=environ.get("CHROMA_PORT", "8000"))

db = Chroma(
    collection_name=collection,
    embedding_function=embeddings,
    client=client
)


def add_docs_from_dir(
        directory_path: str,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        glob_pattern: str = "**/*.txt"
):
    """
        Loads text documents from a directory, splits them into smaller chunks,
        and uploads these chunks to a database.

        Parameters:
        - directory_path: The path to the directory containing the text files to be loaded.
        - chunk_size: The maximum size of each chunk of text. Default is 1000 characters.
        - chunk_overlap: Overlap between chunks. Default is 200 characters.
        - glob_pattern: The pattern used to match files in the directory. Default matches all text files.

        Returns:
        - A list of results from the database upload operation.
    """

    loader = DirectoryLoader(directory_path, glob=glob_pattern)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    res = []
    for document in documents:
        chunks = text_splitter.split_documents([document])
        ids = [str(uuid4()) for _ in range(len(chunks))]
        try:
            res += db.add_documents(
                documents=chunks,
                ids=ids
            )
            print(ids, 'are uploaded..')
        except Exception as e:
            print("Can't upload:", ids)
            print('Due to:', e)
    return res
