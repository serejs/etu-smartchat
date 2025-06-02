from os import environ
from uuid import uuid4

from chromadb import HttpClient
from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from yandex_cloud_ml_sdk import YCloudML

import os
import json

sdk = YCloudML(
    folder_id=os.getenv("YANDEX_FOLDER_ID"),
    auth=os.getenv("YANDEX_AUTH")
)

collection_name = environ.get("CHROMA_COLLECTION")
embeddings = sdk.models.text_embeddings("doc")
client = HttpClient(host=environ.get("CHROMA_HOST", 'localhost'), port=environ.get("CHROMA_PORT", "8000"))

collection = client.get_or_create_collection(collection_name)


def add_docs_from_dir(
        directory_path: str,
        docks_metadata_json_path: str,
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

    try:
        with open(docks_metadata_json_path) as f:
            raw_meta = json.load(f)
    except FileNotFoundError:
        print(f"File {docks_metadata_json_path} not found, using empty metadata.")
        raw_meta = {}

    for document in documents:
        name = os.path.splitext(os.path.basename(document.metadata["source"]))[0]
        if name in raw_meta:
            for meta in raw_meta[name]:
                document.metadata[meta] = raw_meta[name][meta]
        chunks = text_splitter.split_documents([document])
        ids = [str(uuid4()) for _ in range(len(chunks))]
        collection.add(
            documents=[document.page_content for document in chunks],
            ids=ids,
            metadatas=[document.metadata for document in chunks],
            embeddings=[embeddings.run(document.page_content) for document in chunks]
        )
