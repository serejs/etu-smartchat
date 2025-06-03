from os import environ
from uuid import uuid4

import numpy as np
from chromadb import HttpClient
from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from model import sdk

import os
import json

from chromadb import EmbeddingFunction, Embeddings, Documents

collection_name = environ.get("CHROMA_COLLECTION")
docs_embeddings = sdk.models.text_embeddings("doc")


class YaEmbeddings(EmbeddingFunction):
    def __call__(self, input: Documents) -> Embeddings:
        return [np.array(docs_embeddings.run(doc)) for doc in input]


client = HttpClient(host=environ.get("CHROMA_HOST", 'localhost'), port=environ.get("CHROMA_PORT", "8000"))
ef = YaEmbeddings()
collection = client.get_or_create_collection(collection_name, embedding_function=ef)


def add_docs_from_dir(
        directory_path: str,
        docks_metadata_json_path: str = None,
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
    raw_meta = {}
    if docks_metadata_json_path is not None:
        try:
            with open(docks_metadata_json_path) as f:
                raw_meta = json.load(f)
        except FileNotFoundError:
            print(f"File {docks_metadata_json_path} not found, using empty metadata.")

    for document in documents:
        name = os.path.splitext(os.path.basename(document.metadata["source"]))[0]
        name = [key for key in raw_meta.keys() if key in name]
        if len(name) != 0:
            name = name[0]
            document.metadata["source"] = name
            for meta in raw_meta[name]:
                document.metadata[meta] = raw_meta[name][meta]
        chunks = text_splitter.split_documents([document])
        ids = [str(uuid4()) for _ in range(len(chunks))]
        collection.add(
            documents=[chunk.page_content for chunk in chunks],
            ids=ids,
            metadatas=[chunk.metadata for chunk in chunks],
        )
