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
