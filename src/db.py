from os import environ

import numpy as np
from chromadb import EmbeddingFunction, Embeddings, Documents
from chromadb import HttpClient

from model import sdk

collection_name = environ.get("CHROMA_COLLECTION")
docs_embeddings = sdk.models.text_embeddings("doc")


class YaEmbeddings(EmbeddingFunction):
    def __call__(self, input: Documents) -> Embeddings:
        return [np.array(docs_embeddings.run(doc)) for doc in input]


client = HttpClient(host=environ.get("CHROMA_HOST", 'localhost'), port=environ.get("CHROMA_PORT", "8000"))
ef = YaEmbeddings()
collection = client.get_or_create_collection(collection_name, embedding_function=ef)
