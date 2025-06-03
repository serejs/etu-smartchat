import getopt
import json
import os
import sys
from os import environ
from uuid import uuid4

import numpy as np
from chromadb import EmbeddingFunction, Embeddings, Documents
from chromadb import HttpClient
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from yandex_cloud_ml_sdk import YCloudML

argumentList = sys.argv[1:]
options = "m:s:o:t:e"
long_options = ["meta", "size", "overlap", "text", "env"]
docs_metadata_json_path = None
chunk_size = -1
chunk_overlap = -1
text_dir = None

try:
    # Parsing argument
    arguments, values = getopt.getopt(argumentList, options, long_options)

    # checking each argument
    for currentArgument, currentValue in arguments:
        if currentArgument in ("-m", "--meta"):
            docs_metadata_json_path = currentValue
        if currentArgument in ("-s", "--size"):
            chunk_size = int(currentValue)
        if currentArgument in ("-o", "--overlap"):
            chunk_overlap = int(currentValue)
        if currentArgument in ("-t", "--text"):
            text_dir = currentValue
        if currentArgument in ("-e", "--env"):
            load_dotenv(currentValue)
        else:
            load_dotenv()
except getopt.error as err:
    print(str(err))

if chunk_size <= 0 or chunk_overlap < 0 or chunk_size < chunk_overlap:
    raise ValueError("Invalid chunk size or overlap")

if text_dir is None:
    raise ValueError("Text dir is not set")

sdk = YCloudML(
    folder_id=os.getenv("YANDEX_FOLDER_ID"),
    auth=os.getenv("YANDEX_AUTH")
)

collection_name = os.getenv("CHROMA_COLLECTION")
docs_embeddings = sdk.models.text_embeddings("doc")


class YaEmbeddings(EmbeddingFunction):
    def __call__(self, input: Documents) -> Embeddings:
        return [np.array(docs_embeddings.run(doc)) for doc in input]


print(environ.get("CHROMA_HOST", 'localhost'), environ.get("CHROMA_PORT", "8000"))
client = HttpClient(host=environ.get("CHROMA_HOST", 'localhost'), port=environ.get("CHROMA_PORT", "8000"))
ef = YaEmbeddings()
collection = client.get_or_create_collection(collection_name, embedding_function=ef)

if __name__ == '__main__':
    documents = []
    for filename in os.listdir(text_dir):
        if not filename.endswith('.txt'): continue

        file_path = os.path.join(text_dir, filename)

        with open(file_path, 'r', encoding='utf-8') as file:
            text = '\n'.join(file.readlines())

        documents.append(Document(page_content=text, metadata={"source": text}))

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    raw_meta = {}
    if docs_metadata_json_path is not None:
        try:
            with open(docs_metadata_json_path) as f:
                raw_meta = json.load(f)
        except FileNotFoundError:
            print(f"File {docs_metadata_json_path} not found, using empty metadata.")

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
