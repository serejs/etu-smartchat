from langchain_community.document_loaders import DirectoryLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from chromadb import HttpClient

# from src.model import model_path

# model_path = "../../models/Qwen2.5-0.5B-Instruct-Q4_K_M.gguf"

from uuid import uuid4

embeddings = HuggingFaceEmbeddings(
    model_name="DeepPavlov/rubert-base-cased"
)

client = HttpClient(host='86.110.212.181', port=7139)

db = Chroma(
    collection_name="Etu-smartchat-exper01",
    embedding_function=embeddings,
    client=client
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

    res = []
    for document in documents:
        chunks = text_splitter.split_documents([document])
        ids = [str(uuid4()) for _ in range(len(chunks))]
        try:
            print(ids)
            res += db.add_documents(
                documents=chunks,
                ids=ids
            )
            print('ok')
        except Exception as e:
            print(e)
    return res


if __name__ == "__main__":

    print()
