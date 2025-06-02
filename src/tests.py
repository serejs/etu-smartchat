import pytest
from dotenv import load_dotenv

load_dotenv()

from db import embeddings
import numpy as np

def test_embeddings_creation():
    message = "Привет"
    embeded = embeddings.embed_query(message)
    assert all(i is not None for i in embeded)


def test_embeddings():
    message_1 = "Что такое векторная база данных?"
    message_2 = "Приведи определение векторной базы данных"
    embeded_1 = np.array(embeddings.embed_query(message_1))
    embeded_2 = np.array(embeddings.embed_query(message_2))
    distance = np.dot(embeded_1, embeded_2) / (np.linalg.norm(embeded_1) * np.linalg.norm(embeded_2))
    assert distance >= 0.6