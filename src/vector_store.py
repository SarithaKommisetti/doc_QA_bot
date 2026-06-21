import chromadb
from chromadb.utils import embedding_functions
from src.config import GEMINI_API_KEY
from config import (
    DB_PATH,
    COLLECTION_NAME
)

def get_collection():

    client = chromadb.PersistentClient(
        path=DB_PATH
    )

    embedding_function = (
        embedding_functions.DefaultEmbeddingFunction()
    )

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_function
    )

    return collection