import chromadb
from src.config import DB_PATH, COLLECTION_NAME, GEMINI_API_KEY
from google import genai

client = genai.Client(api_key=GEMINI_API_KEY)


def get_embedding(text):
    response = client.models.embed_content(
        model="text-embedding-004",
        contents=text
    )
    return response.embeddings[0].values


import chromadb
from chromadb.utils import embedding_functions
from src.config import DB_PATH, COLLECTION_NAME


def get_collection():
    client = chromadb.PersistentClient(path=DB_PATH)

    embedding_function = (
        embedding_functions.DefaultEmbeddingFunction()
    )

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_function
    )

    return collection