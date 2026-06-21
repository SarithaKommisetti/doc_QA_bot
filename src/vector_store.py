import chromadb
from chromadb.utils.embedding_functions import GoogleGenerativeAiEmbeddingFunction
from src.config import GEMINI_API_KEY, DB_PATH, COLLECTION_NAME

def get_collection():

    client = chromadb.PersistentClient(
        path=DB_PATH
    )

    embedding_function = GoogleGenerativeAiEmbeddingFunction(
        api_key=GEMINI_API_KEY,
        model_name="models/text-embedding-004"
    )

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_function
    )

    return collection