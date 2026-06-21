import chromadb
import google.generativeai as genai
from src.config import GEMINI_API_KEY, DB_PATH, COLLECTION_NAME

genai.configure(api_key=GEMINI_API_KEY)

def get_embedding(text):
    result = genai.embed_content(
        model="models/text-embedding-004",
        content=text
    )
    return result["embedding"]

class GeminiEmbeddingFunction:
    def __call__(self, input):
        return [get_embedding(text) for text in input]

def get_collection():

    client = chromadb.PersistentClient(path=DB_PATH)

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=GeminiEmbeddingFunction()
    )

    return collection