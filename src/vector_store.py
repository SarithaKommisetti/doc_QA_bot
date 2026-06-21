import chromadb
import google.generativeai as genai
from src.config import GEMINI_API_KEY, DB_PATH, COLLECTION_NAME

genai.configure(api_key=GEMINI_API_KEY)

# 🔥 MANUAL EMBEDDING FUNCTION (NO CHROMA WRAPPER)
def get_embedding(text):
    response = genai.embed_content(
        model="models/text-embedding-004",
        content=text
    )
    return response["embedding"]

# wrapper class for chromadb
class GeminiEmbeddingFunction:
    def __call__(self, input):
        return [get_embedding(t) for t in input]

def get_collection():
    client = chromadb.PersistentClient(path=DB_PATH)

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=GeminiEmbeddingFunction()
    )

    return collection