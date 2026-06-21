import os
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
DB_PATH = "chroma_db"
COLLECTION_NAME = "documents"
print("KEY FOUND:", GEMINI_API_KEY is not None)