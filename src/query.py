from google import genai
from src.config import GEMINI_API_KEY

client = genai.Client(api_key=GEMINI_API_KEY)
from dotenv import load_dotenv
from src.vector_store import get_collection

load_dotenv()


def query_rag(user_query, k=3):

    collection = get_collection()

    results = collection.query(
        query_texts=[user_query],
        n_results=k
    )

    context_blocks = []

    for doc, meta in zip(
        results["documents"][0],
        results["metadatas"][0]
    ):

        source = meta["source"]
        page = meta["page"]

        context_blocks.append(
            f"[Source: {source}, Page: {page}]\n{doc}"
        )

    context = "\n\n".join(context_blocks)

    prompt = f"""
You are a document question-answering assistant.

Use ONLY the context provided below.

If the answer is not present in the context, say:
"I cannot find the answer in the provided documents."

CONTEXT:
{context}

QUESTION:
{user_query}

ANSWER:
"""

    response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents=prompt
)

    return response.text


if __name__ == "__main__":

    while True:
        q = input("Ask a question (or type exit): ")

        if q.lower() == "exit":
            break

        print(query_rag(q))