import os
from pypdf import PdfReader
from src.vector_store import get_collection

# -----------------------------
# PDF EXTRACTION
# -----------------------------
def extract_pdf_pages(file_path):
    extracted = []

    try:
        reader = PdfReader(file_path)

        for page_num, page in enumerate(reader.pages):
            text = page.extract_text()

            if text and text.strip():
                extracted.append({
                    "text": " ".join(text.split()),
                    "metadata": {
                        "source": os.path.basename(file_path),
                        "page": page_num + 1
                    }
                })

    except Exception as e:
        print(f"Error reading {file_path}: {e}")

    return extracted


# -----------------------------
# TXT EXTRACTION
# -----------------------------
def extract_txt(file_path):
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()

        return [{
            "text": text,
            "metadata": {
                "source": os.path.basename(file_path),
                "page": 1
            }
        }]

    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return []


# -----------------------------
# CHUNKING
# -----------------------------
def chunk_documents(documents, chunk_size=1000, chunk_overlap=200):
    chunks = []

    for doc in documents:
        text = doc["text"]
        metadata = doc["metadata"]

        start = 0

        while start < len(text):
            end = min(start + chunk_size, len(text))

            chunks.append({
                "text": text[start:end],
                "metadata": {
                    **metadata,
                    "chunk_start": start,
                    "chunk_end": end
                }
            })

            start += (chunk_size - chunk_overlap)

    return chunks


# -----------------------------
# LOAD ALL FILES
# -----------------------------
def load_documents(data_folder):
    all_docs = []

    for filename in os.listdir(data_folder):
        file_path = os.path.join(data_folder, filename)

        if filename.endswith(".pdf"):
            all_docs.extend(extract_pdf_pages(file_path))

        elif filename.endswith(".txt"):
            all_docs.extend(extract_txt(file_path))

    return all_docs


# -----------------------------
# SAVE TO CHROMA (FIXED)
# -----------------------------
def save_chunks_to_db(chunks):

    collection = get_collection()

    ids = [f"chunk_{i}" for i in range(len(chunks))]

    documents = [chunk["text"] for chunk in chunks]

    metadatas = [chunk["metadata"] for chunk in chunks]

    # ⭐ FIX: GENERATE EMBEDDINGS MANUALLY

    collection.add(
        ids=ids,
        documents=documents,
        metadatas=metadatas,
    )

    print(f"Stored {len(chunks)} chunks in ChromaDB.")


# -----------------------------
# MAIN
# -----------------------------
if __name__ == "__main__":

    docs = load_documents("data")

    print("\nLoaded Documents:\n")
    print(docs)

    print(f"\nLoaded {len(docs)} document(s)")

    chunks = chunk_documents(docs)

    print(f"Created {len(chunks)} chunk(s)")

    save_chunks_to_db(chunks)

    if chunks:
        print("\nSample Chunk:\n")
        print(chunks[0]["text"][:300])
    else:
        print("\nNo chunks created!")