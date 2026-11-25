import os
import requests
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")


def get_api_key(name):
    # First check local environment variables / .env
    value = os.getenv(name)

    if value:
        return value

    # Fall back to Streamlit Cloud secrets
    try:
        import streamlit as st
        return st.secrets.get(name)
    except Exception:
        return None


GROQ_KEY = get_api_key("GROQ_API_KEY")
JINA_KEY = get_api_key("JINA_API_KEY")

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
JINA_URL = "https://api.jina.ai/v1/embeddings"

GROQ_MODEL = "openai/gpt-oss-20b"
JINA_MODEL = "jina-embeddings-v4"

CHUNK_SIZE = 800
CHUNK_OVERLAP = 100
TOP_K = 3


# 2. TEXT CHUNKING

def chunk_text(text, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    if size <= overlap:
        raise ValueError("Chunk size must be greater than overlap.")

    chunks = []
    start = 0

    while start < len(text):
        piece = text[start:start + size].strip()

        if piece:
            chunks.append(piece)

        if start + size >= len(text):
            break

        start += size - overlap

    return chunks


# 3. GENERATE EMBEDDINGS

def embed(texts, task):
    if not JINA_KEY:
        raise ValueError("Missing JINA_API_KEY in your .env file.")

    if not texts:
        return []

    response = requests.post(
        JINA_URL,
        headers={
            "Authorization": f"Bearer {JINA_KEY}",
            "Content-Type": "application/json"
        },
        json={
            "model": JINA_MODEL,
            "task": task,
            "input": texts
        },
        timeout=60
    )

    response.raise_for_status()

    data = response.json().get("data", [])

    if len(data) != len(texts):
        raise ValueError("Jina returned an unexpected number of embeddings.")

    data.sort(key=lambda item: item.get("index", 0))

    return [item["embedding"] for item in data]


# 4. BUILD DOCUMENT INDEX

def build_index(text):
    chunks = chunk_text(text)

    if not chunks:
        raise ValueError("The uploaded document contains no usable text.")

    embeddings = embed(chunks, "retrieval.passage")

    if len(chunks) != len(embeddings):
        raise ValueError("Chunk and embedding counts do not match.")

    return chunks, embeddings


# 5. COSINE SIMILARITY

def cosine(a, b):
    if len(a) != len(b):
        raise ValueError("Embedding dimensions do not match.")

    dot = sum(x * y for x, y in zip(a, b))

    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return dot / (norm_a * norm_b)


# 6. RETRIEVE RELEVANT CHUNKS

def retrieve(question, chunks, embeddings, top_k=TOP_K):
    if not chunks or not embeddings:
        return []

    if len(chunks) != len(embeddings):
        raise ValueError("Chunks and embeddings do not match.")

    query_embedding = embed(
        [question],
        "retrieval.query"
    )[0]

    ranked = []

    for chunk, vector in zip(chunks, embeddings):
        score = cosine(query_embedding, vector)
        ranked.append((score, chunk))

    ranked.sort(key=lambda item: item[0], reverse=True)

    return ranked[:top_k]


# 7. GENERATE ANSWER WITH GROQ

def generate_answer(question, ranked):
    if not GROQ_KEY:
        raise ValueError("Missing GROQ_API_KEY in your .env file.")

    if not ranked:
        return "Information not found"

    context = "\n\n".join(
        chunk for _, chunk in ranked
    )

    payload = {
        "model": GROQ_MODEL,
        "messages": [
            {
                "role": "system",
                "content": (
                    'Answer using only the provided context. '
                    'If the answer is absent from the context, '
                    'say exactly "Information not found". '
                    'Do not use outside knowledge or guess.'
                )
            },
            {
                "role": "user",
                "content": (
                    f"Context:\n{context}\n\n"
                    f"Question: {question}"
                )
            }
        ],
        "temperature": 0
    }

    response = requests.post(
        GROQ_URL,
        headers={
            "Authorization": f"Bearer {GROQ_KEY}",
            "Content-Type": "application/json"
        },
        json=payload,
        timeout=60
    )

    response.raise_for_status()

    result = response.json()

    return result["choices"][0]["message"]["content"].strip()
