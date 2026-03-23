from typing import List

from pinecone import Pinecone

from app.core.config import PINECONE_API_KEY
from utils.embeddings import embed

INDEX_NAME = "nutrition-index"

def get_index():
    if not PINECONE_API_KEY:
        raise RuntimeError("PINECONE_API_KEY is not set in environment.")

    pc = Pinecone(api_key=PINECONE_API_KEY)
    return pc.Index(INDEX_NAME)


def search(query: str, top_k: int = 3) -> List[str]:
    if not query or not query.strip():
        return []

    index = get_index()
    q_vec = embed([query])[0]

    result = index.query(
        vector=q_vec.tolist(),
        top_k=top_k,
        include_metadata=True,
    )

    matches = result.get("matches") or []
    texts: List[str] = []
    for m in matches:
        metadata = m.get("metadata") or {}
        text = metadata.get("text")
        if text:
            texts.append(text)
    return texts

