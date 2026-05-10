import logging
from typing import List

from pinecone import Pinecone

from app.core.config import PINECONE_API_KEY
from app.core.errors import ConfigurationError, RetrievalError
from app.core.logger import get_logger
from utils.embeddings import embed

logger = get_logger(__name__)
INDEX_NAME = "nutrition-index"


def get_index():
    if not PINECONE_API_KEY:
        raise ConfigurationError("PINECONE_API_KEY is not set in environment.")

    try:
        pc = Pinecone(api_key=PINECONE_API_KEY)
    except Exception as exc:
        logger.exception("Unable to initialize Pinecone client.")
        raise RetrievalError("Unable to initialize Pinecone client.") from exc

    return pc.Index(INDEX_NAME)


def search(query: str, top_k: int = 3) -> List[str]:
    if not query or not query.strip():
        raise ValueError("The search query must not be empty.")

    logger.info("Running search for query=%s top_k=%s", query, top_k)
    index = get_index()

    try:
        q_vec = embed([query])[0]
    except Exception as exc:
        logger.exception("Failed to compute query embeddings.")
        raise RetrievalError("Failed to compute query embeddings.") from exc

    try:
        vector = q_vec.tolist() if hasattr(q_vec, "tolist") and callable(getattr(q_vec, "tolist", None)) else q_vec
        result = index.query(
            vector=vector,
            top_k=top_k,
            include_metadata=True,
        )
    except Exception as exc:
        logger.exception("Vector search failed.")
        raise RetrievalError("Vector search failed.") from exc

    matches = result.get("matches") or []
    texts: List[str] = []
    for m in matches:
        metadata = m.get("metadata") or {}
        text = metadata.get("text")
        if text:
            texts.append(text)
    return texts

