from sentence_transformers import SentenceTransformer
from app.core.errors import RetrievalError
from app.core.logger import get_logger

logger = get_logger(__name__)

_model = None

def get_model():
    global _model
    if _model is None:
        try:
            _model = SentenceTransformer("all-MiniLM-L6-v2")
            logger.info("Loaded embedding model successfully.")
        except Exception as exc:
            logger.exception("Failed to load the embedding model.")
            raise RetrievalError("Failed to load the embedding model.") from exc
    return _model

def embed(texts):
    try:
        logger.debug("Computing embeddings for %d texts.", len(texts))
        model = get_model()
        return model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True
        ).tolist()
    except Exception as exc:
        logger.exception("Failed to compute embeddings.")
        raise RetrievalError("Failed to compute embeddings.") from exc