import os
from sentence_transformers import SentenceTransformer
from app.core.config import HUGGINGFACE_API_KEY, MODEL_NAME

_model = None

def get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model

def embed(texts):
    model = get_model()
    return model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True
    ).tolist()