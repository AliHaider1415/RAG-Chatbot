import os

import torch
# from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
from app.core.config import HUGGINGFACE_API_KEY, MODEL_NAME

# tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=HUGGINGFACE_API_KEY)
# model = AutoModel.from_pretrained(MODEL_NAME, token=HUGGINGFACE_API_KEY)

model = SentenceTransformer("all-MiniLM-L6-v2")

def embed(texts):
    """
    Returns embeddings as list of floats (compatible with Pinecone)
    """
    return model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True  # optional but recommended for similarity
    ).tolist()