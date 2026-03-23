import os

import torch
from transformers import AutoTokenizer, AutoModel
from app.core.config import HUGGINGFACE_API_KEY, MODEL_NAME

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=HUGGINGFACE_API_KEY)
model = AutoModel.from_pretrained(MODEL_NAME, token=HUGGINGFACE_API_KEY)

def embed(texts):
    inputs = tokenizer(
        texts,
        padding=True,
        truncation=True,
        return_tensors="pt"
    )

    with torch.no_grad():
        outputs = model(**inputs)

    # Mean pooling
    embeddings = outputs.last_hidden_state.mean(dim=1)
    return embeddings.cpu().numpy()