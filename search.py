import os
from dotenv import load_dotenv
from pinecone import Pinecone
from embeddings import embed
from llm import generate_answer

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = "nutrition-index"

pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)

def search(query):
    q_vec = embed([query])[0]
    result = index.query(
        vector=q_vec.tolist(),
        top_k=3,
        include_metadata=True
    )
    return [m["metadata"]["text"] for m in result["matches"]]