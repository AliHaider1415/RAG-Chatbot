import os

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

from utils.llm import generate_answer
from utils.search import search

PORT = int(os.environ.get("PORT", 8000))

app = FastAPI()

_cors_origins = os.getenv("CORS_ALLOW_ORIGINS", "*")
ALLOW_ORIGINS = (
    ["*"] if _cors_origins.strip() == "*" else [o.strip() for o in _cors_origins.split(",") if o.strip()]
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOW_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    question: str

class ChatResponse(BaseModel):
    answer: str

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.get("/")
async def root():
    return {"service": "rag-chatbot", "status": "ok"}


@app.post("/chat/stream")
async def chat(request: ChatRequest):
    try:
        retrieved_chunks = search(request.question)
        context = "\n".join(retrieved_chunks)

        return StreamingResponse(
            generate_answer(context, request.question),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            },
        )
    
    except Exception as e:
        # If retrieval setup fails, return an SSE-formatted error.
        def error_stream(e=e):
            yield f"event: error\ndata: {str(e)}\n\n"

        return StreamingResponse(
            error_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            },
        )
