from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from search import search
from llm import generate_answer
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    question: str

class ChatResponse(BaseModel):
    answer: str

@app.post("/chat/stream")
async def chat(request: ChatRequest):
    retrieved_chunks = search(request.question)
    context = "\n".join(retrieved_chunks)

    # return {"answer": answer}
    return StreamingResponse(
        generate_answer(context, request.question),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )
