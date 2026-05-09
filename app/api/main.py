import os

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, validator

from app.core.errors import ServiceError
from app.core.logger import get_logger
from utils.search import search
from utils.llm import generate_answer

PORT = int(os.environ.get("PORT", 8000))

logger = get_logger("rag_chatbot")
logger.info("Starting RAG chatbot service with port %s", PORT)

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


def _sse_error_response(message: str, status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR) -> StreamingResponse:
    logger.debug("Returning SSE error response: %s", message)

    def error_stream():
        yield f"event: error\ndata: {message}\n\n"

    return StreamingResponse(
        error_stream(),
        media_type="text/event-stream",
        status_code=status_code,
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )


@app.exception_handler(ServiceError)
async def service_error_handler(request: Request, exc: ServiceError):
    logger.exception("Service error: %s", exc)
    return JSONResponse(
        status_code=status.HTTP_502_BAD_GATEWAY,
        content={"error": str(exc)},
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    logger.warning("Validation failed: %s", exc)
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={"error": "Request validation failed.", "details": exc.errors()},
    )


class ChatRequest(BaseModel):
    question: str

    @validator("question")
    def question_must_not_be_empty(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("The question field must not be empty.")
        return value


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
    except ValueError as exc:
        logger.warning("Validation error in chat stream: %s", exc)
        return _sse_error_response(str(exc), status.HTTP_422_UNPROCESSABLE_ENTITY)
    except ServiceError as exc:
        logger.exception("Service failure in chat stream: %s", exc)
        return _sse_error_response(str(exc), status.HTTP_502_BAD_GATEWAY)
    except Exception as exc:
        logger.exception("Unexpected error in chat stream: %s", exc)
        return _sse_error_response("Internal server error.")
