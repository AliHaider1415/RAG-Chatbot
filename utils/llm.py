import functools
from typing import Iterator

from huggingface_hub import InferenceClient, login

from app.core.config import HUGGINGFACE_API_KEY
from app.core.errors import ConfigurationError, LLMError
from app.core.logger import get_logger

logger = get_logger(__name__)


@functools.lru_cache(maxsize=1)
def _get_client() -> InferenceClient:
    if not HUGGINGFACE_API_KEY:
        raise ConfigurationError(
            "HUGGINGFACE_API_KEY is not set in environment. Needed to run the LLM."
        )

    try:
        login(token=HUGGINGFACE_API_KEY)
        client = InferenceClient(api_key=HUGGINGFACE_API_KEY)
        logger.info("LLM client initialized successfully.")
        return client
    except Exception as exc:
        logger.exception("Failed to initialize the LLM client.")
        raise LLMError("Failed to initialize the LLM client.") from exc


def llm_inference(prompt: str) -> Iterator[str]:
    client = _get_client()
    logger.info("Sending prompt to LLM. prompt_length=%s", len(prompt))

    try:
        stream = client.chat.completions.create(
            model="Qwen/Qwen3-Coder-Next:novita",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=300,
            temperature=0.2,
            stream=True,
        )
    except Exception as exc:
        logger.exception("LLM request failed.")
        raise LLMError("LLM request failed.") from exc

    try:
        for chunk in stream:
            # HuggingFace SDK types can vary; handle both dict-like and object-like chunks.
            choices = getattr(chunk, "choices", None) or getattr(chunk, "choices", [])
            if not choices and isinstance(chunk, dict):
                choices = chunk.get("choices") or []
            if not choices:
                continue

            delta = getattr(choices[0], "delta", None) or (
                choices[0].get("delta") if isinstance(choices[0], dict) else None
            )
            if not delta:
                continue

            content = getattr(delta, "content", None)
            if content is None and isinstance(delta, dict):
                content = delta.get("content")

            if content:
                yield content
    except Exception as exc:
        logger.exception("Error while reading streaming chunks from the LLM.")
        raise LLMError("Error while reading streaming chunks from the LLM.") from exc


def generate_answer(context: str, question: str) -> Iterator[str]:
    prompt = f"""
You are an AI Dietician.

You answer questions using ONLY the provided context.
If the answer is not present in the context, say:
"I don't have enough information to answer that."

Context:
{context}

Question:
{question}

Answer:
"""

    try:
        for token in llm_inference(prompt):
            yield f"data: {token}\n\n"

        yield "event: done\ndata: [DONE]\n\n"
    except Exception as e:
        # Surface errors to the SSE client.
        yield f"event: error\ndata: {str(e)}\n\n"

