import functools
from typing import Iterator, Optional

from huggingface_hub import InferenceClient, login

from app.core.config import HUGGINGFACE_API_KEY


@functools.lru_cache(maxsize=1)
def _get_client() -> InferenceClient:
    if not HUGGINGFACE_API_KEY:
        raise RuntimeError(
            "HUGGINGFACE_API_KEY is not set in environment. Needed to run the LLM."
        )

    # Some gated models require an auth token. login() makes the token discoverable.
    login(token=HUGGINGFACE_API_KEY)
    return InferenceClient(api_key=HUGGINGFACE_API_KEY)


def llm_inference(prompt: str) -> Iterator[str]:
    client = _get_client()

    stream = client.chat.completions.create(
        model="Qwen/Qwen3-Coder-Next:novita",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=300,
        temperature=0.2,
        stream=True,
    )

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

