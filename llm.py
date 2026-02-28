from huggingface_hub import InferenceClient
from app.core.config import HUGGINGFACE_API_KEY

client = InferenceClient(
    api_key=HUGGINGFACE_API_KEY,
)

def llm_inference(prompt):

    stream = client.chat.completions.create(
        model="Qwen/Qwen3-Coder-Next:novita",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        max_tokens=300,
        temperature=0.2,
        stream = True
    )

    # return (completion.choices[0].message.content)
    for chunk in stream:
        delta = chunk.choices[0].delta
        if delta and delta.content:
            yield delta.content


def generate_answer(context, question):
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
        yield f"event: error\ndata: {str(e)}\n\n"
    # return response.strip()