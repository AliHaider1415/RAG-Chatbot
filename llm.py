import streamlit as st
from huggingface_hub import InferenceClient

client = InferenceClient(
    api_key=st.secrets["HUGGINGFACE_API_KEY"],
)

def llm_inference(prompt):
    completion = client.chat.completions.create(
        model="Qwen/Qwen3-Coder-Next:novita",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        max_tokens=300,
        temperature=0.2
    )

    return (completion.choices[0].message.content)

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
    response = llm_inference(prompt)
    
    return response.strip()