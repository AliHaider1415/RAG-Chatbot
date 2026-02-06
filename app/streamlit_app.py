import sys
import os

# Add project root directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
from search import search
from llm import generate_answer
st.title("AI Dietician Chatbot")

question = st.text_area("Ask your nutrition question:")

if st.button("Get Answer"):
    if not question.strip():
        st.warning("Please enter a question!")
    else:
        with st.spinner("Thinking..."):
            # Retrieve context from Pinecone
            retrieved_chunks = search(question)
            context = "\n".join(retrieved_chunks)
            
            # Generate answer from LLM
            answer = generate_answer(context, question)
        
        st.markdown("### Answer:")
        st.write(answer)
