import streamlit as st
from retriever import retrieve_context
from generator import generate_answer

st.title("RAG Q&A Chatbot - Loan Dataset")
query = st.text_input("Ask a question about the dataset")

if query:
    context = retrieve_context(query)
    answer = generate_answer(query, context)
    st.markdown("### Answer")
    st.write(answer)
