import streamlit as st
from utils import load_chunks, build_index, retrieve, generate_response

st.set_page_config(page_title="Loan RAG Chatbot", layout="wide")
st.title("📊 Loan Approval RAG Chatbot")

csv_path = "data/Training Dataset.csv"

@st.cache_resource
def setup():
    chunks = load_chunks(csv_path)
    index, all_chunks = build_index(chunks)
    return index, all_chunks

index, all_chunks = setup()

query = st.text_input("❓ Ask a question about the loan data:")
if query:
    with st.spinner("Generating answer..."):
        context = retrieve(query, index, all_chunks)
        answer = generate_response(context, query)
        st.markdown("### 💡 Answer:")
        st.success(answer)
        with st.expander("📄 Retrieved Context"):
            st.code(context)
