# 🧠 RAG Q&A Chatbot - Loan Approval Dataset

This project implements a **Retrieval-Augmented Generation (RAG)** chatbot that can answer questions based on a loan approval dataset using both document retrieval (FAISS + Sentence Transformers) and natural language generation (Flan-T5 model).

Chech out: [RAG ChatBot](https://ragloanchatbot-l6mvi8c4krmjrwb6kuepjj.streamlit.app/)

## 📌 Objective

To build an interactive and intelligent chatbot that:
- Retrieves relevant information from a tabular loan dataset
- Answers user questions in natural language using generative AI
- Demonstrates RAG architecture applied to real-world tabular data

---

## ⚙️ How It Works

1. **Retriever**:
   - Converts each row of the dataset into a sentence-like format
   - Embeds these sentences using `sentence-transformers`
   - Uses FAISS to search and retrieve top relevant rows for a user's query

2. **Generator**:
   - Takes the question and retrieved rows
   - Uses `Flan-T5` to generate a coherent and intelligent answer

3. **Streamlit Interface**:
   - Users ask questions through a web UI
   - The model responds with generated answers based on data context
---

## Example Questions:
-	What features impact loan approval the most?
-	How many applicants were married?
-	Do self-employed people get loans approved?
-	What is the average loan amount?
