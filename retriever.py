import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer("all-MiniLM-L6-v2")
df = pd.read_csv("data/loan_data.csv")
docs = df.apply(lambda row: " ".join([f"{col}: {row[col]}" for col in df.columns]), axis=1).tolist()
embeddings = model.encode(docs)
index = faiss.IndexFlatL2(embeddings[0].shape[0])
index.add(np.array(embeddings))


def retrieve_context(query, top_k=3):
    query_embedding = model.encode([query])
    _, indices = index.search(np.array(query_embedding), top_k)
    return [docs[i] for i in indices[0]]
