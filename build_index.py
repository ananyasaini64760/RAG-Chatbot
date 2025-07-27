# build_index.py
import pandas as pd
import faiss
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer

# Load and prepare data
df = pd.read_csv("data/loan_data.csv")
docs = df.apply(lambda row: " ".join([f"{col}: {row[col]}" for col in df.columns]), axis=1).tolist()

# Embed using sentence-transformers
model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(docs)

# Create FAISS index
index = faiss.IndexFlatL2(embeddings[0].shape[0])
index.add(np.array(embeddings))

# Save everything
faiss.write_index(index, "model/faiss_index.bin")
with open("model/docs.pkl", "wb") as f:
    pickle.dump(docs, f)
