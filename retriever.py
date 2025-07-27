import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer

# Load precomputed data
model = SentenceTransformer("all-MiniLM-L6-v2")
index = faiss.read_index("model/faiss_index.bin")
with open("model/docs.pkl", "rb") as f:
    docs = pickle.load(f)

def retrieve_context(query, top_k=3):
    query_embedding = model.encode([query])
    _, indices = index.search(np.array(query_embedding), top_k)
    return [docs[i] for i in indices[0]]
