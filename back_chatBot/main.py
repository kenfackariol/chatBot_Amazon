from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import ast
import numpy as np
import faiss
import os
from sentence_transformers import SentenceTransformer


app = FastAPI(title="RAG Chatbot API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:4200",
        "http://127.0.0.1:4200"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

INDEX_FILE = "faiss.index"
EMBEDDINGS_FILE = "embeddings.npy"
DATA_FILE = "qa_Baby.json"


# Charger modèle

print("Chargement du modèle...")
model = SentenceTransformer("all-MiniLM-L6-v2")
print("Modèle chargé ")


# Charger dataset

rows = []
with open(DATA_FILE, "r", encoding="utf-8", errors="ignore") as f:
    for line in f:
        if line.strip():
            rows.append(ast.literal_eval(line))

data = pd.DataFrame(rows)
data = data.dropna(subset=["question", "answer"])

data["text"] = (
    "product category: baby. "
    "question: " + data["question"].astype(str) +
    " answer: " + data["answer"].astype(str)
).str.lower()


# FAISS — load ou build

if os.path.exists(INDEX_FILE) and os.path.exists(EMBEDDINGS_FILE):
    print(" Chargement FAISS depuis disque...")
    index = faiss.read_index(INDEX_FILE)
    doc_embeddings = np.load(EMBEDDINGS_FILE)
    print("FAISS chargé instantanément ")

else:
    print(" Création FAISS (une seule fois)...")

    doc_embeddings = model.encode(
        data["text"].tolist(),
        convert_to_numpy=True,
        show_progress_bar=True
    )

    dim = doc_embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(doc_embeddings)

    faiss.write_index(index, INDEX_FILE)
    np.save(EMBEDDINGS_FILE, doc_embeddings)

    print(" FAISS sauvegardé sur disque")

print("Nombre de vecteurs :", index.ntotal)


#  Recherche

def retrieve(query: str, top_k=5):
    query_embedding = model.encode([query], convert_to_numpy=True)
    distances, indices = index.search(query_embedding, top_k)
    return data.iloc[indices[0]]

def build_prompt(query, docs):
    context = "\n".join("- " + row["answer"] for _, row in docs.iterrows())
    return f"""
You are a helpful e-commerce assistant.

Context:
{context}

User question:
{query}

Answer concisely.
"""


# FastAPI

app = FastAPI(title="RAG Chatbot API")

class Query(BaseModel):
    question: str
    top_k: int = 5

@app.post("/ask")
def ask(query: Query):
    print(query)
    docs = retrieve(query.question, query.top_k)
    prompt = build_prompt(query.question, docs)
    return {"response": prompt}

@app.get("/")
def root():
    return {"message": "RAG API running "}
