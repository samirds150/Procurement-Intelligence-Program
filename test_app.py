''' with open("data/processed/contract1_processed.txt", "r", encoding="utf-8") as f:
    lines = f.readlines()

print(f"Total Sentences: {len(lines)}")
print("First 5 sentences:")
for sent in lines[:5]:
    print("-", sent.strip())'''

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

MODEL_NAME = "all-MiniLM-L6-v2"
INDEX_PATH = "models/contracts.index"
DOCS_PATH = "models/contracts_sentences.npy"

# Load model, index, and sentences
embedder = SentenceTransformer(MODEL_NAME)
index = faiss.read_index(INDEX_PATH)
sentences = np.load(DOCS_PATH, allow_pickle=True)

# Try a query
query = "What are the payment terms?"
q_emb = embedder.encode([query], convert_to_numpy=True)

D, I = index.search(q_emb, k=3)

print("🔍 Query:", query)
print("\nTop Matches:")
for idx in I[0]:
    print("-", sentences[idx])

