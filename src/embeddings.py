import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

PROCESSED_PATH = "data/processed/"
MODEL_NAME = "all-MiniLM-L6-v2"
INDEX_PATH = "models/contracts.index"
DOCS_PATH = "models/contracts_sentences.npy"

# Load embedding model (CPU-friendly)
embedder = SentenceTransformer(MODEL_NAME)

def build_embeddings():
    sentences = []
    sources = []

    # collect all sentences from processed contracts
    for filename in os.listdir(PROCESSED_PATH):
        if filename.endswith("_processed.txt"):
            with open(os.path.join(PROCESSED_PATH,filename), "r",encoding= "utf-8") as f:
                for line in f:
                    line = line.strip()
                     # Skip very short sentences (likely headings)
                    if len(line.split()) < 3:
                        continue
                    if line:
                        sentences.append(line)
                        sources.append(line)

    print(f" Collected {len(sentences)}")

    #convert to embeddings
    embeddings = embedder.encode(sentences, convert_to_numpy = True, show_progress_bar = True)

    # save sentences for later refrence
    np.save(DOCS_PATH, np.array(sentences))

    # Build FAISS index
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    # Save FAISS index
    faiss.write_index(index, INDEX_PATH)

    print(f" FAISS index built and saved with {index.ntotal} entries")

if __name__== "__main__":
    build_embeddings()
