import os
import re
import spacy

# load spaCy small English model
nlp = spacy.load("en_core_web_sm")

RAW_PATH = "data/raw/"
PROCESSED_PATH = "data/processed/"

def clean_text(text: str) -> str:
    """
    Basic cleanup: remove extra space, normalize currency symbols, strip text.
    """
    text = re.sub(r"\s+", " ", text)  # collapse multiple spaces/newlines
    text = text.replace("USD", "$").replace("EUR", "€").replace("INR", "₹")
    return text.strip()

def tokenize_sentence(text: str):
    """
    Split contract text into sentences using spaCy.
    """
    doc = nlp(text)
    return [sent.text.strip() for sent in doc.sents if sent.text.strip()]

def preprocess_contracts():
    """
    Process all raw contracts and save them to data/processed
    """
    os.makedirs(PROCESSED_PATH, exist_ok=True)
    for filename in os.listdir(RAW_PATH):
        if filename.endswith(".txt"):
            with open(os.path.join(RAW_PATH, filename), "r", encoding="utf-8") as f:
                raw_text = f.read()

            cleaned = clean_text(raw_text)
            sentences = tokenize_sentence(cleaned)

            out_file = os.path.join(
                PROCESSED_PATH, filename.replace(".txt", "_processed.txt")
            )
            with open(out_file, "w", encoding="utf-8") as out:
                out.write("\n".join(sentences))

            print(f"✅ Processed {filename} -> {out_file}")

if __name__ == "__main__":
    preprocess_contracts()
