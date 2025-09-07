import os

# Define folders
folders = [
    "api",
    "notebooks",
    "src",
    "data/raw",
    "data/processed",
    "models"
]

# Define files
files = [
    "api/__init__.py",
    "api/main.py",
    "src/__init__.py",
    "src/preprocessing.py",
    "src/embeddings.py",
    "src/ner.py",
    "src/classifier.py",
    "src/rag.py",
    "src/utils.py",
    "streamlit_app.py",
    "test_app.py",
    "Dockerfile"
]

# Create folders
for folder in folders:
    os.makedirs(folder, exist_ok=True)

# Create empty files
for filepath in files:
    if not os.path.exists(filepath):
        with open(filepath, "w", encoding="utf-8") as f:
            pass

print("✅ Project structure created successfully!")
