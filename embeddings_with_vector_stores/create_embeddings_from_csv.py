from sentence_transformers import SentenceTransformer
import json
import faiss
import pandas as pd

# Load embedding model
model = SentenceTransformer("google/embeddinggemma-300m")

# Read CSV
df = pd.read_csv("expenses_monthly_2026-07.csv")

# Convert CSV rows into list of dictionaries
documents = df.to_dict(orient="records")

# Convert each dictionary/row into text
texts = [
    ", ".join(f"{key}: {value}" for key, value in doc.items())
    for doc in documents
] # {"Category": "Food", "Amount": 500} - > "Category: Food, Amount: 500"

for doc in documents:
    dict_string = ""
    for key, value in doc.items():
        dict_string += f"{key}: {value},"
        print(dict_string)

# Create embeddings
embeddings = model.encode(texts)

# Create FAISS index
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)

# Add embeddings to FAISS
index.add(embeddings)

# Save FAISS index
faiss.write_index(
    index,
    "document_embeddings_from_csv.faiss"
)

# Save metadata
metadata = {}

for id, doc in enumerate(documents):
    metadata[id] = doc

with open("document_metadata_from_csv.json", "w") as f:
    json.dump(metadata, f, indent=4)

print(
    "Embeddings created and stored in FAISS index "
    "and metadata saved to JSON file successfully."
)