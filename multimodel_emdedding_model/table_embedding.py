import os
import fitz  # PyMuPDF (v1.23.0+)
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# 1. Load Embedding Model
model = SentenceTransformer("all-MiniLM-L6-v2")  


def extract_and_save_tables(pdf_path, output_dir="table_chunks"):
    """
    Detects tables in a PDF, converts them into Markdown syntax,
    saves each table locally as a .txt file, and returns table metadata.
    """
    os.makedirs(output_dir, exist_ok=True)
    doc = fitz.open(pdf_path)
    extracted_tables = []

    for page_num in range(len(doc)):
        page = doc[page_num]

        # Find tables on the current page
        tabs = page.find_tables()

        for tab_idx, tab in enumerate(tabs):
            # Convert table layout into Markdown format
            markdown_table = tab.to_markdown()

            if not markdown_table.strip():
                continue

            # Local filename for the table chunk
            filename = f"page_{page_num + 1}_table_{tab_idx + 1}.txt"
            filepath = os.path.join(output_dir, filename)

            # Save the Markdown table string locally
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(markdown_table)

            extracted_tables.append(
                {
                    "text": markdown_table,
                    "filepath": filepath,
                    "metadata": f"Page {page_num + 1}, Table {tab_idx + 1}",
                }
            )

    return extracted_tables


# 2. Extract and Save Tables
pdf_path = r"C:\Users\Santhosh\OneDrive\Documents\GenAI_Workspace\genai_sessions\multimodel_emdedding_model\sample-tables.pdf"
output_folder = "table_chunks"
extracted_tables = extract_and_save_tables(pdf_path, output_dir=output_folder)

if not extracted_tables:
    print("No structured tables detected in the PDF.")
else:
    print(
        f"Extracted and saved {len(extracted_tables)} tables into '{output_folder}/'."
    )

    # 3. Generate CLIP Embeddings for Table Chunks
    table_texts = [item["text"] for item in extracted_tables]
    table_embeddings = model.encode(
        table_texts, normalize_embeddings=True, convert_to_numpy=True
    ).astype("float32")

    # 4. Populate FAISS Index
    dimension = table_embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)  # Cosine Similarity
    index.add(table_embeddings)
    print(f"Indexed {index.ntotal} table chunks in FAISS.")

    # 5. Execute User Query against Tables
    user_query = "explain me role and actors of table"
    query_embedding = model.encode(
        [user_query], normalize_embeddings=True, convert_to_numpy=True
    ).astype("float32")

    # 6. Retrieve Top Matching Table
    k = min(3, index.ntotal)
    distances, indices = index.search(query_embedding, k)

    # 7. Print Results with Local File Path & Table Preview
    print(
        f"\nTop matching table chunks for: '{user_query}'\n" + "-" * 70
    )
    for rank, (score, idx) in enumerate(zip(distances[0], indices[0]), 1):
        item = extracted_tables[idx]
        print(f"Rank {rank}: Score = {score:.4f} | Path: {item['filepath']}")
        print(f"       Metadata: {item['metadata']}")
        print("       Table Preview:")
        print(item["text"])
        print("-" * 70)