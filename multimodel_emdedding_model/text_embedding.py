import os
import pdfplumber
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# 1. Load CLIP Model
model = SentenceTransformer("sentence-transformers/clip-ViT-B-32")


def table_to_markdown(table):
    """
    Converts a 2D list (extracted by pdfplumber) into a clean Markdown table string.
    """
    if not table or not any(table):
        return ""

    # Clean None values and internal newline characters in cell contents
    cleaned_table = []
    for row in table:
        cleaned_row = [
            str(cell).strip().replace("\n", " ") if cell is not None else ""
            for cell in row
        ]
        if any(cleaned_row):  # Skip completely empty rows
            cleaned_table.append(cleaned_row)

    if not cleaned_table:
        return ""

    headers = cleaned_table[0]
    rows = cleaned_table[1:]

    # Construct Markdown Syntax
    header_line = "| " + " | ".join(headers) + " |"
    separator_line = "| " + " | ".join(["---"] * len(headers)) + " |"
    data_lines = ["| " + " | ".join(row) + " |" for row in rows]

    return "\n".join([header_line, separator_line] + data_lines)


def extract_and_save_tables_pdfplumber(pdf_path, output_dir="pdfplumber_tables"):
    """
    Extracts tables using pdfplumber, converts them to Markdown,
    saves each table locally as a .txt file, and returns table metadata.
    """
    os.makedirs(output_dir, exist_ok=True)
    extracted_tables = []

    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages):
            # Extract raw tables (returns 2D lists)
            tables = page.extract_tables()

            for tab_idx, table in enumerate(tables):
                markdown_table = table_to_markdown(table)

                if not markdown_table.strip():
                    continue

                filename = f"page_{page_num + 1}_table_{tab_idx + 1}.txt"
                filepath = os.path.join(output_dir, filename)

                # Save local table chunk
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
pdf_path = r"C:\Users\Santhosh\OneDrive\Documents\GenAI_Workspace\genai_sessions\multimodel_emdedding_model\Population- india.pdf"
output_folder = "pdfplumber_tables"
extracted_tables = extract_and_save_tables_pdfplumber(
    pdf_path, output_dir=output_folder
)

if not extracted_tables:
    print("No structured tables detected in the PDF by pdfplumber.")
else:
    print(
        f"Extracted and saved {len(extracted_tables)} tables into '{output_folder}/'."
    )

    # 3. Batch Generate Embeddings
    table_texts = [item["text"] for item in extracted_tables]
    table_embeddings = model.encode(
        table_texts, normalize_embeddings=True, convert_to_numpy=True
    ).astype("float32")

    # 4. Populate FAISS Index
    dimension = table_embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)  # Cosine Similarity
    index.add(table_embeddings)
    print(f"Indexed {index.ntotal} table chunks in FAISS.")

    # 5. Execute User Query
    user_query = "cancellation charges and percentage"
    query_embedding = model.encode(
        [user_query], normalize_embeddings=True, convert_to_numpy=True
    ).astype("float32")

    # 6. Retrieve Top Match
    k = min(3, index.ntotal)
    distances, indices = index.search(query_embedding, k)

    # 7. Print Results with Local File Path
    print(
        f"\nTop matching table chunks for: '{user_query}'\n" + "-" * 70
    )
    for rank, (score, idx) in enumerate(zip(distances[0], indices[0]), 1):
        item = extracted_tables[idx]
        print(f"Rank {rank}: Score = {score:.4f} | Path: {item['filepath']}")
        print(f"       Metadata: {item['metadata']}")
        print("       Table Preview:\n" + item["text"])
        print("-" * 70)