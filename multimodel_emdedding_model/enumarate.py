import fitz  # PyMuPDF
import pdfplumber
import os
import json

PDF_PATH = "bus_ticket.pdf"
OUTPUT_DIR = "./extracted_content"

IMAGE_DIR = os.path.join(OUTPUT_DIR, "images")
os.makedirs(IMAGE_DIR, exist_ok=True)

# ============================================================
# 2. EXTRACT TABLES USING PDFPLUMBER
# ============================================================

def extract_tables(pdf_path):

    results = []

    with pdfplumber.open(pdf_path) as pdf:

        for page_number, page in enumerate(pdf.pages, start=1):

            tables = page.extract_tables()

            for table_index, table in enumerate(tables, start=1):

                if not table:
                    continue

                results.append({
                    "type": "table",
                    "page": page_number,
                    "table_index": table_index,
                    "content": table
                })

    return results

print(extract_tables(PDF_PATH))