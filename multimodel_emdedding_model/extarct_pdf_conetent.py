import fitz  # PyMuPDF
import pdfplumber
import os
import json


PDF_PATH = "bus_ticket.pdf"
OUTPUT_DIR = "./extracted_content"

IMAGE_DIR = os.path.join(OUTPUT_DIR, "images")
os.makedirs(IMAGE_DIR, exist_ok=True)


# ============================================================
# 1. EXTRACT TEXT + IMAGES USING PYMUPDF
# ============================================================

def extract_text_and_images(pdf_path):

    results = []

    doc = fitz.open(pdf_path) 

    for page_number, page in enumerate(doc, start=1):

        # ----------------------------------------------------
        # Extract text
        # ----------------------------------------------------
        text = page.get_text("text")

        if text.strip():
            results.append({
                "type": "text",
                "page": page_number,
                "content": text.strip()
            })

        # ----------------------------------------------------
        # Extract images
        # ----------------------------------------------------
        images = page.get_images(full=True)

        for image_index, image in enumerate(images, start=1): 

            xref = image[0]

            image_info = doc.extract_image(xref)

            image_bytes = image_info["image"]
            image_ext = image_info["ext"]

            image_filename = (
                f"page_{page_number}_image_{image_index}.{image_ext}"
            )

            image_path = os.path.join(
                IMAGE_DIR,
                image_filename
            )

            with open(image_path, "wb") as image_file:
                image_file.write(image_bytes)

            results.append({
                "type": "image",
                "page": page_number,
                "image_path": image_path
            })

    doc.close()

    return results

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

text_and_images = extract_text_and_images(PDF_PATH)

print("extracted_images_and_text")

