import io
import os
import fitz  # PyMuPDF
from PIL import Image
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# 1. Load CLIP Model
model = SentenceTransformer("sentence-transformers/clip-ViT-B-32")


def extract_and_save_images(pdf_path, output_dir="extracted_images"):
    """
    Extracts embedded images from a PDF, saves them to a local folder,
    and returns a list of dictionaries containing image objects and file paths.
    """
    os.makedirs(output_dir, exist_ok=True)
    doc = fitz.open(pdf_path)
    extracted_data = []

    for page_num in range(len(doc)):
        page = doc[page_num]
        image_list = page.get_images(full=True)

        for img_idx, img_info in enumerate(image_list):
            xref = img_info[0]
            base_image = doc.extract_image(xref) #base_image = {"image": image_bytes, "ext": ext, "width": width, "height": height, "bpc": bpc}
            image_bytes = base_image["image"]
            ext = base_image["ext"]  # e.g., 'png', 'jpeg'

            try:
                # Open image from bytes
                image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

                # Build a unique local file path
                filename = f"page_{page_num + 1}_img_{img_idx + 1}.{ext}" #page_1_img_1.png
                filepath = os.path.join(output_dir, filename)

                # Save image to disk
                image.save(filepath)

                extracted_data.append(
                    {
                        "image": image,
                        "filepath": filepath,
                        "metadata": f"Page {page_num + 1}, Image {img_idx + 1}",
                    }
                )
            except Exception as e:
                print(
                    f"Skipping unreadable image on page {page_num + 1}: {e}"
                )

    return extracted_data


# 2. Extract and Save Images locally
pdf_path = r"C:\Users\Santhosh\OneDrive\Documents\GenAI_Workspace\genai_sessions\multimodel_emdedding_model\bus_ticket.pdf"
output_folder = "extracted_images"
extracted_images = extract_and_save_images(pdf_path, output_dir=output_folder)

if not extracted_images:
    print("No images found in the PDF.")
else:
    print(
        f"Extracted and saved {len(extracted_images)} images into '{output_folder}/'."
    )

    # 3. Batch Generate Embeddings
    images = [item["image"] for item in extracted_images]
    image_embeddings = model.encode(
        images, normalize_embeddings=True, convert_to_numpy=True
    ).astype("float32") #[["42345234", "234234234", "234234234"],["234234234", "234234234", "234234234"]...]

    # 4. Initialize and Populate FAISS Index
    dimension = image_embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)  # Cosine Similarity
    index.add(image_embeddings)

    # 5. Execute User Query
    user_query = "location of the dropping point"
    query_embedding = model.encode(
        [user_query], normalize_embeddings=True, convert_to_numpy=True
    ).astype("float32")

    # 6. Retrieve Top Matches
    k = min(5, index.ntotal)
    distances, indices = index.search(query_embedding, k)

    # 7. Print Results with Local File Paths for Verification
    print(f"\nTop matches for query: '{user_query}'\n" + "-" * 60)
    for rank, (score, idx) in enumerate(zip(distances[0], indices[0]), 1):
        item = extracted_images[idx]
        print(
            f"Rank {rank}: Score = {score:.4f} | Path: {item['filepath']} ({item['metadata']})"
        )