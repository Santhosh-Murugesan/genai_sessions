from pathlib import Path
import json

from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.datamodel.document import PictureItem, TableItem


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

PDF_FILE = "bus_ticket.pdf"

OUTPUT_DIR = Path("extracted_data")
IMAGE_DIR = OUTPUT_DIR / "images"

OUTPUT_DIR.mkdir(exist_ok=True)
IMAGE_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------
# Configure Docling
# ---------------------------------------------------------

pipeline_options = PdfPipelineOptions()

# Digital PDFs don't normally need OCR.
pipeline_options.do_ocr = False

# Extract table structure
pipeline_options.do_table_structure = True

# Extract images / figures / charts
pipeline_options.generate_picture_images = True

# Image quality
pipeline_options.images_scale = 2.0


converter = DocumentConverter(
    format_options={
        InputFormat.PDF: PdfFormatOption(
            pipeline_options=pipeline_options
        )
    }
)


# ---------------------------------------------------------
# Convert PDF
# ---------------------------------------------------------

print(f"Processing: {PDF_FILE}")

result = converter.convert(PDF_FILE)

doc = result.document

print("PDF processed successfully.")


# ---------------------------------------------------------
# Output records
# ---------------------------------------------------------

records = []

picture_counter = 0
table_counter = 0


# ---------------------------------------------------------
# Iterate through document elements
# ---------------------------------------------------------

for element, level in doc.iterate_items():

    # =====================================================
    # TEXT
    # =====================================================

    if hasattr(element, "text") and element.text:

        text = element.text.strip()

        if text:

            record = {
                "type": "text",
                "content": text,
                "metadata": {
                    "source": PDF_FILE,
                    "element_type": element.__class__.__name__,
                    "level": level
                }
            }

            records.append(record)


    # =====================================================
    # TABLE
    # =====================================================

    elif isinstance(element, TableItem):

        table_counter += 1

        dataframe = element.export_to_dataframe()

        table_markdown = dataframe.to_markdown(index=False)

        record = {
            "type": "table",
            "content": table_markdown,
            "metadata": {
                "source": PDF_FILE,
                "table_id": table_counter,
                "rows": len(dataframe),
                "columns": len(dataframe.columns)
            }
        }

        records.append(record)

        # Save table as CSV

        csv_file = (
            OUTPUT_DIR /
            f"table_{table_counter}.csv"
        )

        dataframe.to_csv(csv_file, index=False)


    # =====================================================
    # PICTURES / FIGURES / CHARTS
    # =====================================================

    elif isinstance(element, PictureItem):

        picture_counter += 1

        image_file = (
            IMAGE_DIR /
            f"picture_{picture_counter}.png"
        )

        image = element.get_image(doc)

        image.save(image_file)

        record = {
            "type": "picture",
            "content": (
                f"Picture/figure extracted from "
                f"{PDF_FILE}"
            ),
            "metadata": {
                "source": PDF_FILE,
                "picture_id": picture_counter,
                "image_path": str(image_file)
            }
        }

        records.append(record)


# ---------------------------------------------------------
# Save JSONL
# ---------------------------------------------------------

jsonl_file = OUTPUT_DIR / "documents.jsonl"

with open(jsonl_file, "w", encoding="utf-8") as f:

    for record in records:

        f.write(
            json.dumps(
                record,
                ensure_ascii=False
            )
            + "\n"
        )


print()
print("====================================")
print("Extraction completed")
print("====================================")
print(f"Text/Table/Picture records : {len(records)}")
print(f"Tables                     : {table_counter}")
print(f"Pictures                   : {picture_counter}")
print(f"JSONL                      : {jsonl_file}")
print(f"Images                     : {IMAGE_DIR}")