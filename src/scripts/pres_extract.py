import io
import json

import fitz
import pytesseract
from PIL import Image

"""
Для colab
!sudo apt update
!sudo apt install tesseract-ocr -y
!sudo apt install libtesseract-dev -y
!sudo apt install tesseract-ocr-eng -y  # Для английского языка
!sudo apt install tesseract-ocr-rus -y  # Для русского языка
%pip install PyMuPDF pytesseract
"""


def extract_text(pres_path) -> None:
    """
    Extracts text from a PDF file.
    Takes the path to the file and creates a file named 'file' in JSON format.
    Structure:
        "amount_slides": int - number of slides,
        "titles": list - titles (first lines) of the pages,
        "bodies": list - body text of the pages,
        "text_images": list[list] - recognized text from images.
      """
    output = {
        "amount_slides": 0,
        "titles":        [],
        "bodies":        [],
        "text_images":   []
    }

    doc = fitz.open(pres_path)
    output["amount_slides"] = len(doc)

    for page_num, page in enumerate(doc):
        # Extract titles
        title = ""
        blocks = page.get_text("dict")["blocks"]
        for block in blocks:
            if "lines" in block and block["type"] == 0:
                for line in block["lines"]:
                    for span in line["spans"]:
                        if not title and span["text"].strip():
                            title = span["text"].strip()
                            break
                if title:
                    break
        output["titles"].append(title)

        # Extract main text
        full_text = page.get_text().strip()
        body = full_text.replace(title, "", 1).strip() if title else full_text
        output["bodies"].append(body)

        # Extract images from slides
        slide_images_text = []
        img_list = page.get_images(full=True)

        for img_info in img_list:
            xref = img_info[0]
            try:
                base_image = doc.extract_image(xref)
                image_data = base_image["image"]
                img = Image.open(io.BytesIO(image_data))
                text = pytesseract.image_to_string(img, lang="rus+eng").strip()
                if text:
                    slide_images_text.append(text)
            except Exception as e:
                print(f"Slide # {page_num + 1} error: {e}")

        output["text_images"].append(slide_images_text)

    doc.close()

    # Save in JSON
    with open(json_filename, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=4)
