import fitz  # PyMuPDF
import easyocr
from PIL import Image
import io
import json


def extract_text(pres_path, name) -> None:
    """
    извлекает текст из pdf файла
    принимает путь до файла и создает файл с именем name формата json
    структура:
          "amount_slides": int - число слайдов,
          "titles": list - заголовки(первые строки) страниц,
          "bodies": list - тело страниц,
          "text_images": list[list] - распознаный текст с картинок
    """
    output = {
        "amount_slides": 0,
        "titles": [],
        "bodies": [],
        "text_images": []
    }

    # Инициализация EasyOCR (русский и английский)
    reader = easyocr.Reader(['ru', 'en'])

    doc = fitz.open(pres_path)
    output["amount_slides"] = len(doc)

    for page_num, page in enumerate(doc):
        # Извлечение заголовка (первый крупный текст)
        title = ""
        blocks = page.get_text("dict")["blocks"]
        for block in blocks:
            if block["type"] == 0:  # Текстовый блок
                lines = block["lines"]
                for line in lines:
                    spans = line["spans"]
                    if spans:
                        span = max(spans, key=lambda s: s["size"])  # Самый крупный шрифт
                        title = span["text"].strip()
                        if title: break
                if title: break
        output["titles"].append(title)

        # Извлечение основного текста
        full_text = page.get_text().strip()
        body = full_text.replace(title, "", 1).strip() if title else full_text
        output["bodies"].append(body)

        # Извлечение текста из встроенных изображений
        slide_images_text = []
        img_list = page.get_images(full=True)

        for img_info in img_list:
            xref = img_info[0]
            try:
                base_image = doc.extract_image(xref)
                image_data = base_image["image"]
                img = Image.open(io.BytesIO(image_data))

                # Распознавание текста
                results = reader.readtext(image_data, detail=0)
                text = " ".join(results).strip()
                if text:
                    slide_images_text.append(text)
            except Exception as e:
                print(f"Ошибка в слайде {page_num + 1}: {e}")

        output["text_images"].append(slide_images_text)

    doc.close()

    # Сохранение в JSON
    with open(name + ".json", "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=4)