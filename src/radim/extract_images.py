import fitz  # PyMuPDF
from PIL import Image
import io
import os
"""
!pip install PyMuPDF
"""
def extract_images(file_path) -> list[Image.Image]:
    """
    Извлекает встроенные изображения из PDF и возвращает список PIL.Image.
    принимает путь до pdf файла
    возвращает список списков картинок
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Файл {file_path} не найден")

    doc = fitz.open(file_path)
    images = [[] for _ in range(len(doc))]
    seen_xrefs = set()  # Для избежания дубликатов

    for page_num,page in enumerate(doc):
        img_list = page.get_images(full=True)

        for img_info in img_list:
            xref = img_info[0]
            if xref in seen_xrefs:
                continue
            seen_xrefs.add(xref)

            try:
                base_image = doc.extract_image(xref)
                image_data = base_image["image"]
                image = Image.open(io.BytesIO(image_data))
                images[page_num].append(image)
            except Exception as e:
                print(f"Ошибка при извлечении изображения (xref={xref}): {e}")

    doc.close()
    return images