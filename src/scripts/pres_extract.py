import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import io
import os
import json
from uuid import uuid4
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def extract_text(
        collection,  # ChromaDB коллекция
        pdf_path,  # Путь к PDF файлу
        chunk_size=1000,  # Размер чанка
        chunk_overlap=100,  # Перекрытие чанков
        metadata_json_path=None  # Путь к JSON с метаданными
) -> None:
    """
    Обрабатывает PDF и добавляет данные в ChromaDB
    """
    pdf_filename = os.path.basename(pdf_path)
    raw_meta = {}

    # Загрузка метаданных из JSON (если указан)
    if metadata_json_path:
        try:
            with open(metadata_json_path, 'r') as f:
                raw_meta = json.load(f)
        except FileNotFoundError:
            print(f"Metadata file {metadata_json_path} not found. Using empty metadata.")

    # Ключ для поиска метаданных (имя файла без расширения)
    pdf_key = os.path.splitext(pdf_filename)[0]
    pdf_metadata = raw_meta.get(pdf_key, {})

    doc = fitz.open(pdf_path)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    for page_num, page in enumerate(doc):
        # Извлечение текста
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

        full_text = page.get_text().strip()
        body = full_text.replace(title, "", 1).strip() if title else full_text
        slide_text = f"{title}\n{body}"

        # Извлечение текста с изображений
        img_list = page.get_images(full=True)
        for img_info in img_list:
            xref = img_info[0]
            try:
                base_image = doc.extract_image(xref)
                image_data = base_image["image"]
                img = Image.open(io.BytesIO(image_data))
                text = pytesseract.image_to_string(img, lang="rus+eng").strip()
                if text:
                    slide_text += f"\n{text}"
            except Exception as e:
                print(f"Ошибка в слайде {page_num + 1}: {e}")

        # Формирование метаданных
        metadata = {
            "source": pdf_filename,
            "slide": page_num + 1,
            "title": title
        }
        metadata.update(pdf_metadata)  # Добавляем общие метаданные

        # Создание документа и разбиение на чанки
        doc_page = Document(page_content=slide_text, metadata=metadata)
        chunks = text_splitter.split_documents([doc_page])

        # Добавление чанков в ChromaDB
        ids = [str(uuid4()) for _ in range(len(chunks))]
        collection.add(
            documents=[chunk.page_content for chunk in chunks],
            ids=ids,
            metadatas=[chunk.metadata for chunk in chunks]
        )

    doc.close()