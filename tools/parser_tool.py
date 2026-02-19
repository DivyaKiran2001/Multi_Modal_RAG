# tools/parser_tool.py

import uuid
from urllib.parse import urljoin
from bs4 import BeautifulSoup
from .config import CHUNK_SIZE, CHUNK_OVERLAP

def valid_image(url):
    return not any(k in url.lower() for k in ["logo", "icon", "avatar"])

def extract_text_chunks(html, base_url):
    soup = BeautifulSoup(html, "html.parser")
    text = soup.get_text(" ", strip=True)

    chunks = []
    start = 0

    while start < len(text):
        end = start + CHUNK_SIZE
        chunks.append({
            "id": str(uuid.uuid4()),
            "url": base_url,
            "text": text[start:end]
        })
        start = end - CHUNK_OVERLAP

    return chunks

def extract_images(html, base_url):
    soup = BeautifulSoup(html, "html.parser")
    images = set()

    for img in soup.find_all("img", src=True):
        img_url = urljoin(base_url, img["src"])
        if valid_image(img_url):
            images.add(img_url)

    return list(images)

def parse_pages(pages: list) -> dict:
    all_chunks = []
    all_images = set()

    for page in pages:
        all_chunks.extend(
            extract_text_chunks(page["clean_html"], page["url"])
        )
        all_images.update(
            extract_images(page["raw_html"], page["url"])
        )

    return {
        "chunks": all_chunks,
        "images": list(all_images)
    }
