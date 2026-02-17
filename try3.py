


import asyncio
import uuid
import requests
import chromadb
import base64
import os

from PIL import Image
from io import BytesIO
from urllib.parse import urlparse, urljoin

from playwright.async_api import async_playwright
from bs4 import BeautifulSoup
import trafilatura

from sentence_transformers import SentenceTransformer
from chromadb.config import Settings

import google.generativeai as genai


# =========================================================
# CONFIG
# =========================================================

CHUNK_SIZE = 1200
CHUNK_OVERLAP = 200

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
gemini_model = genai.GenerativeModel("gemini-1.5-flash")


# =========================================================
# CRAWLER
# =========================================================

class WebsiteCrawler:
    def __init__(self, base_url, max_depth=2, max_pages=5):
        self.base_url = base_url
        self.base_domain = urlparse(base_url).netloc
        self.max_depth = max_depth
        self.max_pages = max_pages
        self.visited = set()
        self.pages = []

    async def crawl(self):
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            queue = [(self.base_url, 0)]

            while queue and len(self.visited) < self.max_pages:
                url, depth = queue.pop(0)

                if url in self.visited or depth > self.max_depth:
                    continue

                print("Crawling:", url)
                self.visited.add(url)

                try:
                    page = await browser.new_page()
                    await page.goto(url, timeout=45000)
                    await page.wait_for_load_state("domcontentloaded")

                    html = await page.content()
                    await page.close()

                    extracted = trafilatura.extract(
                        html,
                        include_links=True,
                        include_images=True,
                        include_tables=True,
                        output_format="html"
                    )

                    if not extracted:
                        continue

                    self.pages.append({"url": url, "html": extracted})

                    soup = BeautifulSoup(extracted, "html.parser")

                    for link in soup.find_all("a", href=True):
                        full_url = urljoin(url, link["href"])
                        parsed = urlparse(full_url)

                        if parsed.netloc != self.base_domain:
                            continue

                        clean_url = parsed._replace(fragment="", query="").geturl()

                        if clean_url not in self.visited:
                            queue.append((clean_url, depth + 1))

                except Exception as e:
                    print("Failed:", url, e)

            await browser.close()

        return self.pages


# =========================================================
# CHUNKING
# =========================================================

def chunk_page(html, base_url):
    soup = BeautifulSoup(html, "html.parser")

    elements = []

    for tag in soup.descendants:
        if tag.name == "img" and tag.get("src"):
            elements.append({
                "type": "image",
                "content": urljoin(base_url, tag["src"])
            })
        elif tag.string and tag.string.strip():
            elements.append({
                "type": "text",
                "content": tag.string.strip()
            })

    full_text = ""
    image_positions = []

    for el in elements:
        if el["type"] == "text":
            full_text += el["content"] + " "
        else:
            image_positions.append((len(full_text), el["content"]))

    chunks = []
    start = 0

    while start < len(full_text):
        end = start + CHUNK_SIZE
        chunk_text = full_text[start:end]

        chunk_images = [
            img_url for pos, img_url in image_positions
            if start <= pos <= end
        ]

        chunks.append({
            "id": str(uuid.uuid4()),
            "url": base_url,
            "text": chunk_text,
            "images": list(set(chunk_images))
        })

        start = end - CHUNK_OVERLAP

    return chunks


def parse_pages(pages):
    all_chunks = []
    for page in pages:
        all_chunks.extend(chunk_page(page["html"], page["url"]))
    return all_chunks


# =========================================================
# VECTOR STORE
# =========================================================

class MultiModalVectorStore:

    def __init__(self):

        self.client = chromadb.Client(
            Settings(persist_directory="./chroma_db")
        )

        self.text_collection = self.client.get_or_create_collection("text_collection")
        self.image_collection = self.client.get_or_create_collection("image_collection")

        # Use CLIP for both text & image retrieval
        self.clip_model = SentenceTransformer("clip-ViT-B-32")

    def add_documents(self, chunks):

        print("Embedding text...")
        texts = [chunk["text"] for chunk in chunks]
        ids = [chunk["id"] for chunk in chunks]
        embeddings = self.clip_model.encode(texts, batch_size=8)

        self.text_collection.add(
            documents=texts,
            embeddings=embeddings.tolist(),
            ids=ids,
            metadatas=[{"url": chunk["url"]} for chunk in chunks]
        )

        print("Embedding images...")
        image_urls = []
        image_ids = []
        image_metadatas = []

        for chunk in chunks:
            for img_url in chunk["images"]:
                image_urls.append(img_url)
                image_ids.append(str(uuid.uuid4()))
                image_metadatas.append({
                    "chunk_id": chunk["id"],
                    "source_url": chunk["url"],
                    "image_url": img_url
                })

        images = []
        valid_urls = []
        valid_ids = []
        valid_meta = []

        for i, url in enumerate(image_urls):
            try:
                response = requests.get(url, timeout=10)
                img = Image.open(BytesIO(response.content)).convert("RGB")
                images.append(img)
                valid_urls.append(url)
                valid_ids.append(image_ids[i])
                valid_meta.append(image_metadatas[i])
            except:
                continue

        if images:
            img_embeddings = self.clip_model.encode(images, batch_size=4)

            self.image_collection.add(
                documents=valid_urls,
                embeddings=img_embeddings.tolist(),
                ids=valid_ids,
                metadatas=valid_meta
            )

    def search_text(self, query, k=4):
        emb = self.clip_model.encode(query)
        return self.text_collection.query(
            query_embeddings=[emb.tolist()],
            n_results=k
        )

    def search_images(self, query, k=2):
        emb = self.clip_model.encode(query)
        return self.image_collection.query(
            query_embeddings=[emb.tolist()],
            n_results=k
        )


# =========================================================
# GEMINI ANSWER FUNCTION
# =========================================================

def ask_question(vector_store, query):

    # Retrieve text
    text_results = vector_store.search_text(query)
    text_context = "\n\n".join(text_results["documents"][0])

    # Retrieve images
    image_results = vector_store.search_images(query)

    image_parts = []

    if image_results["documents"]:
        for img_url in image_results["documents"][0]:
            try:
                response = requests.get(img_url, timeout=10)
                img_base64 = base64.b64encode(response.content).decode("utf-8")

                image_parts.append({
                    "mime_type": "image/png",
                    "data": img_base64
                })
            except:
                continue

    prompt = f"""
You are a technical documentation assistant.

Use the provided documentation context and images to answer the question.

TEXT CONTEXT:
{text_context}

QUESTION:
{query}

If images are relevant, explain what they show and how they relate to the answer.
"""

    contents = [{
        "role": "user",
        "parts": [{"text": prompt}] + image_parts
    }]

    response = gemini_model.generate_content(contents)
    return response.text


# =========================================================
# MAIN
# =========================================================

async def main():

    url = "https://docs.cloud.google.com/agent-builder/agent-engine/overview"

    crawler = WebsiteCrawler(url)
    pages = await crawler.crawl()

    print("Total pages crawled:", len(pages))

    chunks = parse_pages(pages)
    print("Total chunks created:", len(chunks))

    vector_store = MultiModalVectorStore()
    vector_store.add_documents(chunks)
    print("Vector storing completed")

    query = "Give me list of supported regions for Vertex AI Agent Engine"

    answer = ask_question(vector_store, query)

    print("\n===== FINAL ANSWER =====\n")
    print(answer)


if __name__ == "__main__":
    asyncio.run(main())
