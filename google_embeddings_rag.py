import asyncio
import uuid
import os
import base64
import pickle
import faiss
import numpy as np
import aiohttp

from urllib.parse import urlparse, urljoin
from bs4 import BeautifulSoup
from playwright.async_api import async_playwright
import trafilatura
import google.generativeai as genai


# =========================================================
# CONFIG
# =========================================================

CHUNK_SIZE = 1200
CHUNK_OVERLAP = 200
PERSIST_DIR = "faiss_store"

genai.configure(api_key="AIzaSyBHnyoY3TBV-h2HrvcygkLDoO5HR1m6IYc")
gemini_model = genai.GenerativeModel("gemini-1.5-flash")


# =========================================================
# ASYNC WEBSITE CRAWLER
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
# GEMINI EMBEDDING STORE (MANUAL, NO LANGCHAIN)
# =========================================================

class FastMultiModalStore:

    def __init__(self, persist_dir=PERSIST_DIR):

        self.persist_dir = persist_dir
        os.makedirs(self.persist_dir, exist_ok=True)

        self.text_index_path = os.path.join(self.persist_dir, "text.index")
        self.image_index_path = os.path.join(self.persist_dir, "image.index")
        self.text_data_path = os.path.join(self.persist_dir, "text_data.pkl")
        self.image_urls_path = os.path.join(self.persist_dir, "image_urls.pkl")

        self.text_index = None
        self.image_index = None
        self.text_data = []
        self.image_urls = []

        self.load_if_exists()

    # -----------------------------------------------------

    def load_if_exists(self):

        if os.path.exists(self.text_index_path):
            print("Loading existing FAISS indexes...")

            self.text_index = faiss.read_index(self.text_index_path)

            if os.path.exists(self.image_index_path):
                self.image_index = faiss.read_index(self.image_index_path)

            with open(self.text_data_path, "rb") as f:
                self.text_data = pickle.load(f)

            if os.path.exists(self.image_urls_path):
                with open(self.image_urls_path, "rb") as f:
                    self.image_urls = pickle.load(f)

            print("Indexes loaded successfully.")

    # -----------------------------------------------------

    def embed_text(self, text):
        response = genai.embed_content(
            model="embedding-001",
            content=text
        )
        return np.array(response["embedding"]).astype("float32")

    def embed_image(self, image_url):
        response = genai.embed_content(
            model="multimodalembedding",
            content=image_url
        )
        return np.array(response["embedding"]).astype("float32")

    # -----------------------------------------------------

    async def build(self, chunks):

        if self.text_index is not None:
            print("Using existing stored index.")
            return

        print("Embedding text...")

        self.text_data = chunks
        text_embeddings = np.vstack(
            [self.embed_text(c["text"]) for c in chunks]
        )

        dim = text_embeddings.shape[1]
        self.text_index = faiss.IndexFlatIP(dim)
        self.text_index.add(text_embeddings)

        print("Embedding images...")

        all_image_urls = list({
            img for chunk in chunks for img in chunk["images"]
        })

        self.image_urls = all_image_urls

        if all_image_urls:
            img_embeddings = np.vstack(
                [self.embed_image(url) for url in all_image_urls]
            )

            self.image_index = faiss.IndexFlatIP(dim)
            self.image_index.add(img_embeddings)

        print("Saving FAISS indexes...")

        faiss.write_index(self.text_index, self.text_index_path)

        if self.image_index:
            faiss.write_index(self.image_index, self.image_index_path)

        with open(self.text_data_path, "wb") as f:
            pickle.dump(self.text_data, f)

        with open(self.image_urls_path, "wb") as f:
            pickle.dump(self.image_urls, f)

        print("Indexes saved successfully.")

    # -----------------------------------------------------

    def search_text(self, query, k=5):
        query_emb = self.embed_text(query).reshape(1, -1)
        scores, indices = self.text_index.search(query_emb, k)
        return [self.text_data[i] for i in indices[0]]

    def search_images(self, query, k=3):
        if self.image_index is None:
            return []

        query_emb = self.embed_image(query).reshape(1, -1)
        scores, indices = self.image_index.search(query_emb, k)
        return [self.image_urls[i] for i in indices[0]]


# =========================================================
# GEMINI ANSWER
# =========================================================

async def ask_question(store, query):

    text_chunks = store.search_text(query)
    image_urls = store.search_images(query)

    text_context = "\n\n".join([c["text"] for c in text_chunks])

    prompt = f"""
You are a technical documentation assistant.

TEXT CONTEXT:
{text_context}

QUESTION:
{query}

If images are relevant, explain what they show.
"""

    response = gemini_model.generate_content(prompt)
    return response.text


# =========================================================
# MAIN
# =========================================================

async def main():

    url = "https://docs.cloud.google.com/agent-builder/agent-engine/overview"

    crawler = WebsiteCrawler(url, max_depth=2, max_pages=3)
    pages = await crawler.crawl()

    print("Pages crawled:", len(pages))

    chunks = parse_pages(pages)
    print("Chunks created:", len(chunks))

    store = FastMultiModalStore()
    await store.build(chunks)

    query = "Give me list of supported regions for Vertex AI Agent Engine"
    answer = await ask_question(store, query)

    print("\n===== FINAL ANSWER =====\n")
    print(answer)


if __name__ == "__main__":
    asyncio.run(main())
