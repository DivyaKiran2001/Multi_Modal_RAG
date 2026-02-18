

import asyncio
import uuid
import os
import pickle
import time
import faiss
import numpy as np

from urllib.parse import urlparse, urljoin
from bs4 import BeautifulSoup
from playwright.async_api import async_playwright
import trafilatura
import google.generativeai as genai


# =========================================================
# CONFIG
# =========================================================

CHUNK_SIZE = 900
CHUNK_OVERLAP = 200
PERSIST_DIR = "faiss_store"

REQUEST_DELAY = 0.2  # small delay for safety

os.makedirs(PERSIST_DIR, exist_ok=True)

# 🔑 Gemini API key


genai.configure(api_key="")


# Gemini LLM for answering
gemini_model = genai.GenerativeModel("gemini-2.5-flash")


# =========================================================
# WEBSITE CRAWLER
# =========================================================

class WebsiteCrawler:

    def __init__(self, base_url, max_depth=1, max_pages=2):
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
                    await page.goto(url, timeout=60000, wait_until="domcontentloaded")

                  

                    html = await page.content()
                    await page.close()

                    extracted = trafilatura.extract(
                        html,
                        include_links=True,
                        include_images=False,
                        include_tables=True,
                        output_format="html"
                    )

                    if extracted:
                        self.pages.append({"url": url, "html": extracted})

                        # 🔁 Keep internal links (your original logic preserved)
                        soup = BeautifulSoup(extracted, "html.parser")
                        for link in soup.find_all("a", href=True):
                            full_url = urljoin(url, link["href"])
                            parsed = urlparse(full_url)

                            if parsed.netloc == self.base_domain:
                                clean = parsed._replace(
                                    fragment="", query=""
                                ).geturl()

                                if clean not in self.visited:
                                    queue.append((clean, depth + 1))

                except Exception as e:
                    print("Failed:", url, e)

            await browser.close()

        return self.pages


# =========================================================
# CHUNKING
# =========================================================

def chunk_page(html, base_url):

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


def parse_pages(pages):
    all_chunks = []
    for p in pages:
        all_chunks.extend(chunk_page(p["html"], p["url"]))
    return all_chunks


# =========================================================
# TEXT-ONLY FAISS STORE (Gemini Embeddings)
# =========================================================

class TextStore:

    def __init__(self):

        self.index_path = f"{PERSIST_DIR}/text.index"
        self.data_path = f"{PERSIST_DIR}/text.pkl"

        self.index = None
        self.data = []

        self.load()

    def load(self):
        if os.path.exists(self.index_path):
            print("Loading existing index...")
            self.index = faiss.read_index(self.index_path)
            self.data = pickle.load(open(self.data_path, "rb"))

    # ---------- GEMINI TEXT EMBEDDING ----------

    def embed_text(self, text):

        time.sleep(REQUEST_DELAY)

        response = genai.embed_content(
            model="gemini-embedding-001",  # FREE Gemini embedding model
            content=text[:1000]     # safety trim
        )

        return np.array(response["embedding"]).astype("float32")

    # ---------- BUILD ----------

    async def build(self, chunks):

        if self.index is not None:
            print("Using existing index.")
            return

        print("Embedding text...")

        vectors = []
        metadata = []

        for chunk in chunks:
            vec = self.embed_text(chunk["text"])
            vectors.append(vec)
            metadata.append(chunk)
        if not vectors:
            print("No content found. Skipping index build.")
            return


        embeddings = np.vstack(vectors)

        faiss.normalize_L2(embeddings)

        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(embeddings)

        self.data = metadata

        faiss.write_index(self.index, self.index_path)
        pickle.dump(self.data, open(self.data_path, "wb"))

        print("Index built successfully.")

    # ---------- SEARCH ----------

    def search(self, query, k=5):

        q = self.embed_text(query).reshape(1, -1)
        faiss.normalize_L2(q)

        _, idx = self.index.search(q, k)
        return [self.data[i] for i in idx[0]]


# =========================================================
# GEMINI QA
# =========================================================

async def ask_question(store, query):

    results = store.search(query)
    print(results)

    context = "\n\n".join(r["text"] for r in results)

    prompt = f"""
You are a technical assistant.

CONTEXT:
{context}

QUESTION:
{query}
"""

    response = gemini_model.generate_content(prompt)
    return response.text


# =========================================================
# MAIN
# =========================================================

async def main():

    url = "https://docs.cloud.google.com/agent-builder/agent-engine/overview"

    crawler = WebsiteCrawler(url)
    pages = await crawler.crawl()

    chunks = parse_pages(pages)

    store = TextStore()
    await store.build(chunks)

    answer = await ask_question(
        store,
        "Give me methods of how we can deploy an agent"
    )

    print("\nFINAL ANSWER:\n", answer)


if __name__ == "__main__":
    asyncio.run(main())
