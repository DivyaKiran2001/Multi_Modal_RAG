import asyncio
import uuid
import time
import faiss
import numpy as np
import requests
from vertexai.preview.vision_models import Image

from urllib.parse import urlparse, urljoin
from bs4 import BeautifulSoup
from playwright.async_api import async_playwright
import trafilatura

import vertexai
from vertexai.generative_models import GenerativeModel, Part
from vertexai.language_models import TextEmbeddingModel
from vertexai.preview.vision_models import MultiModalEmbeddingModel


# =========================================================
# CONFIG
# =========================================================

PROJECT_ID = "august-clover-487311-g6"
LOCATION = "us-central1"

CHUNK_SIZE = 900
CHUNK_OVERLAP = 200
TEXT_BATCH_SIZE = 20
REQUEST_DELAY = 0.5
MAX_IMAGES_TOTAL = 10
TOP_K_TEXT = 5
TOP_K_IMAGES = 2

vertexai.init(project=PROJECT_ID, location=LOCATION)

gemini_model = GenerativeModel("gemini-2.5-flash")

text_embedding_model = TextEmbeddingModel.from_pretrained(
    "text-embedding-004"
)

multimodal_model = MultiModalEmbeddingModel.from_pretrained(
    "multimodalembedding@001"
)


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

            browser = await p.chromium.launch(
                headless=True,
                args=["--disable-dev-shm-usage", "--no-sandbox"]
            )

            queue = [(self.base_url, 0)]

            while queue and len(self.visited) < self.max_pages:
                url, depth = queue.pop(0)

                if url in self.visited or depth > self.max_depth:
                    continue

                print("Crawling:", url)
                self.visited.add(url)

                try:
                    page = await browser.new_page()
                    await page.goto(
                        url,
                        timeout=120000,
                        wait_until="domcontentloaded"
                    )

                    html = await page.content()
                    await page.close()

                    extracted = trafilatura.extract(
                        html,
                        include_links=True,
                        include_images=True,
                        output_format="html"
                    )

                    if extracted:
                        self.pages.append({
                            "url": url,
                            "raw_html": html,
                            "clean_html": extracted
                        })

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
# EXTRACTION
# =========================================================

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


def parse_pages(pages):
    all_chunks = []
    all_images = set()

    for page in pages:
        all_chunks.extend(
            extract_text_chunks(page["clean_html"], page["url"])
        )
        all_images.update(
            extract_images(page["raw_html"], page["url"])
        )

    return all_chunks, list(all_images)


# =========================================================
# MULTIMODAL STORE
# =========================================================

class MultiModalStore:

    def __init__(self):
        self.text_index = None
        self.image_index = None
        self.text_data = []
        self.image_data = []

    def embed_text_batch(self, texts):
        vectors = []

        for i in range(0, len(texts), TEXT_BATCH_SIZE):
            batch = texts[i:i+TEXT_BATCH_SIZE]
            time.sleep(REQUEST_DELAY)

            response = text_embedding_model.get_embeddings(batch)

            for emb in response:
                vectors.append(np.array(emb.values).astype("float32"))

        return vectors

    def embed_image(self, image_url):
        try:
            time.sleep(REQUEST_DELAY)

            response = requests.get(image_url)
            response.raise_for_status()

            image = Image(image_bytes=response.content)
            emb = multimodal_model.get_embeddings(image=image)

            return np.array(emb.image_embedding).astype("float32")

        except Exception as e:
            print("Skipped image:", image_url, e)
            return None

    def embed_query_for_image(self, query):
        time.sleep(REQUEST_DELAY)

        emb = multimodal_model.get_embeddings(
            contextual_text=query
        )

        return np.array(emb.text_embedding).astype("float32")

    def route_query(self, query: str) -> str:
        """
        Uses Gemini to classify whether the query
        needs TEXT, IMAGE, or BOTH retrieval.
        """

        prompt = f"""
    You are a query router.

    Classify the user query into one of these three categories:
    TEXT   -> needs textual explanation only
    IMAGE  -> refers specifically to diagrams, figures, or visuals
    BOTH   -> requires both text and image understanding

    Return ONLY one word: TEXT, IMAGE, or BOTH.

    Query: {query}
    """

        response = gemini_model.generate_content(prompt)
        decision = response.text.strip().upper()

        if decision not in ["TEXT", "IMAGE", "BOTH"]:
            return "BOTH"  # safe fallback

        return decision


    async def build(self, chunks, images):

        print("Embedding TEXT chunks...")
        texts = [c["text"] for c in chunks]

        text_vectors = self.embed_text_batch(texts)
        self.text_data = chunks

        text_vectors = np.vstack(text_vectors)
        faiss.normalize_L2(text_vectors)

        self.text_index = faiss.IndexFlatIP(text_vectors.shape[1])
        self.text_index.add(text_vectors)

        print("Embedding IMAGES...")
        image_vectors = []
        valid_images = []

        for img in images[:MAX_IMAGES_TOTAL]:
            vec = self.embed_image(img)
            if vec is not None:
                image_vectors.append(vec)
                valid_images.append(img)

        if len(image_vectors) > 0:
            image_vectors = np.vstack(image_vectors).astype("float32")
            faiss.normalize_L2(image_vectors)

            self.image_index = faiss.IndexFlatIP(image_vectors.shape[1])
            self.image_index.add(image_vectors)

            self.image_data = valid_images


    def search(self, query, route):

        results = []

        # ---------------- TEXT ONLY ----------------
        if route == "TEXT":

            q_text = self.embed_text_batch([query])[0].reshape(1, -1)
            faiss.normalize_L2(q_text)

            if self.text_index:
                _, idx = self.text_index.search(q_text, TOP_K_TEXT)

                for i in idx[0]:
                    results.append({
                        "type": "text",
                        "content": self.text_data[i]["text"]
                    })

            return results

        # ---------------- IMAGE ONLY ----------------
        if route == "IMAGE":

            if self.image_index:
                q_image = self.embed_query_for_image(query).reshape(1, -1)
                faiss.normalize_L2(q_image)

                _, idx = self.image_index.search(q_image, TOP_K_IMAGES)

                for i in idx[0]:
                    results.append({
                        "type": "image",
                        "content": self.image_data[i]
                    })

            return results

        # ---------------- BOTH ----------------

        # TEXT
        q_text = self.embed_text_batch([query])[0].reshape(1, -1)
        faiss.normalize_L2(q_text)

        if self.text_index:
            _, idx = self.text_index.search(q_text, TOP_K_TEXT)

            for i in idx[0]:
                results.append({
                    "type": "text",
                    "content": self.text_data[i]["text"]
                })

        # IMAGE
        if self.image_index:
            q_image = self.embed_query_for_image(query).reshape(1, -1)
            faiss.normalize_L2(q_image)

            _, idx = self.image_index.search(q_image, TOP_K_IMAGES)

            for i in idx[0]:
                results.append({
                    "type": "image",
                    "content": self.image_data[i]
                })

        return results


    # def search(self, query):

    #     results = []

    #     # TEXT SEARCH
    #     q_text = self.embed_text_batch([query])[0].reshape(1, -1)
    #     faiss.normalize_L2(q_text)

    #     if self.text_index:
    #         _, idx = self.text_index.search(q_text, TOP_K_TEXT)
    #         for i in idx[0]:
    #             results.append({
    #                 "type": "text",
    #                 "content": self.text_data[i]["text"]
    #             })

    #     # IMAGE SEARCH (Always retrieve top N)
    #     if self.image_index:
    #         q_image = self.embed_query_for_image(query).reshape(1, -1)
    #         faiss.normalize_L2(q_image)

    #         _, idx = self.image_index.search(q_image, TOP_K_IMAGES)

    #         for i in idx[0]:
    #             results.append({
    #                 "type": "image",
    #                 "content": self.image_data[i]
    #             })

    #     return results


# =========================================================
# TRUE MULTIMODAL QA
# =========================================================

# async def ask_question(store, query):

#     results = store.search(query)
#     print("****RESULTS****",results)

#     content_parts = []

#     content_parts.append(
#         Part.from_text(
#             f"""
# Answer the question using the provided context.

# If the images are relevant to the question, use them.
# If they are not relevant, ignore them completely.

# QUESTION:
# {query}
# """
#         )
#     )

#     text_context = "\n".join(
#         r["content"] for r in results if r["type"] == "text"
#     )

#     if text_context:
#         content_parts.append(
#             Part.from_text(f"\nContext:\n{text_context}")
#         )

#     for r in results:
#         if r["type"] == "image":
#             try:
#                 img_response = requests.get(r["content"])
#                 img_response.raise_for_status()

#                 content_parts.append(
#                     Part.from_data(
#                         img_response.content,
#                         mime_type="image/png"
#                     )
#                 )
#             except Exception:
#                 continue

#     response = gemini_model.generate_content(content_parts)

#     return response.text
# async def ask_question(store, query):

#     # 1️⃣ Route first (LLM-based routing)
#     route = store.route_query(query)
#     print("ROUTE DECISION:", route)

#     # 2️⃣ Retrieve based on route
#     results = store.search(query)
#     print("****RESULTS****", results)

#     content_parts = []

#     # 3️⃣ Build instruction based on route
#     if route == "TEXT":
#         instruction = (
#             "Answer the question using ONLY the provided text context.\n"
#             "Do not rely on any visual reasoning.\n\n"
#             f"QUESTION:\n{query}"
#         )

#     elif route == "IMAGE":
#         instruction = (
#             "Answer the question using ONLY the provided images.\n"
#             "Extract information strictly from the diagrams or visuals.\n"
#             "Do not use external assumptions.\n\n"
#             f"QUESTION:\n{query}"
#         )

#     else:  # BOTH
#         instruction = (
#             "Answer the question using both the provided text and images.\n"
#             "Combine visual and textual information if needed.\n\n"
#             f"QUESTION:\n{query}"
#         )

#     content_parts.append(Part.from_text(instruction))

#     # 4️⃣ Add text context (only if present)
#     text_context = "\n".join(
#         r["content"] for r in results if r["type"] == "text"
#     )

#     if text_context:
#         content_parts.append(
#             Part.from_text(f"\nContext:\n{text_context}")
#         )

#     # 5️⃣ Attach images (only if present)
#     for r in results:
#         if r["type"] == "image":
#             try:
#                 img_response = requests.get(r["content"])
#                 img_response.raise_for_status()

#                 content_parts.append(
#                     Part.from_data(
#                         img_response.content,
#                         mime_type="image/png"
#                     )
#                 )

#             except Exception:
#                 continue

#     # 6️⃣ Generate response
#     response = gemini_model.generate_content(content_parts)

#     return response.text

async def ask_question(store, query):

    # ✅ Route once (production-safe)
    route = store.route_query(query)
    print("ROUTE DECISION:", route)

    # ✅ Pass route explicitly
    results = store.search(query, route)
    print("****RESULTS****", results)

    content_parts = []

    # Build instruction based on route
    if route == "TEXT":
        instruction = (
            "Answer using ONLY the provided text context.\n"
            "Do not rely on visual reasoning.\n\n"
            f"QUESTION:\n{query}"
        )

    elif route == "IMAGE":
        instruction = (
            "Answer using ONLY the provided images.\n"
            "Extract information strictly from diagrams or visuals.\n\n"
            f"QUESTION:\n{query}"
        )

    else:  # BOTH
        instruction = (
            "Answer using both text and images if needed.\n\n"
            f"QUESTION:\n{query}"
        )

    content_parts.append(Part.from_text(instruction))

    # Add text context
    text_context = "\n".join(
        r["content"] for r in results if r["type"] == "text"
    )

    if text_context:
        content_parts.append(
            Part.from_text(f"\nContext:\n{text_context}")
        )

    # Attach images
    for r in results:
        if r["type"] == "image":
            try:
                img_response = requests.get(r["content"])
                img_response.raise_for_status()

                content_parts.append(
                    Part.from_data(
                        img_response.content,
                        mime_type="image/png"
                    )
                )

            except Exception:
                continue

    response = gemini_model.generate_content(content_parts)

    return response.text


# =========================================================
# MAIN
# =========================================================

async def main():

    url = "https://docs.cloud.google.com/agent-builder/agent-engine/overview"

    crawler = WebsiteCrawler(url)
    pages = await crawler.crawl()

    chunks, images = parse_pages(pages)

    store = MultiModalStore()
    await store.build(chunks, images)

    answer = await ask_question(
        store,
        # "What are the tools present in the agent engine runtime diagram under the agent section?"
        # "What are the Agent framweorks present in the agent engine runtime diagram under the agent frameworks section?"
        # "What is the code for building the Vertex AI Agent Engine as mentioned in the diagram?"
        "How to deploy from source files?"
    )

    print("\nFINAL ANSWER:\n", answer)


if __name__ == "__main__":
    asyncio.run(main())
