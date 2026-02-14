# import os
# import asyncio
# import json
# import base64
# from urllib.parse import urljoin, urlparse
# from dotenv import load_dotenv
# from bs4 import BeautifulSoup

# from playwright.async_api import async_playwright
# from unstructured.partition.html import partition_html
# from unstructured.chunking.title import chunk_by_title

# from langchain_core.documents import Document
# from langchain_community.vectorstores import Chroma
# from langchain_google_genai import (
#     GoogleGenerativeAIEmbeddings,
#     ChatGoogleGenerativeAI
# )


# load_dotenv()
# GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# class WebsiteCrawler:
#     def __init__(self, base_url, max_depth=2, max_pages=30):
#         self.base_url = base_url
#         self.base_domain = urlparse(base_url).netloc
#         self.max_depth = max_depth
#         self.max_pages = max_pages
#         self.visited = set()
#         self.pages = []

#     async def crawl(self):
#         async with async_playwright() as p:
#             browser = await p.chromium.launch(headless=True)
#             page = await browser.new_page()
#             await self._crawl_page(page, self.base_url, depth=0)
#             await browser.close()

#         return self.pages

#     async def _crawl_page(self, page, url, depth):
#         if (
#             url in self.visited or
#             depth > self.max_depth or
#             len(self.visited) >= self.max_pages
#         ):
#             return

#         print(f"Crawling: {url} | Depth: {depth}")

#         self.visited.add(url)

#         try:
#             await page.goto(url, timeout=60000)
#             content = await page.content()
#             self.pages.append({"url": url, "html": content})

#             soup = BeautifulSoup(content, "html.parser")

#             for link in soup.find_all("a", href=True):
#                 full_url = urljoin(url, link["href"])
#                 parsed = urlparse(full_url)

#                 # Restrict to same domain
#                 if parsed.netloc == self.base_domain:
#                     # Remove fragments
#                     clean_url = parsed._replace(fragment="").geturl()
#                     await self._crawl_page(page, clean_url, depth + 1)

#         except Exception as e:
#             print(f"Failed to crawl {url}: {e}")


# def parse_and_chunk(page_data):
#     all_chunks = []

#     for page in page_data:
#         elements = partition_html(text=page["html"])
#         chunks = chunk_by_title(
#             elements,
#             max_characters=2000,
#             new_after_n_chars=1500,
#             combine_text_under_n_chars=300
#         )

#         for chunk in chunks:
#             all_chunks.append({
#                 "url": page["url"],
#                 "text": chunk.text,
#                 "orig_elements": chunk.metadata.orig_elements
#             })

#     return all_chunks
# def extract_content_types(chunks):
#     processed_docs = []

#     for chunk in chunks:
#         images = []
#         tables = []
#         hyperlinks = []

#         for el in chunk["orig_elements"]:
#             el_type = type(el).__name__

#             if el_type == "Table":
#                 tables.append(el.text)

#             elif el_type == "Image":
#                 if hasattr(el.metadata, "image_base64"):
#                     images.append(el.metadata.image_base64)

#             elif el_type == "Link":
#                 hyperlinks.append(el.text)

#         processed_docs.append({
#             "url": chunk["url"],
#             "text": chunk["text"],
#             "tables": tables,
#             "images": images,
#             "hyperlinks": hyperlinks
#         })

#     return processed_docs

# llm = ChatGoogleGenerativeAI(
#     model="gemini-2.5-flash",
#     temperature=0
# )

# def create_enriched_text(doc):
#     content = f"Page URL: {doc['url']}\n\n"
#     content += f"TEXT:\n{doc['text']}\n\n"

#     if doc["tables"]:
#         content += "TABLES:\n"
#         for table in doc["tables"]:
#             content += table + "\n\n"

#     if doc["hyperlinks"]:
#         content += "HYPERLINKS FOUND:\n"
#         for link in doc["hyperlinks"]:
#             content += link + "\n"

#     # Describe images
#     for img_base64 in doc["images"]:
#         response = llm.invoke([
#             {
#                 "role": "user",
#                 "content": [
#                     {"type": "text", "text": "Describe this image in detail."},
#                     {
#                         "type": "image_url",
#                         "image_url": {
#                             "url": f"data:image/png;base64,{img_base64}"
#                         }
#                     }
#                 ]
#             }
#         ])
#         content += "\nIMAGE DESCRIPTION:\n"
#         content += response.content + "\n"

#     return content

# def create_vector_store(enriched_docs):
#     embedding_model = GoogleGenerativeAIEmbeddings(
#         model="models/embedding-001"
#     )

#     documents = []

#     for doc in enriched_docs:
#         enriched_text = create_enriched_text(doc)

#         documents.append(
#             Document(
#                 page_content=enriched_text,
#                 metadata={"source_url": doc["url"]}
#             )
#         )

#     vectorstore = Chroma.from_documents(
#         documents=documents,
#         embedding=embedding_model,
#         persist_directory="./chroma_db"
#     )

#     vectorstore.persist()
#     return vectorstore



# def ask_question(vectorstore, query):
#     retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
#     docs = retriever.invoke(query)

#     context = "\n\n".join([doc.page_content for doc in docs])

#     prompt = f"""
#     Answer the question using the context below.
#     If not found, say you don't know.

#     CONTEXT:
#     {context}

#     QUESTION:
#     {query}
#     """

#     response = llm.invoke(prompt)
#     return response.content

# async def main():
#     url = "https://docs.cloud.google.com/agent-builder/agent-engine/overview"

#     crawler = WebsiteCrawler(url, max_depth=2, max_pages=20)
#     pages = await crawler.crawl()

#     chunks = parse_and_chunk(pages)
#     processed = extract_content_types(chunks)

#     db = create_vector_store(processed)

#     answer = ask_question(db, "What is Agent Engine?")
#     print("\nANSWER:\n", answer)

# asyncio.run(main())

import os
import asyncio
import uuid
import base64
import requests
from urllib.parse import urljoin, urlparse
from dotenv import load_dotenv
from bs4 import BeautifulSoup

from playwright.async_api import async_playwright
from unstructured.partition.html import partition_html
from unstructured.chunking.title import chunk_by_title

from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_google_genai import (
    GoogleGenerativeAIEmbeddings,
    ChatGoogleGenerativeAI
)

# =========================================================
# ENV SETUP
# =========================================================

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not found")

# =========================================================
# LLM (Vision-capable)
# =========================================================

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0
)

# =========================================================
# GENERIC CONTENT-DENSITY CRAWLER
# =========================================================

class WebsiteCrawler:
    def __init__(self, base_url, max_depth=2, max_pages=50):
        self.base_url = base_url
        self.base_domain = urlparse(base_url).netloc
        self.max_depth = max_depth
        self.max_pages = max_pages
        self.visited = set()
        self.pages = []

    def extract_main_content(self, soup):

        # Remove obvious non-content elements
        for tag in soup.find_all(["nav", "header", "footer", "script", "style"]):
            tag.decompose()

        candidates = []

        for tag in soup.find_all(["article", "main", "section", "div"]):
            text = tag.get_text(strip=True)
            text_length = len(text)

            if text_length < 300:
                continue

            link_count = len(tag.find_all("a"))
            score = text_length - (link_count * 50)

            candidates.append((score, tag))

        if not candidates:
            return None

        return max(candidates, key=lambda x: x[0])[1]

    async def crawl(self):
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)

            queue = [(self.base_url, 0)]

            while queue and len(self.visited) < self.max_pages:
                url, depth = queue.pop(0)

                if url in self.visited or depth > self.max_depth:
                    continue

                print(f"Crawling: {url}")
                self.visited.add(url)

                try:
                    page = await browser.new_page()
                    await page.goto(url, timeout=45000)
                    await page.wait_for_load_state("domcontentloaded")
                    html = await page.content()
                    await page.close()

                    soup = BeautifulSoup(html, "html.parser")
                    main_content = self.extract_main_content(soup)

                    if not main_content:
                        continue

                    self.pages.append({
                        "url": url,
                        "html": str(main_content)
                    })

                    # Traverse only content hyperlinks
                    for link in main_content.find_all("a", href=True):
                        full_url = urljoin(url, link["href"])
                        parsed = urlparse(full_url)

                        if parsed.netloc != self.base_domain:
                            continue

                        clean_url = parsed._replace(
                            fragment="",
                            query=""
                        ).geturl()

                        if any(x in clean_url.lower() for x in [
                            "login", "signup", "privacy",
                            "terms", "mailto:", "tel:"
                        ]):
                            continue

                        if clean_url not in self.visited:
                            queue.append((clean_url, depth + 1))

                except Exception as e:
                    print(f"Failed: {url} | {e}")

            await browser.close()

        return self.pages

# =========================================================
# PARSE + CHUNK
# =========================================================

def parse_and_chunk(page_data):

    chunks_data = []

    for page in page_data:
        elements = partition_html(text=page["html"])

        chunks = chunk_by_title(
            elements,
            max_characters=2000,
            new_after_n_chars=1500,
            combine_text_under_n_chars=300
        )

        for chunk in chunks:
            if chunk.text and len(chunk.text.strip()) > 100:
                chunks_data.append({
                    "url": page["url"],
                    "text": chunk.text,
                    "elements": chunk.metadata.orig_elements
                })

    return chunks_data

# =========================================================
# IMAGE DOWNLOAD
# =========================================================

def download_image_as_base64(image_url):
    try:
        response = requests.get(image_url, timeout=10)
        response.raise_for_status()
        return base64.b64encode(response.content).decode("utf-8")
    except:
        return None

# =========================================================
# EXTRACT CONTENT
# =========================================================

def extract_content(chunks):

    text_docs = []
    image_docs = []

    for chunk in chunks:

        chunk_id = str(uuid.uuid4())
        tables = []
        image_urls = []

        for el in chunk["elements"]:
            el_type = type(el).__name__

            if el_type == "Table":
                tables.append(el.text)

            elif el_type == "Image":
                if hasattr(el.metadata, "image_url"):
                    image_urls.append(el.metadata.image_url)

        combined_text = chunk["text"]

        if tables:
            combined_text += "\n\nTABLES:\n"
            for table in tables:
                combined_text += table + "\n"

        text_docs.append(
            Document(
                page_content=combined_text,
                metadata={
                    "source_url": chunk["url"],
                    "chunk_id": chunk_id,
                    "type": "text"
                }
            )
        )

        for img_url in image_urls:
            img_base64 = download_image_as_base64(img_url)

            if img_base64:
                image_docs.append(
                    Document(
                        page_content=img_base64,
                        metadata={
                            "source_url": chunk["url"],
                            "chunk_id": chunk_id,
                            "image_url": img_url,
                            "type": "image"
                        }
                    )
                )

    return text_docs, image_docs

# =========================================================
# CREATE VECTOR STORES
# =========================================================

def create_vector_stores(text_docs, image_docs):

    if not text_docs:
        print("No valid text documents found.")
        return None, None

    text_embedding = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001"
    )

    image_embedding = GoogleGenerativeAIEmbeddings(
        model="models/multimodalembedding"
    )

    text_store = Chroma.from_documents(
        documents=text_docs,
        embedding=text_embedding,
        persist_directory="./chroma_text"
    )

    image_store = None

    if image_docs:
        image_store = Chroma.from_documents(
            documents=image_docs,
            embedding=image_embedding,
            persist_directory="./chroma_image"
        )

    text_store.persist()
    if image_store:
        image_store.persist()

    return text_store, image_store

# =========================================================
# MULTIMODAL QUESTION ANSWERING
# =========================================================

def ask_question(text_store, image_store, query):

    if not text_store:
        return "Vector store not available."

    text_results = text_store.similarity_search(query, k=4)

    text_context = "\n\n".join([doc.page_content for doc in text_results])

    image_analysis = ""

    if image_store:
        image_results = image_store.similarity_search(query, k=2)

        for img_doc in image_results:
            img_base64 = img_doc.page_content

            response = llm.invoke([
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Analyze and explain this image."},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{img_base64}"
                            }
                        }
                    ]
                }
            ])

            image_analysis += response.content + "\n\n"

    final_prompt = f"""
    Use the text context and image analysis below to answer the question.

    TEXT CONTEXT:
    {text_context}

    IMAGE ANALYSIS:
    {image_analysis}

    QUESTION:
    {query}
    """

    final_answer = llm.invoke(final_prompt)

    return final_answer.content

# =========================================================
# MAIN
# =========================================================

async def main():

    url = "https://google.github.io/adk-docs/"

    crawler = WebsiteCrawler(
        base_url=url,
        max_depth=2,
        max_pages=50
    )

    pages = await crawler.crawl()
    print("Total pages crawled:", len(pages))

    chunks = parse_and_chunk(pages)
    text_docs, image_docs = extract_content(chunks)

    text_store, image_store = create_vector_stores(text_docs, image_docs)

    answer = ask_question(
        text_store,
        image_store,
        "Explain the architecture diagram"
    )

    print("\nFINAL ANSWER:\n", answer)

if __name__ == "__main__":
    asyncio.run(main())


