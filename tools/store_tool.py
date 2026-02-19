# tools/store_tool.py

import time
import faiss
import numpy as np
import requests
from vertexai.preview.vision_models import Image
from .config import *


class MultiModalStore:

    def __init__(self):
        self.text_index = None
        self.image_index = None
        self.text_data = []
        self.image_data = []

    # ---------------- TEXT EMBEDDING ----------------

    def embed_text_batch(self, texts):

        vectors = []

        for i in range(0, len(texts), TEXT_BATCH_SIZE):
            batch = texts[i:i + TEXT_BATCH_SIZE]
            time.sleep(REQUEST_DELAY)

            response = text_embedding_model.get_embeddings(batch)

            for emb in response:
                vectors.append(np.array(emb.values).astype("float32"))

        return vectors

    # ---------------- IMAGE EMBEDDING ----------------

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

    # ---------------- QUERY EMBEDDING FOR IMAGE ----------------

    def embed_query_for_image(self, query):
        time.sleep(REQUEST_DELAY)

        emb = multimodal_model.get_embeddings(
            contextual_text=query
        )

        return np.array(emb.text_embedding).astype("float32")

    # ---------------- ROUTER ----------------

    def route_query(self, query: str) -> str:
        """
        Uses Gemini to classify whether the query
        needs TEXT, IMAGE, or BOTH retrieval.
        """

        prompt = f"""
You are a query router.

Classify the user query into one of these categories:
TEXT   -> needs textual explanation only
IMAGE  -> refers specifically to diagrams, figures, or visuals
BOTH   -> requires both text and image understanding

Return ONLY one word: TEXT, IMAGE, or BOTH.

Query: {query}
"""

        response = gemini_model.generate_content(prompt)
        decision = response.text.strip().upper()

        if decision not in ["TEXT", "IMAGE", "BOTH"]:
            return "BOTH"

        return decision

    # ---------------- BUILD INDEXES ----------------

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

            if len(image_vectors.shape) == 1:
                image_vectors = image_vectors.reshape(1, -1)

            faiss.normalize_L2(image_vectors)

            self.image_index = faiss.IndexFlatIP(image_vectors.shape[1])
            self.image_index.add(image_vectors)

            self.image_data = valid_images

        else:
            print("No valid image embeddings created.")

    # ---------------- SEARCH ----------------

    def search(self, query, route):

        results = []

        # ---- TEXT SEARCH ----
        if route in ["TEXT", "BOTH"]:

            q_text = self.embed_text_batch([query])[0].reshape(1, -1)
            faiss.normalize_L2(q_text)

            if self.text_index:
                _, idx = self.text_index.search(q_text, TOP_K_TEXT)

                for i in idx[0]:
                    results.append({
                        "type": "text",
                        "content": self.text_data[i]["text"]
                    })

        # ---- IMAGE SEARCH ----
        if route in ["IMAGE", "BOTH"] and self.image_index:

            q_image = self.embed_query_for_image(query).reshape(1, -1)
            faiss.normalize_L2(q_image)

            _, idx = self.image_index.search(q_image, TOP_K_IMAGES)

            for i in idx[0]:
                results.append({
                    "type": "image",
                    "content": self.image_data[i]
                })

        return results


# Singleton instance
store_instance = MultiModalStore()
