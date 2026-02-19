# tools/config.py

import vertexai
from vertexai.generative_models import GenerativeModel
from vertexai.language_models import TextEmbeddingModel
from vertexai.preview.vision_models import MultiModalEmbeddingModel

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
