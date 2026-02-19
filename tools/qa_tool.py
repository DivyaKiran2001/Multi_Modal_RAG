# tools/qa_tool.py

import requests
from vertexai.generative_models import Part
from .config import gemini_model
from .store_tool import store_instance

async def ask_question_tool(query: str) -> dict:

    route = store_instance.route_query(query)
    results = store_instance.search(query, route)

    content_parts = []

    if route == "TEXT":
        instruction = (
            "Answer using ONLY the provided text context.\n\n"
            f"QUESTION:\n{query}"
        )
    elif route == "IMAGE":
        instruction = (
            "Answer using ONLY the provided images.\n\n"
            f"QUESTION:\n{query}"
        )
    else:
        instruction = (
            "Answer using both text and images if needed.\n\n"
            f"QUESTION:\n{query}"
        )

    content_parts.append(Part.from_text(instruction))

    text_context = "\n".join(
        r["content"] for r in results if r["type"] == "text"
    )

    if text_context:
        content_parts.append(Part.from_text(f"\nContext:\n{text_context}"))

    for r in results:
        if r["type"] == "image":
            try:
                img_response = requests.get(r["content"])
                content_parts.append(
                    Part.from_data(
                        img_response.content,
                        mime_type="image/png"
                    )
                )
            except:
                continue

    response = gemini_model.generate_content(content_parts)

    return {"answer": response.text}
