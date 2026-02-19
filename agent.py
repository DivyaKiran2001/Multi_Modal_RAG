# agent.py

from google.adk.agents import Agent
from .tools.crawler_tool import crawl_website
from .tools.parser_tool import parse_pages
from .tools.store_tool import store_instance
from .tools.qa_tool import ask_question_tool

# -------------------------------
# STATIC URL (FIXED)
# -------------------------------

STATIC_URL = "https://docs.cloud.google.com/agent-builder/agent-engine/overview"

# Track whether index is already built
_index_built = False


# -------------------------------
# INTERNAL INDEX BUILDER
# -------------------------------

async def _ensure_index():
    global _index_built

    if _index_built:
        return

    crawl_data = await crawl_website(STATIC_URL)
    parsed = parse_pages(crawl_data["pages"])

    await store_instance.build(
        parsed["chunks"],
        parsed["images"]
    )

    _index_built = True


# -------------------------------
# PUBLIC QA TOOL
# -------------------------------

async def static_qa_tool(query: str) -> dict:
    """
    Answers questions using pre-indexed static documentation.
    Automatically builds index if not already built.
    """

    await _ensure_index()

    result = await ask_question_tool(query)

    return result


# -------------------------------
# ROOT AGENT
# -------------------------------

root_agent = Agent(
    name="agent_engine_doc_assistant",
    model="gemini-2.5-flash",
    description="Answers questions about Vertex AI Agent Engine documentation.",
    instruction="""
You are an expert assistant for Vertex AI Agent Engine documentation.

Answer user questions strictly using the indexed documentation content.

Do not ask for URLs.
Do not attempt to crawl new websites.
Only answer questions about Agent Engine documentation.
""",
    tools=[static_qa_tool],
)
