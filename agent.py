# agent.py

from google.adk.agents import Agent
from .tools.crawler_tool import crawl_website
from .tools.parser_tool import parse_pages
from .tools.store_tool import store_instance
from .tools.qa_tool import ask_question_tool



async def build_index_tool(url: str) -> dict:
    """
    Crawls a website and builds multimodal index
    """
    crawl_data = await crawl_website(url)
    parsed = parse_pages(crawl_data["pages"])

    await store_instance.build(
        parsed["chunks"],
        parsed["images"]
    )

    return {"status": "Index built successfully"}


root_agent = Agent(
    name="multimodal_adk_web_agent",
    model="gemini-2.5-flash",
    description="Multimodal ADK Web Agent with crawler, text+image RAG",
    instruction="""
You are a multimodal web research agent.

You have access to tools:
- build_index_tool → Crawl and index a website
- ask_question_tool → Ask questions over the indexed website

Always build index before answering questions.
""",
    tools=[
        build_index_tool,
        ask_question_tool
    ],
)
