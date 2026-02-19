# tools/crawler_tool.py

import asyncio
from urllib.parse import urlparse, urljoin
from bs4 import BeautifulSoup
from playwright.async_api import async_playwright
import trafilatura

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

                self.visited.add(url)

                try:
                    page = await browser.new_page()
                    await page.goto(url, timeout=120000, wait_until="domcontentloaded")
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
                                clean = parsed._replace(fragment="", query="").geturl()
                                if clean not in self.visited:
                                    queue.append((clean, depth + 1))

                except Exception:
                    continue

            await browser.close()

        return self.pages


async def crawl_website(url: str) -> dict:
    crawler = WebsiteCrawler(url)
    pages = await crawler.crawl()
    return {"pages": pages}
