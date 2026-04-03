"""HTML fetching and text extraction utilities."""

import asyncio
import hashlib
import logging
import re
from typing import Optional

import httpx
from bs4 import BeautifulSoup
from diskcache import Cache

logger = logging.getLogger("ResearchAgent")

SCRAPE_SEMAPHORE = asyncio.Semaphore(5)
USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)


def clean_html_to_text(html: str) -> str:
    """Strip boilerplate HTML elements and return clean body text."""
    soup = BeautifulSoup(html, "lxml")

    for tag in soup.find_all(
        ["script", "style", "nav", "footer", "aside", "iframe",
         "noscript", "svg", "button", "input", "select"]
    ):
        tag.decompose()

    for el in soup.find_all(
        class_=re.compile(
            r"(sidebar|cookie|banner|popup|modal|advertisement|social-share|newsletter|related-posts)",
            re.I,
        )
    ):
        el.decompose()

    body = soup.find("body") or soup
    text = body.get_text(separator="\n", strip=True)

    lines = [line.strip() for line in text.split("\n") if len(line.strip()) >= 3]

    deduped = []
    for line in lines:
        if not deduped or line != deduped[-1]:
            deduped.append(line)

    return "\n".join(deduped)


async def scrape_url(url: str, cache: Cache) -> Optional[str]:
    """Fetch a URL, extract clean text, and cache the result for 1 hour."""
    cache_key = hashlib.md5(f"scrape:{url}".encode()).hexdigest()
    if cached := cache.get(cache_key):
        return cached

    async with SCRAPE_SEMAPHORE:
        try:
            async with httpx.AsyncClient(timeout=7, follow_redirects=True) as client:
                r = await client.get(url, headers={"User-Agent": USER_AGENT})
                r.raise_for_status()

            content_type = r.headers.get("content-type", "")
            if "text/html" not in content_type:
                logger.info(f"  [SCRAPE SKIP] {url} — not HTML ({content_type[:40]})")
                return None

            text = clean_html_to_text(r.text)
            if len(text) < 50:
                logger.info(f"  [SCRAPE SKIP] {url} — too little content after cleaning")
                return None

            text = text[:15000]
            cache.set(cache_key, text, expire=3600)
            logger.info(f"  [SCRAPED] {url} — {len(text)} chars")
            return text
        except Exception as e:
            logger.warning(f"  [SCRAPE FAIL] {url}: {e}")
            return None
