"""Web search provider implementations (Brave, Firecrawl)."""

import asyncio
import hashlib
import logging
import os
import re
from urllib.parse import urlparse
from typing import List, Dict

import httpx
from diskcache import Cache

logger = logging.getLogger("ResearchAgent")

# Domains that consistently fail scraping (JS-only, paywalled, or block bots).
# Results from these domains are filtered out before scraping to avoid wasting slots.
BLOCKED_DOMAINS = {
    "reddit.com", "www.reddit.com",
    "facebook.com", "www.facebook.com",
    "youtube.com", "www.youtube.com", "m.youtube.com",
    "twitter.com", "x.com",
    "instagram.com", "www.instagram.com",
    "tiktok.com", "www.tiktok.com",
    "linkedin.com", "www.linkedin.com",
    "forbes.com", "www.forbes.com",
    "pinterest.com", "www.pinterest.com",
    "quora.com", "www.quora.com",
}


def is_blocked_url(url: str) -> bool:
    """Return True if the URL belongs to a domain known to block scraping."""
    try:
        domain = urlparse(url).netloc.lower()
        return domain in BLOCKED_DOMAINS
    except Exception:
        return False


def filter_search_results(results: List[Dict]) -> List[Dict]:
    """Remove results from blocked domains, logging each rejection."""
    filtered = []
    for r in results:
        url = r.get("url", "")
        if is_blocked_url(url):
            logger.info(f"  [FILTER] Blocked: {url}")
        else:
            filtered.append(r)
    return filtered


async def search_brave(query: str, limit: int = 7) -> List[Dict]:
    """Query the Brave Search API and return a list of {url, title, snippet} dicts."""
    api_key = os.getenv("BRAVE_API_KEY")
    if not api_key:
        raise RuntimeError("BRAVE_API_KEY not set")
    async with httpx.AsyncClient(timeout=15) as client:
        r = await client.get(
            "https://api.search.brave.com/res/v1/web/search",
            params={"q": query, "count": limit, "text_decorations": False},
            headers={"Accept": "application/json", "X-Subscription-Token": api_key},
        )
        r.raise_for_status()
        data = r.json()
    return [
        {
            "url": item.get("url", ""),
            "title": item.get("title", ""),
            "snippet": item.get("description", ""),
        }
        for item in data.get("web", {}).get("results", [])
    ]


async def search_firecrawl(query: str, limit: int = 7) -> List[Dict]:
    """Query the Firecrawl API and return a list of {url, title, snippet} dicts."""
    from firecrawl import Firecrawl

    fc = Firecrawl(api_key=os.getenv("FIRECRAWL_API_KEY"))
    # Request extra results to compensate for blocked domains being filtered out
    fetch_limit = limit + 3
    res = await asyncio.to_thread(
        fc.search, query, limit=fetch_limit,
        scrape_options={"formats": ["markdown"], "waitFor": 3000},
    )

    raw_web = []
    if hasattr(res, "web") and res.web:
        raw_web = res.web
    elif hasattr(res, "data") and isinstance(res.data, list):
        raw_web = res.data
    elif isinstance(res, list):
        raw_web = res
    elif isinstance(res, dict):
        raw_web = res.get("web", []) or res.get("data", [])

    pages = []
    for item in raw_web:
        if item is None:
            continue
        if isinstance(item, dict):
            url = item.get("url", "")
            title = item.get("title", "")
            snip = item.get("markdown", "") or item.get("description", "")
        elif hasattr(item, "url"):
            url = getattr(item, "url", "") or ""
            title = getattr(item, "title", "") or ""
            snip = getattr(item, "markdown", "") or getattr(item, "description", "") or ""
        else:
            s = str(item)
            url_m = re.search(r"url='([^']+)'", s)
            title_m = re.search(r"title='([^']+)'", s)
            desc_m = re.search(r"description='([^']+)'", s)
            url = url_m.group(1) if url_m else ""
            title = title_m.group(1) if title_m else ""
            snip = desc_m.group(1) if desc_m else ""
        if url:
            pages.append({"url": url, "title": title, "snippet": snip})

    logger.info(f"  [FIRECRAWL] parsed {len(pages)} pages")
    return pages


async def do_search(query: str, search_backend: str, cache: Cache, limit: int = 7) -> List[Dict]:
    """Run a web search using the configured backend, with caching."""
    cache_key = hashlib.md5(f"search:{search_backend}:{query}:{limit}".encode()).hexdigest()
    if cached := cache.get(cache_key):
        logger.info(f"  [SEARCH CACHE HIT] {query}")
        return cached

    logger.info(f"  [SEARCH/{search_backend.upper()}] {query}")
    try:
        if search_backend == "brave":
            results = await search_brave(query, limit)
        else:
            results = await search_firecrawl(query, limit)
        results = filter_search_results(results)
        cache.set(cache_key, results, expire=3600)
        return results
    except Exception as e:
        logger.error(f"  [SEARCH ERROR] {query}: {e}")
        return []
