"""Dynamic Research Agent — FastAPI server with agentic search pipeline.

Supports three query types (ranked, informational, navigational) and streams
results progressively via Server-Sent Events so the frontend can render data
as it arrives.
"""

import os
import time
import asyncio
import json
import hashlib
import argparse
import logging
from typing import List, Dict, AsyncGenerator
from collections import defaultdict

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from diskcache import Cache
from dotenv import load_dotenv

from utils.llm import (
    make_llm_client, llm_complete, parse_json,
    MAX_SUB_QUERY_TOKENS, MAX_SCHEMA_TOKENS, MAX_EXTRACT_TOKENS, MAX_BACKFILL_TOKENS,
)
from prompts import (
    ROLE_CLASSIFIER, ROLE_DATA_EXTRACTOR,
    ROLE_RANKED_EXTRACTOR, ROLE_BACKFILL_AGENT,
    CLASSIFY_AND_SCHEMA_SYSTEM, EXTRACT_SYSTEM,
    RANKED_EXTRACT_SYSTEM, BACKFILL_SYSTEM,
)
from utils.scraper import scrape_url
from utils.search import do_search
from utils.dedup import deduplicate_entities

# ── LLM Call Budget ──────────────────────────────────────────────────────────
MAX_LLM_CALLS_PER_REQUEST = 6

# Max backfill API calls (search + scrape + LLM) per /api/search request.
# Each backfill fills ONE missing cell. Keep low to avoid runaway costs.
MAX_BACKFILLS_PER_REQUEST = 3

load_dotenv()

# ── CLI Args ─────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Dynamic Research Agent")
parser.add_argument("--llm", choices=["ollama", "openrouter"], default="ollama",
                    help="LLM backend (default: ollama)")
parser.add_argument("--search", choices=["brave", "firecrawl"], default="firecrawl",
                    help="Search backend (default: firecrawl)")
parser.add_argument("--ollama-model", default=os.getenv("OLLAMA_MODEL", "llama3.2"),
                    help="Ollama model name (default: llama3.2)")
parser.add_argument("--ollama-url", default=os.getenv("OLLAMA_URL", "http://localhost:11434/v1"),
                    help="Ollama base URL")
parser.add_argument("--port", type=int, default=8000)
args, _ = parser.parse_known_args()

# ── App & Cache ──────────────────────────────────────────────────────────────
app = FastAPI(title="Dynamic Research Agent v4")
cache = Cache("agent_cache")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler("agent_search.log", mode="a"), logging.StreamHandler()],
)
logger = logging.getLogger("ResearchAgent")
logger.info(f"Config: llm={args.llm}  search={args.search}")

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# ── LLM Client ───────────────────────────────────────────────────────────────
llm_client, llm_model = make_llm_client(args)

# ── Rate Limiting ────────────────────────────────────────────────────────────
RATE_LIMIT = int(os.getenv("RATE_LIMIT_PER_MIN", "5"))
_rate_store: Dict[str, List[float]] = defaultdict(list)


def check_rate_limit(ip: str):
    """Enforce a sliding-window rate limit of N requests per minute per IP."""
    now = time.time()
    window = [t for t in _rate_store[ip] if now - t < 60]
    _rate_store[ip] = window
    if len(window) >= RATE_LIMIT:
        raise HTTPException(status_code=429, detail=f"Rate limit exceeded. Max {RATE_LIMIT} searches/minute.")
    _rate_store[ip].append(now)


# ── LLM Budget ───────────────────────────────────────────────────────────────

class LLMBudget:
    """Tracks LLM calls within a single request to enforce a global cap."""
    def __init__(self, limit: int):
        self.limit = limit
        self.calls = 0

    def exhausted(self) -> bool:
        return self.calls >= self.limit

    def call(self, messages, json_mode=True, max_tokens=1500) -> str:
        """Make an LLM call, raising if the budget is exhausted."""
        if self.exhausted():
            raise RuntimeError(f"LLM call budget exhausted ({self.limit} calls)")
        self.calls += 1
        logger.info(f"  [LLM CALL {self.calls}/{self.limit}]")
        return llm_complete(llm_client, llm_model, args.llm, messages, json_mode, max_tokens)


def _llm(messages, json_mode=True, max_tokens=1500):
    """Shorthand wrapper for non-budgeted calls (backfill endpoint)."""
    return llm_complete(llm_client, llm_model, args.llm, messages, json_mode, max_tokens)


# ── SSE Helper ───────────────────────────────────────────────────────────────

def sse_event(event: str, data: dict) -> str:
    """Format a Server-Sent Event message."""
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


# ── Pipeline Step 1: Classify + Schema + Sub-Queries (1 LLM call) ───────────

async def classify_and_plan(query: str, budget: LLMBudget) -> Dict:
    """Classify query, generate sub-queries, and design schema in a single LLM call."""
    logger.info(f"Classifying + schema: {query}  [{args.llm}/{llm_model}]")
    raw = budget.call(
        messages=[
            {"role": "system", "content": f"{ROLE_CLASSIFIER}\n\n{CLASSIFY_AND_SCHEMA_SYSTEM}"},
            {"role": "user", "content": f"Topic: {query}"},
        ],
        max_tokens=MAX_SCHEMA_TOKENS,
    )
    result = parse_json(raw)
    logger.info(f"  Type: {result.get('query_type')}  Metric: {result.get('ranking_metric')}")
    logger.info(f"  Sub-Queries: {result.get('sub_queries')}")
    logger.info(f"  Schema: {[f.get('display_name') for f in result.get('schema', [])]}")
    return result


# ── Search + Scrape (no LLM calls) ──────────────────────────────────────────

async def search_and_scrape(sub_queries: List[str], max_pages: int = 10) -> tuple[List[Dict], int]:
    """Run a single search query, deduplicate URLs, scrape content. Returns (pages_with_content, scraped_count)."""
    # Use only the first sub-query to minimize API calls (1 search call instead of 3)
    best_query = sub_queries[0] if sub_queries else ""
    search_results_list = await do_search(best_query, args.search, cache, limit=max_pages)
    logger.info(f"  [SEARCH] '{best_query}' -> {len(search_results_list)} results")

    unique_pages: Dict[str, Dict] = {}
    for p in search_results_list:
        url = p.get("url")
        if url and url not in unique_pages:
            unique_pages[url] = p
            if len(unique_pages) >= max_pages:
                break

    logger.info(f"Total: {len(unique_pages)} unique URLs (capped at {max_pages})")

    if not unique_pages:
        return [], 0

    # Prefer Firecrawl markdown content over re-scraping
    urls_to_scrape = []
    prefetched = {}
    for url, page_meta in unique_pages.items():
        snippet = page_meta.get("snippet", "")
        if len(snippet) >= 200:
            prefetched[url] = snippet
        else:
            urls_to_scrape.append(url)

    scrape_results = await asyncio.gather(*[scrape_url(u, cache) for u in urls_to_scrape])
    scraped_map = dict(zip(urls_to_scrape, scrape_results))

    pages_with_content = []
    for url, page_meta in unique_pages.items():
        text = prefetched.get(url) or scraped_map.get(url) or page_meta.get("snippet", "")
        if text and len(text) >= 100:
            pages_with_content.append({"url": url, "content": text})

    scraped_count = sum(1 for v in scraped_map.values() if v) + len(prefetched)
    logger.info(
        f"Content: {len(prefetched)} from search, "
        f"{sum(1 for v in scraped_map.values() if v)}/{len(urls_to_scrape)} scraped, "
        f"{len(pages_with_content)} usable"
    )
    return pages_with_content, scraped_count


# ── Extraction Helpers ───────────────────────────────────────────────────────

CHUNK_SIZE = 5000
MAX_CHUNKS = 3


def _build_schema_lookup(schema: List[Dict]) -> Dict[str, str]:
    """Build a case-insensitive lookup from field key variants to canonical names."""
    lookup = {}
    for sf in schema:
        lookup[sf["name"].lower()] = sf["name"]
        lookup[sf.get("display_name", "").lower()] = sf["name"]
        lookup[sf["name"].lower().replace("_", " ")] = sf["name"]
        lookup[sf["name"].lower().replace("_", "")] = sf["name"]
    return lookup


def _normalize_entities(entities: List[Dict], url: str, schema_lookup: Dict[str, str]) -> List[Dict]:
    """Attach source URL and normalize field keys to match schema names."""
    for ent in entities:
        ent["source_url"] = url
        normalized = {}
        for key, field in ent.get("fields", {}).items():
            canonical = schema_lookup.get(key.lower(), key)
            if isinstance(field, dict):
                field["sources"] = [{"url": url}]
            normalized[canonical] = field
        ent["fields"] = normalized
    return entities


async def extract_ranked_names(
    pages: List[Dict], query: str, ranking_metric: str, budget: LLMBudget
) -> List[Dict]:
    """Lightweight extraction: get entity names + metric values from top pages in 1 call."""
    if budget.exhausted():
        return []

    # Combine top 2-3 pages into a single extraction call
    combined_parts = []
    for p in pages[:3]:
        combined_parts.append(f"--- SOURCE: {p['url']} ---\n{p['content'][:3000]}")
    combined_text = "\n\n".join(combined_parts)

    try:
        raw = budget.call(
            messages=[
                {"role": "system", "content": f"{ROLE_RANKED_EXTRACTOR}\n\n{RANKED_EXTRACT_SYSTEM}"},
                {"role": "user", "content": (
                    f"Topic: '{query}'\nRanking metric: {ranking_metric}\n\n"
                    f"Page contents:\n{combined_text}"
                )},
            ],
            max_tokens=1500,
        )
        all_names = parse_json(raw).get("entities", [])
        for ent in all_names:
            if not ent.get("source_url"):
                ent["source_url"] = pages[0]["url"] if pages else ""
        logger.info(f"  [RANKED EXTRACT] {len(all_names)} names from {len(pages[:3])} pages")
    except Exception as e:
        logger.error(f"  [RANKED EXTRACT ERROR]: {e}")
        return []

    # Deduplicate by name
    seen = set()
    unique = []
    for ent in all_names:
        name = (ent.get("name") or "").strip().lower()
        if name and name not in seen:
            seen.add(name)
            unique.append(ent)
    return unique[:5]


async def extract_full(
    pages: List[Dict], query: str, schema: List[Dict], budget: LLMBudget
) -> List[Dict]:
    """Batched extraction — combines multiple pages into fewer LLM calls.

    Concatenates page content (truncated) and sends in batches of ~8K chars
    to minimize LLM calls while staying within context limits.
    """
    schema_desc = json.dumps(schema)
    schema_lookup = _build_schema_lookup(schema)
    all_entities = []

    # Check cache first
    uncached_pages = []
    for p in pages:
        cache_key = hashlib.md5(f"extract:{p['url']}:{query}".encode()).hexdigest()
        if cached := cache.get(cache_key):
            logger.info(f"  [EXTRACT CACHE HIT] {p['url']}")
            all_entities.extend(cached)
        else:
            uncached_pages.append(p)

    if not uncached_pages or budget.exhausted():
        return all_entities

    # Batch pages together — combine content with source markers
    BATCH_CHAR_LIMIT = 8000
    batches = []
    current_batch = []
    current_chars = 0

    for p in uncached_pages:
        # Truncate each page to fit more pages per batch
        per_page_limit = BATCH_CHAR_LIMIT // min(len(uncached_pages), 3)
        page_text = f"\n--- SOURCE: {p['url']} ---\n{p['content'][:per_page_limit]}"
        if current_chars + len(page_text) > BATCH_CHAR_LIMIT and current_batch:
            batches.append(current_batch)
            current_batch = []
            current_chars = 0
        current_batch.append({"page": p, "text": page_text})
        current_chars += len(page_text)

    if current_batch:
        batches.append(current_batch)

    for batch in batches:
        if budget.exhausted():
            logger.warning(f"  [EXTRACT] Budget exhausted, skipping remaining batches")
            break

        combined_text = "".join(item["text"] for item in batch)
        batch_urls = [item["page"]["url"] for item in batch]
        logger.info(f"  [EXTRACT BATCH] {len(batch)} pages, {len(combined_text)} chars")

        try:
            raw = budget.call(
                messages=[
                    {"role": "system", "content": f"{ROLE_DATA_EXTRACTOR}\n\n{EXTRACT_SYSTEM}"},
                    {"role": "user", "content": (
                        f"Topic: '{query}'\nSchema: {schema_desc}\n"
                        f"Source URLs: {json.dumps(batch_urls)}\n\n"
                        f"Page contents:\n{combined_text}"
                    )},
                ],
                max_tokens=MAX_EXTRACT_TOKENS,
            )
            entities = parse_json(raw).get("entities", [])

            # Normalize — try to match source URL from entity or default to first page
            for ent in entities:
                source = batch_urls[0]
                # Check if entity has source info
                for field in ent.get("fields", {}).values():
                    if isinstance(field, dict):
                        for s in field.get("sources", []):
                            if s.get("url") in batch_urls:
                                source = s["url"]
                                break
                ent = _normalize_entities([ent], source, schema_lookup)[0]

            entities = _normalize_entities(entities, batch_urls[0], schema_lookup)
            all_entities.extend(entities)
            logger.info(f"  [EXTRACT BATCH] -> {len(entities)} entities")

            # Cache per-page (approximate: assign all entities to first page in batch)
            for item in batch:
                cache_key = hashlib.md5(f"extract:{item['page']['url']}:{query}".encode()).hexdigest()
                page_entities = [e for e in entities if e.get("source_url") == item["page"]["url"]]
                if page_entities:
                    cache.set(cache_key, page_entities, expire=86400)

        except RuntimeError:
            break
        except Exception as e:
            logger.error(f"  [LLM ERROR] batch extraction: {e}")

    return all_entities


def _column_has_data(field_name: str, display_name: str, entities: List[Dict]) -> bool:
    """Check whether at least one entity has a non-null value for a given column."""
    for ent in entities:
        fields = ent.get("fields", {})
        cell = fields.get(field_name) or fields.get(display_name)
        if not cell:
            for k, v in fields.items():
                if k.lower().replace(" ", "_") == field_name.lower() or k.lower() == display_name.lower():
                    cell = v
                    break
        if isinstance(cell, dict) and cell.get("value"):
            return True
        elif cell and not isinstance(cell, dict):
            return True
    return False


def _count_missing_fields(ent: Dict, schema: List[Dict]) -> int:
    """Count how many schema fields (excluding the first) are missing a value."""
    fields = ent.get("fields", {})
    missing = 0
    for sf in schema[1:]:
        field = fields.get(sf["name"])
        if not field or not (isinstance(field, dict) and field.get("value")):
            missing += 1
    return missing


def _collect_backfill_tasks(entities: List[Dict], schema: List[Dict], max_total: int) -> List[Dict]:
    """Collect the most impactful backfill tasks up to a cap.

    Prioritizes entities with the most missing fields, but skips entities
    that are already mostly filled (<=1 missing field).
    """
    tasks = []
    for ei, ent in enumerate(entities):
        entity_name = None
        first_field = list(ent.get("fields", {}).values())
        if first_field and isinstance(first_field[0], dict):
            entity_name = first_field[0].get("value")
        if not entity_name:
            continue

        missing = _count_missing_fields(ent, schema)
        if missing <= 1:
            continue  # entity is already mostly filled, not worth backfilling

        for sf in schema[1:]:
            if len(tasks) >= max_total:
                return tasks
            field = ent["fields"].get(sf["name"])
            if field and isinstance(field, dict) and field.get("value"):
                continue
            # Skip fields that are redundant with the entity name
            if entity_name.lower() in sf.get("display_name", "").lower() or \
               sf.get("display_name", "").lower() in entity_name.lower():
                continue
            tasks.append({"ei": ei, "sf": sf, "entity_name": entity_name, "ent": ent})

    return tasks


def _prune_schema(schema: List[Dict], entities: List[Dict]) -> List[Dict]:
    """Remove schema columns that have no data across all entities."""
    active = []
    for i, sf in enumerate(schema):
        if i == 0 or _column_has_data(sf["name"], sf.get("display_name", ""), entities):
            active.append(sf)
    return active


def _ranked_to_entities(ranked: List[Dict], schema: List[Dict], ranking_metric: str) -> List[Dict]:
    """Convert lightweight ranked extraction results into full entity format."""
    entities = []
    # Find the schema field that best matches the ranking metric
    metric_field = schema[0]["name"]  # fallback
    for sf in schema:
        if ranking_metric.lower() in sf.get("display_name", "").lower() or \
           ranking_metric.lower() in sf["name"].lower().replace("_", " "):
            metric_field = sf["name"]
            break

    name_field = schema[0]["name"] if schema else "name"

    for r in ranked:
        fields = {
            name_field: {
                "value": r.get("name"),
                "evidence": r.get("evidence", ""),
                "sources": [{"url": r.get("source_url", "")}],
            }
        }
        if r.get("metric_value") and metric_field != name_field:
            fields[metric_field] = {
                "value": r["metric_value"],
                "evidence": r.get("evidence", ""),
                "sources": [{"url": r.get("source_url", "")}],
            }
        entities.append({"fields": fields, "source_url": r.get("source_url", "")})
    return entities


# ── Request Models ───────────────────────────────────────────────────────────

class SearchRequest(BaseModel):
    query: str


class BackfillRequest(BaseModel):
    query: str
    entity_name: str
    field_name: str
    field_display_name: str


# ── SSE Streaming Search Endpoint ────────────────────────────────────────────

@app.post("/api/search")
async def agentic_search(request: SearchRequest, req: Request):
    """SSE streaming search: sends progressive results as events."""
    client_ip = req.client.host if req.client else "unknown"
    check_rate_limit(client_ip)

    async def generate() -> AsyncGenerator[str, None]:
        start_time = time.time()
        budget = LLMBudget(MAX_LLM_CALLS_PER_REQUEST)

        try:
            # ── Step 1: Classify + Schema + Sub-Queries (1 LLM call) ──
            plan = await classify_and_plan(request.query, budget)
            query_type = plan.get("query_type", "informational")
            ranking_metric = plan.get("ranking_metric")
            sub_queries = plan.get("sub_queries", [])
            schema = plan.get("schema", [])[:5]  # Cap at 5 columns

            yield sse_event("status", {
                "step": "classify",
                "query_type": query_type,
                "ranking_metric": ranking_metric,
                "sub_queries": sub_queries,
            })

            yield sse_event("schema", {"schema_fields": schema})

            # ── Step 3: Search + scrape ──────────────────────────────────
            max_pages = 6 if query_type == "ranked" else 5
            pages, scraped_count = await search_and_scrape(sub_queries, max_pages)

            yield sse_event("status", {
                "step": "search_done",
                "pages_found": len(pages),
                "scraped_count": scraped_count,
            })

            if not pages:
                yield sse_event("complete", {
                    "query": request.query,
                    "query_type": query_type,
                    "schema_fields": schema,
                    "entities": [],
                    "metadata": {
                        "pipeline_duration_seconds": round(time.time() - start_time, 2),
                        "count": 0, "sources_found": 0, "sources_scraped": 0,
                        "llm_calls": budget.calls,
                    },
                })
                return

            # ── Step 4: Branch by query type ─────────────────────────────

            if query_type == "ranked" and ranking_metric:
                # --- RANKED PIPELINE ---
                # 4a: Lightweight extraction — just names + metric
                ranked = await extract_ranked_names(pages, request.query, ranking_metric, budget)
                logger.info(f"Ranked extraction found {len(ranked)} unique entities")

                if not ranked:
                    # Fallback to full extraction if ranked extraction found nothing
                    logger.info("Ranked extraction empty, falling back to full extraction")
                    query_type = "informational"
                else:
                    # Convert to entity format and send initial results
                    entities = _ranked_to_entities(ranked, schema, ranking_metric)

                    # Send full schema (not pruned) so all column headers render;
                    # backfill will fill in the empty columns
                    yield sse_event("entities", {
                        "entities": entities,
                        "schema_fields": schema,
                        "partial": True,
                    })

                    # 4b: Backfill remaining fields (capped)
                    backfill_tasks = _collect_backfill_tasks(entities, schema, MAX_BACKFILLS_PER_REQUEST)
                    logger.info(f"  [BACKFILL] {len(backfill_tasks)} tasks queued (cap: {MAX_BACKFILLS_PER_REQUEST})")
                    for task in backfill_tasks:
                        try:
                            result = await _do_backfill(
                                request.query, task["entity_name"], task["sf"]["name"], task["sf"]["display_name"]
                            )
                            if result.get("value"):
                                task["ent"]["fields"][task["sf"]["name"]] = result
                                yield sse_event("cell_update", {
                                    "entity_index": task["ei"],
                                    "field_name": task["sf"]["name"],
                                    "field_data": result,
                                })
                        except Exception as e:
                            logger.warning(f"  [BACKFILL] {task['entity_name']}.{task['sf']['name']}: {e}")

                    active_schema = _prune_schema(schema, entities)
                    duration = round(time.time() - start_time, 2)

                    yield sse_event("complete", {
                        "query": request.query,
                        "query_type": "ranked",
                        "schema_fields": active_schema,
                        "entities": entities,
                        "metadata": {
                            "pipeline_duration_seconds": duration,
                            "count": len(entities),
                            "sources_found": len(pages),
                            "sources_scraped": scraped_count,
                            "llm_calls": budget.calls,
                        },
                    })
                    return

            # --- INFORMATIONAL / NAVIGATIONAL PIPELINE ---
            max_extract_pages = 4 if query_type == "informational" else 3
            extract_pages = pages[:max_extract_pages]

            all_entities = await extract_full(extract_pages, request.query, schema, budget)
            logger.info(f"Extracted: {len(all_entities)} raw entities (LLM calls: {budget.calls}/{budget.limit})")

            # If nothing found, retry with remaining pages (one more attempt)
            if not all_entities and len(pages) > max_extract_pages and not budget.exhausted():
                retry_pages = pages[max_extract_pages:max_extract_pages + 2]
                logger.info(f"  [RETRY] 0 entities found, trying {len(retry_pages)} more pages")
                all_entities = await extract_full(retry_pages, request.query, schema, budget)
                logger.info(f"  [RETRY] Got {len(all_entities)} entities")

            # Deduplicate
            final_entities = await deduplicate_entities(
                all_entities, schema, llm_client, llm_model, args.llm, budget
            )
            final_entities = final_entities[:5]

            # Stream results with full schema so all headers render
            yield sse_event("entities", {
                "entities": final_entities,
                "schema_fields": schema,
                "partial": True,
            })

            # Backfill missing cells (capped to avoid excessive calls)
            backfill_tasks = _collect_backfill_tasks(final_entities, schema, MAX_BACKFILLS_PER_REQUEST)
            logger.info(f"  [BACKFILL] {len(backfill_tasks)} tasks queued (cap: {MAX_BACKFILLS_PER_REQUEST})")
            for task in backfill_tasks:
                try:
                    result = await _do_backfill(
                        request.query, task["entity_name"], task["sf"]["name"], task["sf"]["display_name"]
                    )
                    if result.get("value"):
                        task["ent"]["fields"][task["sf"]["name"]] = result
                        yield sse_event("cell_update", {
                            "entity_index": task["ei"],
                            "field_name": task["sf"]["name"],
                            "field_data": result,
                        })
                except Exception as e:
                    logger.warning(f"  [BACKFILL] {task['entity_name']}.{task['sf']['name']}: {e}")

            active_schema = _prune_schema(schema, final_entities)
            duration = round(time.time() - start_time, 2)

            yield sse_event("complete", {
                "query": request.query,
                "query_type": query_type,
                "schema_fields": active_schema,
                "entities": final_entities,
                "metadata": {
                    "pipeline_duration_seconds": duration,
                    "count": len(final_entities),
                    "raw_entity_count": len(all_entities),
                    "sources_found": len(pages),
                    "sources_scraped": scraped_count,
                    "llm_calls": budget.calls,
                },
            })

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"FATAL PIPELINE ERROR: {e}", exc_info=True)
            yield sse_event("error", {"detail": str(e)})

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ── Internal Backfill (no rate limiting, uses cache) ─────────────────────────

async def _do_backfill(query: str, entity_name: str, field_name: str, field_display_name: str) -> Dict:
    """Internal backfill for a single cell — used by the pipeline, not the endpoint."""
    cache_key = hashlib.md5(f"backfill:{entity_name}:{field_name}".encode()).hexdigest()
    if cached := cache.get(cache_key):
        return cached

    search_query = f"{entity_name} {field_display_name}"
    logger.info(f"  [BACKFILL] {search_query}")

    results = await do_search(search_query, args.search, cache, limit=2)
    if not results:
        return {"value": None}

    texts = []
    for r in results[:1]:
        snippet = r.get("snippet", "")
        if len(snippet) >= 200:
            texts.append(snippet[:3000])
        else:
            content = await scrape_url(r["url"], cache)
            if content:
                texts.append(content[:3000])

    if not texts:
        texts = [r.get("snippet", "") for r in results if r.get("snippet")]
    if not texts:
        return {"value": None}

    combined = "\n\n".join(texts)
    prompt = (
        f'Extract ONLY the "{field_display_name}" for "{entity_name}".\n'
        f"Context query: {query}\n\n"
        f'Return JSON: {{"value": "the value", "evidence": "supporting text snippet", "source_url": "url"}}\n'
        f'If not found, return {{"value": null}}'
    )

    raw = _llm(
        messages=[
            {"role": "system", "content": f"{ROLE_BACKFILL_AGENT}\n\n{BACKFILL_SYSTEM}"},
            {"role": "user", "content": f"{prompt}\n\nText:\n{combined[:4000]}"},
        ],
        max_tokens=MAX_BACKFILL_TOKENS,
    )
    result = parse_json(raw)
    if result.get("value"):
        source_url = results[0].get("url", "")
        response = {
            "value": result["value"],
            "evidence": result.get("evidence", ""),
            "sources": [{"url": source_url}],
            "source_url": source_url,
        }
        cache.set(cache_key, response, expire=86400)
        logger.info(f"  [BACKFILL] Found: {entity_name}.{field_name} = {result['value']}")
        return response
    return {"value": None}


# ── Public Backfill Endpoint ─────────────────────────────────────────────────

@app.post("/api/backfill")
async def backfill_cell(request: BackfillRequest, req: Request):
    """Public backfill endpoint with rate limiting."""
    client_ip = req.client.host if req.client else "unknown"
    check_rate_limit(client_ip)
    return await _do_backfill(request.query, request.entity_name, request.field_name, request.field_display_name)


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok"}


@app.get("/api/config")
async def get_config():
    """Return the current server configuration."""
    return {
        "llm": args.llm,
        "llm_model": llm_model,
        "search": args.search,
        "rate_limit_per_min": RATE_LIMIT,
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=args.port)
