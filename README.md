# Dynamic Agentic Search Agent

An agentic search tool that classifies user queries by intent, generates dynamic schemas, scrapes the web, extracts structured entities using LLMs, and streams results progressively via Server-Sent Events. The pipeline adapts its strategy based on whether the user wants ranked results, general information, or navigational/transactional data.

# Demo
https://github.com/user-attachments/assets/bf34f38a-90ce-43bc-a672-aef8051ef9fe

## Solution

### 1. Query Classification + Schema Generation (1 LLM Call)

The user submits a query via `POST /api/search`. In a **single LLM call**, the Classifier Agent determines:
- **Query type**: `ranked`, `informational`, or `navigational`
- **Ranking metric** (for ranked queries): e.g. "customer rating", "ROI percentage"
- **2 sub-queries**: diverse search strings for broad coverage
- **Table schema**: 4-5 column definitions (name, type, display name) tailored to the query topic

Combining classification and schema generation was done to reduce the number of additonal LLM calls

| Query Type | Example | Behavior |
|------------|---------|----------|
| **Ranked** | "Best pizza in Brooklyn", "Top ROI MS CS programs" | Extract entity names + metric, then backfill details |
| **Informational** | "AI healthcare startups", "Cafes in Amherst" | Batched full extraction, no ranking needed |
| **Navigational** | "Buy MacBook Pro", "Spotify pricing" | Batched extraction, always includes website URL |

### 2. Web Search + Scraping (1 Search API Call)

A single search query is sent to Brave or Firecrawl (instead of searching each sub-query separately). Results are:
- **Filtered** through a URL blocklist (reddit, facebook, youtube, forbes, linkedin, etc.)
- **Deduplicated** by URL and capped at 5-6 unique pages
- **Scraped** with Firecrawl markdown preferred over re-fetching with httpx
- Pages with less than 100 characters of content are discarded as unusable

Firecrawl requests extra results (`limit + 3`) to compensate for blocked domains being filtered out.

### 3. Batched Extraction (1-2 LLM Calls)

**Ranked pipeline:**
- Combines top 2-3 pages into a **single LLM call** that extracts entity names + ranking metric values
- Streams the initial table immediately with all column headers visible
- **Backfills** up to 3 remaining fields via targeted single-field searches

**Informational/navigational pipeline:**
- Concatenates multiple pages into batches (~8K chars each) for **1-2 LLM calls** instead of 1 per page
- Deduplication via name normalization + optional LLM-assisted merge
- Streams results, then backfills up to 3 missing cells
- If 0 entities found on the first pass, retries with 2 additional pages

### 4. Progressive Streaming (SSE)

Results stream to the frontend as Server-Sent Events:

| Event | When | What it contains |
|-------|------|-----------------|
| `status` | After classify, after search | Query type, metric, progress |
| `schema` | After classification | Column definitions (stored, not rendered yet) |
| `entities` | After extraction | Initial entity data with full schema headers |
| `cell_update` | During backfill | Individual cell values — cells fill in one by one |
| `complete` | Pipeline done | Final entities, pruned schema, metadata |

The frontend **waits until at least one entity has 2+ filled columns** before showing the table — avoiding the jarring experience of empty columns appearing and disappearing. A 15-second timeout stops the loading shimmer on cells that couldn't be backfilled.

### 5. API Call Budget

Each `/api/search` request targets **5-6 total API calls**:

| Step | API Calls |
|------|-----------|
| Classify + Schema | 1 LLM call |
| Web Search | 1 search call |
| Extraction | 1-2 LLM calls |
| Backfill | up to 3 × (1 search + 1 LLM) |
| **Total** | **~5-6** |

LLM calls within the pipeline are capped at 6 (`MAX_LLM_CALLS_PER_REQUEST`). Backfill is capped at 3 tasks (`MAX_BACKFILLS_PER_REQUEST`) and skips entities that are already mostly filled (≤1 missing field). Backfill also skips fields redundant with the entity name.

## Agent Roles

| Agent | Purpose | Budget Cost |
|-------|---------|-------------|
| **Classifier + Schema Designer** | Classifies intent, generates sub-queries, and designs table schema | 1 call |
| **Ranked Extractor** | Lightweight: extracts entity names + metric from batched pages | 1 call |
| **Data Extractor** | Batched full extraction: all schema fields from combined page text | 1-2 calls |
| **Dedup Agent** | Groups records referring to the same entity | 1 call per ambiguous group |
| **Backfill Agent** | Extracts a single missing field for a named entity | Depends on how we configure (0-3 calls) |

## How to Run

### Prerequisites
- Python 3.12+
- An LLM backend: The code is compatible with either [Ollama](https://ollama.ai) running locally, or an [OpenRouter](https://openrouter.ai) API key, but OpenRouter is recommended.
- A search backend: a [Brave Search API](https://brave.com/search/api/) key, or a [Firecrawl](https://firecrawl.dev) API key

### Setup

```bash
cd ag-search

python -m venv agvenv
source agvenv/bin/activate

pip install -r requirements.txt
```

### Environment Variables

Create a `.env` file in the project root:

```env

# Required for using FireCrawl (default search backend)
FIRECRAWL_API_KEY=your_firecrawl_key

# Required for Brave Search (optional if you don't want to use Firecrawl)
BRAVE_API_KEY=your_brave_api_key

# Required if using --llm openrouter
OPENROUTER_API_KEY=your_openrouter_key
OPENROUTER_MODEL=openai/gpt-4o-mini    # optional, this is the default

# Optional
OLLAMA_MODEL=llama3.2                   # default Ollama model
OLLAMA_URL=http://localhost:11434/v1    # default Ollama URL
RATE_LIMIT_PER_MIN=15                   # default rate limit
```

### Running

```bash
# OpenRouter + FireCrawl Search on port 8000 (you can change values as needed)
python main.py --llm openrouter --search firecrawl

# To clear local cache
python -c "from diskcache import Cache; Cache('agent_cache').clear()"
```

### Frontend

Open `frontend/index.html` directly in a browser. It connects to the backend API at `http://localhost:8000` by default.

### Example Request

```bash
# SSE streaming response
curl -N -X POST http://localhost:8000/api/search \
  -H "Content-Type: application/json" \
  -d '{"query": "best pizza places in New York City"}'
```

## Design Tradeoffs

### Combined Classify + Schema (1 call) vs. Separate (2 calls)
- **Choice**: Merge classification, sub-query generation, and schema design into a single LLM call.
- **Pro**: Saves one LLM round-trip (~1-2s). Schema is informed by query type from the start.
- **Con**: Slightly more complex prompt. If the LLM fails, both classification and schema must be retried.

### Single Search Call vs. Multiple Sub-Queries
- **Choice**: Send one search query instead of 3 separate sub-queries.
- **Pro**: Uses 1 search API call instead of 3 — major cost saving with Firecrawl.
- **Con**: Less diversity in search results. Mitigated by requesting more results per call.

### Batched Extraction vs. Per-Page Extraction
- **Choice**: Concatenate multiple pages into ~8K char batches for fewer LLM calls.
- **Pro**: 1-2 extraction calls instead of 3-5. Significant cost and latency reduction.
- **Con**: LLM sees more context per call, which can reduce extraction precision for individual pages.

### Delayed Table Reveal
- **Choice**: Don't show the table until at least one entity has ≥2 filled columns.
- **Pro**: Avoids jarring UX of empty columns appearing and disappearing as data streams in.
- **Con**: Slightly longer perceived wait before seeing any results.

### Capped Backfill (3 per request)
- **Choice**: Limit backfill to 3 tasks, skip mostly-filled entities (≤1 missing field).
- **Pro**: Prevents runaway API costs. A ranked query with 5 entities no longer generates 20+ backfill calls.
- **Con**: Some cells may remain empty. The 15-second frontend timeout removes loading indicators gracefully.

### URL Blocklist
- **Choice**: Hard-block domains known to fail scraping (reddit, youtube, forbes, linkedin, etc.).
- **Pro**: Eliminates wasted scrape slots and LLM calls on garbage content.
- **Con**: May miss useful content. Firecrawl compensates by fetching extra results.

### Dynamic Schema vs. Fixed Schema
- **Choice**: The schema is generated per-query by the LLM.
- **Pro**: Works for any topic without code changes.
- **Con**: Different queries about the same topic may produce inconsistent schemas.

### diskcache for Caching
- **Choice**: File-based cache rather than Redis.
- **Pro**: No external dependencies. Survives server restarts.
- **Con**: Not shareable across machines.

## Production Challenges

### Reliability
- **Scraping**: Many sites serve JS-rendered content that `httpx` cannot execute. Firecrawl helps but adds cost.

### Scalability
- **Single-process**: The app runs as one uvicorn process. 
- **In-memory rate limiter**: Does not work behind a load balancer.
- **Synchronous LLM calls**: `llm_complete` blocks the event loop during inference.

### Cost
- **Backfill calls**: Capped at 3 per request but still outside the LLM budget. Cached aggressively.
- **Firecrawl costs**: Each search call uses Firecrawl credits. Extra results requested to offset blocked domains.

### Security
- **CORS wildcard**: Allows any origin which is risky in production.
- **No authentication**: API is open to anyone (biggest challenge)
- **SSRF risk**: The scraper fetches arbitrary URLs from search results.

## Future Improvements

- **Async LLM client**: Switch to `AsyncOpenAI` to avoid blocking the event loop.
- **Headless browser fallback**: Use Playwright for JS-heavy pages.
- **Distributed caching**: Replace `diskcache` with Redis for multi-instance deployments.
- **Authentication & per-user quotas**: Add API key auth and usage tracking.
- **Schema consistency**: Cache schemas per topic category for comparable results.
- **Confidence scoring**: Have the LLM assign confidence to extracted values.
- **Observability**: Add OpenTelemetry tracing and token usage metrics.


## Project Structure

```
ag-search/
├── backend/
│   ├── __init__.py
│   ├── main.py                 # FastAPI app, SSE streaming, pipeline orchestration
│   ├── prompts.py              # All LLM system prompts and agent role definitions
│   └── utils/
│       ├── __init__.py
│       ├── llm.py              # LLM client setup, completion, JSON parsing
│       ├── scraper.py          # HTML fetching and text extraction
│       ├── search.py           # Web search providers (Brave, Firecrawl) + URL blocklist
│       └── dedup.py            # Entity deduplication and field merging
├── frontend/
│   └── index.html              # Single-page UI with SSE streaming (Tailwind CSS)
├── .env                        # API keys and config
├── .gitignore
├── requirements.txt            # Python dependencies
└── README.md
```
