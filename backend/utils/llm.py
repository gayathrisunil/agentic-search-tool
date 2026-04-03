"""LLM client setup and completion helpers."""

import json
import re
import logging
from typing import List, Dict, Any

from openai import OpenAI

logger = logging.getLogger("ResearchAgent")

# ── Token Budgets ────────────────────────────────────────────────────────────
MAX_SUB_QUERY_TOKENS = 300
MAX_SCHEMA_TOKENS = 500
MAX_EXTRACT_TOKENS = 3000
MAX_MERGE_TOKENS = 1000
MAX_BACKFILL_TOKENS = 300


def make_llm_client(args) -> tuple[OpenAI, str]:
    """Create an OpenAI-compatible client for the configured LLM backend."""
    if args.llm == "ollama":
        import os
        client = OpenAI(base_url=args.ollama_url, api_key="ollama")
        model = args.ollama_model
    else:
        import os
        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.getenv("OPENROUTER_API_KEY"),
        )
        model = os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini")
    return client, model


def llm_complete(
    client: OpenAI,
    model: str,
    llm_backend: str,
    messages: List[Dict],
    json_mode: bool = False,
    max_tokens: int = 1500,
) -> str:
    """Send a chat completion request and return the response text."""
    kwargs: Dict[str, Any] = dict(model=model, messages=messages, max_tokens=max_tokens)
    if json_mode:
        if llm_backend == "ollama":
            kwargs["format"] = "json"
        else:
            kwargs["response_format"] = {"type": "json_object"}
    res = client.chat.completions.create(**kwargs)
    return res.choices[0].message.content


def parse_json(raw: str) -> Any:
    """Strip markdown fences and parse JSON, attempting repair on truncated output."""
    raw = raw.strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        last_brace = raw.rfind("}")
        if last_brace > 0:
            repaired = raw[: last_brace + 1]
            open_brackets = repaired.count("[") - repaired.count("]")
            open_braces = repaired.count("{") - repaired.count("}")
            repaired += "]" * open_brackets + "}" * open_braces
            try:
                return json.loads(repaired)
            except json.JSONDecodeError:
                pass
        raise
