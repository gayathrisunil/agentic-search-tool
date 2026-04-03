"""Entity deduplication and field merging logic."""

import json
import logging
import re
from typing import List, Dict, Optional

from utils.llm import llm_complete, parse_json, MAX_MERGE_TOKENS
from prompts import ROLE_DEDUP_AGENT, DEDUP_SYSTEM

logger = logging.getLogger("ResearchAgent")


def _get_primary_key(entity: Dict, schema: List[Dict]) -> Optional[str]:
    """Return the value of the first schema field, used as the entity identifier."""
    fields = entity.get("fields", {})
    for sf in schema:
        name = sf["name"]
        if name in fields and isinstance(fields[name], dict):
            val = fields[name].get("value")
            if val:
                return str(val).strip()
        dn = sf.get("display_name", "")
        if dn in fields and isinstance(fields[dn], dict):
            val = fields[dn].get("value")
            if val:
                return str(val).strip()
    return None


def _normalize_name(name: str) -> str:
    """Lowercase and strip common suffixes/punctuation for fuzzy name comparison."""
    name = name.lower().strip()
    name = re.sub(r"['\u2019`]s$", "", name)
    name = re.sub(r"\b(inc|llc|corp|co|ltd|the|restaurant|pizza|cafe|bar)\.?\b", "", name)
    name = re.sub(r"[^a-z0-9\s]", "", name)
    return re.sub(r"\s+", " ", name).strip()


def _merge_fields(a: Dict, b: Dict) -> Dict:
    """Merge two entity field dicts, preferring non-null values and longer evidence."""
    merged = {}
    all_keys = set(list(a.keys()) + list(b.keys()))
    for key in all_keys:
        fa = a.get(key, {})
        fb = b.get(key, {})
        if not isinstance(fa, dict):
            fa = {"value": fa}
        if not isinstance(fb, dict):
            fb = {"value": fb}

        va = fa.get("value")
        vb = fb.get("value")

        if va and not vb:
            merged[key] = fa
        elif vb and not va:
            merged[key] = fb
        elif va and vb:
            ea = len(fa.get("evidence", "") or "")
            eb = len(fb.get("evidence", "") or "")
            chosen = fa if ea >= eb else fb
            sources_a = fa.get("sources", [])
            sources_b = fb.get("sources", [])
            seen_urls = set()
            combined = []
            for s in sources_a + sources_b:
                u = s.get("url", "")
                if u not in seen_urls:
                    seen_urls.add(u)
                    combined.append(s)
            chosen["sources"] = combined
            merged[key] = chosen
        else:
            merged[key] = fa or fb
    return merged


async def deduplicate_entities(
    entities: List[Dict],
    schema: List[Dict],
    client,
    model: str,
    llm_backend: str,
    budget=None,
) -> List[Dict]:
    """Group entities by normalized name, merge duplicates, and sort by completeness."""
    if not entities:
        return []

    groups: Dict[str, List[Dict]] = {}
    ungrouped = []

    for ent in entities:
        pk = _get_primary_key(ent, schema)
        if not pk:
            ungrouped.append(ent)
            continue
        norm = _normalize_name(pk)
        if not norm:
            ungrouped.append(ent)
            continue

        matched = False
        for gkey in list(groups.keys()):
            if norm in gkey or gkey in norm:
                groups[gkey].append(ent)
                matched = True
                break
            tokens_a = set(norm.split())
            tokens_b = set(gkey.split())
            if tokens_a and tokens_b:
                overlap = len(tokens_a & tokens_b) / max(len(tokens_a), len(tokens_b))
                if overlap > 0.6:
                    groups[gkey].append(ent)
                    matched = True
                    break
        if not matched:
            groups[norm] = [ent]

    merged = []
    ambiguous_groups = []

    for gkey, group in groups.items():
        if len(group) == 1:
            merged.append(group[0])
        elif len(group) == 2:
            combined_fields = _merge_fields(
                group[0].get("fields", {}), group[1].get("fields", {})
            )
            merged.append({"fields": combined_fields, "source_url": group[0].get("source_url", "")})
        else:
            ambiguous_groups.append(group)

    for group in ambiguous_groups:
        try:
            # Check budget before making LLM call; fall through to simple merge if exhausted
            if budget is not None and budget.calls >= budget.limit:
                raise RuntimeError("budget exhausted")

            group_summary = json.dumps(
                [
                    {
                        "index": i,
                        "fields": {
                            k: v.get("value") if isinstance(v, dict) else v
                            for k, v in ent.get("fields", {}).items()
                        },
                    }
                    for i, ent in enumerate(group)
                ],
                indent=2,
            )

            if budget is not None:
                raw = budget.call(
                    messages=[
                        {"role": "system", "content": f"{ROLE_DEDUP_AGENT}\n\n{DEDUP_SYSTEM}"},
                        {"role": "user", "content": f"Records to deduplicate:\n{group_summary}"},
                    ],
                    json_mode=True,
                    max_tokens=MAX_MERGE_TOKENS,
                )
            else:
                raw = llm_complete(
                    client, model, llm_backend,
                    messages=[
                        {"role": "system", "content": f"{ROLE_DEDUP_AGENT}\n\n{DEDUP_SYSTEM}"},
                        {"role": "user", "content": f"Records to deduplicate:\n{group_summary}"},
                    ],
                    json_mode=True,
                    max_tokens=MAX_MERGE_TOKENS,
                )
            merge_plan = parse_json(raw).get("groups", [])

            for idx_group in merge_plan:
                if not idx_group:
                    continue
                base = group[idx_group[0]]
                for idx in idx_group[1:]:
                    if idx < len(group):
                        base["fields"] = _merge_fields(
                            base.get("fields", {}), group[idx].get("fields", {})
                        )
                merged.append(base)
        except Exception as e:
            logger.warning(f"  [MERGE LLM ERROR] {e} — falling back to simple merge")
            base = group[0]
            for other in group[1:]:
                base["fields"] = _merge_fields(base.get("fields", {}), other.get("fields", {}))
            merged.append(base)

    merged.extend(ungrouped)

    def completeness(ent):
        fields = ent.get("fields", {})
        non_null = sum(
            1 for v in fields.values() if isinstance(v, dict) and v.get("value") is not None
        )
        return -non_null

    merged.sort(key=completeness)

    logger.info(f"  [DEDUP] {len(entities)} raw -> {len(merged)} after merge")
    return merged
