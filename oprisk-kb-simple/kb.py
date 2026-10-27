"""Knowledge base: 3 levels, plain functions.

L0  level0()                      index.yaml + dataset list from the Ossie files. Always in the prompt.
L1  search_sections(query, ...)   BM25 over business-knowledge sections; returns only the top
                                  matches, with every $ref inside them expanded.
L2  load_tables(model, datasets)  Apache Ossie model filtered to the requested datasets
                                  (+ the relationships and metrics that use them).

$ref syntax (Level 1):  {$ref: l1/shared.yaml#rag_status}   path is relative to knowledge/,
`#a.b` picks a key inside the file. Refs are expanded recursively; cycles raise an error.
"""
from __future__ import annotations

import json
import math
import re
from collections import Counter
from functools import lru_cache
from pathlib import Path

import yaml

KB_DIR = Path(__file__).resolve().parent / "knowledge"
STOPWORDS = {"a", "an", "the", "is", "are", "of", "to", "in", "for", "and", "or", "what", "how",
             "when", "which", "do", "does", "we", "our", "be", "by", "on", "with", "it", "this"}


# ---------- files and $ref ----------
@lru_cache(maxsize=None)  # files are read once per process; restart to pick up edits
def read_yaml(rel: str) -> dict:
    path = (KB_DIR / rel).resolve()
    if KB_DIR not in path.parents:
        raise ValueError(f"path outside knowledge folder: {rel}")
    data = yaml.safe_load(path.read_text()) or {}
    return json.loads(json.dumps(data, default=str))  # JSON-safe: int keys -> str, dates -> str


def expand_refs(value, chain: tuple = ()):
    """Replace every {$ref: file#key} in `value` (at any depth) with what it points to."""
    if isinstance(value, list):
        return [expand_refs(v, chain) for v in value]
    if not isinstance(value, dict):
        return value
    if "$ref" in value:
        ref = value["$ref"]
        if ref in chain:
            raise ValueError(f"circular $ref: {' -> '.join(chain + (ref,))}")
        file, _, key_path = ref.partition("#")
        target = read_yaml(file)
        for key in filter(None, key_path.split(".")):
            target = target[key]
        return expand_refs(target, chain + (ref,))
    return {k: expand_refs(v, chain) for k, v in value.items()}


# ---------- L0 ----------
def level0() -> dict:
    index = read_yaml("index.yaml")
    domains = {}
    for name, d in index["domains"].items():
        model = read_yaml(d["tables"])
        domains[name] = {
            "summary": d["summary"],
            "ossie_model": model["name"],
            "datasets": {ds["name"]: ds.get("description", "") for ds in model["datasets"]},
        }
    return {"summary": index["summary"], "domains": domains}


# ---------- L1: BM25 over sections ----------
def _tokens(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in STOPWORDS]


def _text(value) -> str:
    """Flatten any YAML value (keys and values) into searchable text."""
    if isinstance(value, dict):
        return " ".join(f"{k.replace('_', ' ')} {_text(v)}" for k, v in value.items())
    if isinstance(value, list):
        return " ".join(_text(v) for v in value)
    return str(value)


@lru_cache(maxsize=None)
def _sections() -> list[dict]:
    """Every L1 section, refs expanded, with its tokens for BM25."""
    out = []
    for domain, d in read_yaml("index.yaml")["domains"].items():
        for sid, s in read_yaml(d["knowledge"])["sections"].items():
            content = expand_refs(s.get("content", {}))
            # title and keywords count twice: they say what the section is *about*
            about = f"{sid.replace('_', ' ')} {s.get('title', '')} {' '.join(s.get('keywords', []))}"
            out.append({
                "path": f"{domain}.{sid}",
                "domain": domain,
                "title": s.get("title", sid),
                "content": content,
                "tables": s.get("tables", []),
                "_tokens": _tokens(f"{about} {about} {_text(content)}"),
            })
    return out


def _bm25(query: list[str], docs: list[list[str]], k1: float = 1.5, b: float = 0.75) -> list[float]:
    n = len(docs)
    avg_len = sum(map(len, docs)) / n
    df = Counter(t for d in docs for t in set(d))
    scores = []
    for doc in docs:
        tf = Counter(doc)
        score = 0.0
        for t in query:
            if t in tf:
                idf = math.log(1 + (n - df[t] + 0.5) / (df[t] + 0.5))
                score += idf * tf[t] * (k1 + 1) / (tf[t] + k1 * (1 - b + b * len(doc) / avg_len))
        scores.append(score)
    return scores


def search_sections(query: str, domain: str = "", top_k: int = 3) -> list[dict]:
    sections = [s for s in _sections() if not domain or s["domain"] == domain]
    if not sections:
        return []
    scores = _bm25(_tokens(query), [s["_tokens"] for s in sections])
    ranked = sorted(zip(scores, sections), key=lambda x: x[0], reverse=True)
    return [
        {"path": s["path"], "title": s["title"], "score": round(score, 2),
         "content": s["content"], "tables": s["tables"]}
        for score, s in ranked[:top_k] if score > 0
    ]


# ---------- L2: Ossie tables ----------
def load_tables(model: str, datasets: list[str]) -> dict:
    files = {d["tables"] for d in read_yaml("index.yaml")["domains"].values()}
    models = {read_yaml(f)["name"]: read_yaml(f) for f in files}
    if model not in models:
        raise KeyError(f"unknown model '{model}'. Options: {sorted(models)}")
    m = models[model]
    by_name = {ds["name"]: ds for ds in m["datasets"]}
    unknown = [d for d in datasets if d not in by_name]
    if unknown or not datasets:
        raise KeyError(f"unknown datasets {unknown} in '{model}'. Options: {sorted(by_name)}")

    wanted = set(datasets)
    # relationships touching a requested dataset (shows the join path, even to other tables)
    rels = [r for r in m.get("relationships", []) if r["from"] in wanted or r["to"] in wanted]
    # metrics that use a requested dataset, plus metrics built from those metrics
    metrics, names = [], set()
    for metric in m.get("metrics", []):
        expr = json.dumps(metric.get("expression", {}))
        if any(f"{d}." in expr for d in wanted) or any(re.search(rf"\b{n}\b", expr) for n in names):
            metrics.append(metric)
            names.add(metric["name"])
    return {
        "model": model,
        "description": m.get("description", ""),
        "ai_context": m.get("ai_context", {}),
        "datasets": [by_name[d] for d in datasets],
        "relationships": rels,
        "metrics": metrics,
    }
