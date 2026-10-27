# Operational-risk knowledge agent (simple)

One ADK `LlmAgent` that loads knowledge in three levels, only as deep as the question needs.

| Level | What | Where | How it's loaded |
|---|---|---|---|
| L0 | Domains + dataset names | `knowledge/index.yaml` (+ dataset list read from L2) | Always in the prompt |
| L1 | Business knowledge: definitions, rules, thresholds | `knowledge/l1/*.yaml` sections | `search_knowledge` → BM25, top 3 sections, `$ref`s expanded |
| L2 | Tables: datasets, columns, joins, metrics | `knowledge/l2/*.yaml` (Apache Ossie) | `load_tables(model, datasets)` → only those datasets |

```
oprisk-kb-simple/
  knowledge/index.yaml      L0
  knowledge/l1/*.yaml       L1 sections (+ shared.yaml fragments)
  knowledge/l2/*.yaml       L2 Ossie semantic models
  kb.py                     all knowledge logic, plain functions
  oprisk_agent/agent.py     the agent + its 2 tools
  tests/                    no API key needed
```

## Run

```bash
pip install -r requirements.txt
export GOOGLE_API_KEY=...
adk web .          # pick "oprisk_agent"
pytest
```

Try: "Which tables exist for KRIs?" (L0 only) · "What is a near miss?" (L1) ·
"Is a collateral documentation error a credit or op loss?" (L1) ·
"Total net loss by event type this year — which columns?" (L1 + L2).

## Writing L1 sections

```yaml
sections:
  rag_and_breaches:                 # path becomes kri.rag_and_breaches
    title: RAG status and breach rules
    keywords: [rag, amber, red, breach]  # title + keywords weigh double in search
    content:                        # any YAML; returned as-is
      rag_status: {$ref: l1/shared.yaml#rag_status}   # expanded on load
    tables: [oprisk_kri.kri_values] # pointers to L2 (model.dataset), NOT auto-loaded
```

- `$ref: file#a.b` points into any file under `knowledge/`. Refs are expanded
  recursively; a cycle raises an error.
- `tables` is how a section links to Ossie. It's a pointer, so the agent decides
  whether to load the tables.

## L2 (Ossie)

Standard Ossie model files: `datasets` with `fields`, plus `relationships` and
`metrics`. `load_tables` returns only the requested datasets, the relationships
touching them (so the join path is visible), and the metrics that use them.
These sample files follow the published TPC-DS example; validate yours against
`core-spec/osi-schema.json` in github.com/apache/ossie.

## Notes

- Files are cached once per process (`lru_cache`), so restart after editing YAML.
- To expose this over MCP later, wrap `kb.search_sections` and `kb.load_tables` as
  two MCP tools. Nothing else depends on where the files live.
- Content is illustrative sample policy, not authoritative.
