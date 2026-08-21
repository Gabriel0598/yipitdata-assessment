# AGENTS.md

Guidance for AI coding agents (opencode) working in this repository. This
project is a submission for a data-engineering take-home exercise; keep changes
consistent with the existing architecture and conventions below.

## What this is

A local Python medallion (bronze → silver → gold) pipeline that turns messy
technology-news articles into (1) a queryable company ARR-observation model and
(2) a vector/hybrid search index over articles.

## Commands

```bash
# Run the full pipeline (embeddings + DuckDB)
.venv/bin/python run_pipeline.py

# Skip slow embedding step
.venv/bin/python run_pipeline.py --no-embeddings

# Run tests
.venv/bin/python -m pytest tests/ -q

# Inspect the persisted DuckDB
.venv/bin/python -m src.duckdb_store
.venv/bin/python -i query_duckdb.py
```

Python used is **3.11** (`.venv`). Do not switch the interpreter or upgrade
major package versions without a strong reason.

## Repository conventions / guardrails

- **Medallion layout is sacred.** Bronze holds raw sources plus a file hash and
  ingest timestamp. Silver is one row per cleaned/enriched article. Gold has
  exactly three tables: `dim_company`, `dim_article`, `fact_arr_observation`.
  Do not collapse or rename these without updating docs and the DuckDB DDL.
- **Idempotency is a hard requirement.** All surrogate keys are deterministic
  SHA-1 hashes of natural keys (`COMP-…`, `ART-…`, `ARR-…`). Any new key must
  stay deterministic so re-runs never duplicate records. There is a test for
  this — keep it green.
- **Never lose lineage.** Keep `revenue_raw` / `arr_raw_value`, the original
  currency, the source `article_id`, and the raw company name. Missing or
  undisclosed revenue must never be modeled as a valid ARR observation.
- **Configuration lives in `src/config.py`.** FX rates, size thresholds, the
  category taxonomy, and the AI-export filters all belong there — not hard-coded
  in cleaning functions.
- **Cleaning functions are pure and reusable** under `src/cleaning/`
  (`revenue.py`, `dates.py`, `categories.py`, `companies.py`). Keep them free of
  I/O and side effects so they stay unit-testable.
- **Every modeled table is exported to its own CSV** in `data/output/`
  (required), plus an optional persisted DuckDB file. `ai_articles_enriched.csv`
  is a required downstream export that does NOT replace the modeled-table CSVs.
- **Embeddings**: `all-MiniLM-L6-v2` over title + summary, L2-normalized,
  persisted as `embeddings.npy` + `embedding_ids.json` (rows aligned to
  `silver_articles`). When adding embeddings, keep the article-id alignment
  explicit — do not zip embeddings against a reordered filtered subset.
- **Tests** live in `tests/` and must pass before finishing (`pytest tests/ -q`).
  New cleaning rules should come with golden-case tests. Embedding tests use
  synthetic vectors so the suite runs without downloading the model.
- **Docs**: keep `README.md` and `DATA_ARCHITECTURE.md` in sync with any
  behavior change. The architecture doc is the source of truth for the model,
  cleaning rules, and reliability/backfill strategy.
- **Do not commit** `.venv/`, `__pycache__/`, or `.pytest_cache/` (gitignored).
  `data/output/` IS tracked (deliverables are versioned), so be deliberate when
  changing those generated files.

## Scope / avoid

- Do not modify the raw inputs in `data/raw/`.
- Do not add secrets, tokens, or personal identifiers.
- Keep the pipeline runnable locally with a single entry point
  (`run_pipeline.py`).
