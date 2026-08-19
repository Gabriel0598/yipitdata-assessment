# YipitData — Data Engineering Assignment

A local Python data pipeline that turns messy technology-news articles into:

1. a **queryable company ARR observations model** (medallion tables exported as CSV), and
2. a **vector search index** to support semantic and hybrid search.

Every valid `revenue` value in the source is treated as a **reported ARR observation**
for the article's company as of the article's `published_date`; the cleaned/modeled
outputs make that ARR interpretation explicit while preserving source lineage.

---

## System requirements

- Python **3.11** (recommended — `sentence-transformers` / `torch` wheels are stable on 3.11)
- ~2.5 GB disk for dependencies (torch + sentence-transformers)
- Internet access on **first** run to download the `all-MiniLM-L6-v2` embedding model (~130 MB)

> The pipeline core (cleaning, modeling, CSV exports) does **not** require torch. Only
> the optional semantic-search layer (`--no-embeddings` disables it) needs it.

## Installation

```bash
cd ~/Documents/General-Projects-Source-Code/yipitdata_assessment
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If `python3.11` is not on your PATH, point `pyenv`/any 3.11+ interpreter at the venv command.

## Running the pipeline

```bash
# Full run: build model, generate embeddings, export all CSVs, persist DuckDB
python run_pipeline.py

# Skip the slow embedding step (model + exports still produced)
python run_pipeline.py --no-embeddings

# Also skip persisting the DuckDB database
python run_pipeline.py --no-duckdb
```

The raw sources are read from `data/raw/` (copied verbatim from the assignment).
All outputs are written to `data/output/`:

| File | Contents |
|------|----------|
| `bronze_articles.csv` / `bronze_companies.csv` | raw source + file hash + ingest timestamp (provenance) |
| `silver_articles.csv` | one row per cleaned/enriched article |
| `dim_company.csv` | company dimension (1 row per canonical company) |
| `dim_article.csv` | article dimension (all articles, incl. invalid/unmatched) |
| `fact_arr_observation.csv` | one row per **valid** ARR observation |
| `unmatched_companies.csv` | article companies with no metadata match |
| `parse_errors.csv` | every failed revenue/date parse for audit |
| `ai_articles_enriched.csv` | **required** downstream AI export |
| `yipitdata.duckdb` | persisted DuckDB (optional) with the same tables |
| `embeddings/embeddings.npy`, `embedding_ids.json` | reusable serialized embeddings |

### Regenerating all CSVs

Re-running `python run_pipeline.py` rebuilds every table from raw and overwrites the
output CSVs. Surrogate keys are deterministic hashes of natural keys, so repeated runs
**never create duplicate modeled records** (verified by an idempotency test).

## Example queries / usage

```python
# Latest ARR per company, from the fact table (window function)
-- DuckDB
WITH ranked AS (
  SELECT company_id, observation_date, arr_usd,
         ROW_NUMBER() OVER (PARTITION BY company_id ORDER BY observation_date DESC) rn
  FROM fact_arr_observation
) SELECT company_id, observation_date, arr_usd FROM ranked WHERE rn = 1
ORDER BY arr_usd DESC;

# Semantic search over all articles
from src.embeddings import find_similar_articles
find_similar_articles("a company raising a large funding round", top_k=5)

# Hybrid search: SQL filters + vector similarity
from src import duckdb_store
duckdb_store.hybrid_search(
    "machine learning model breakthrough",
    categories=["AI_ML"], published_year_from=2022,
    published_year_to=2024, arr_min_usd=50_000_000, top_k=3,
)

# Build the AI export programmatically
import src.pipeline as pipeline
from src.export import build_ai_articles_enriched
res = pipeline.run_pipeline()
ai = build_ai_articles_enriched(res.dim_article, None, None, None)
```

## Tests

```bash
python -m pytest tests/ -q        # 134 tests
```

Covers golden-case parsing (revenue formats, currency conversion, ranges, dates and
date-ambiguity rules), the category taxonomy, company alias/fuzzy resolution, pipeline
row counts, the fact grain, enrichment columns, the AI export filters and a full
**idempotency** rebuild check. The embedding tests use synthetic vectors so the suite
runs without downloading the model.

## Notes & assumptions

See [DATA_ARCHITECTURE.md](./DATA_ARCHITECTURE.md) for the data model, the ambiguity
rules (date interpretation, fuzzy matching), and how the pipeline would run reliably
beyond a local batch.
