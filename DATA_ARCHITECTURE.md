# Data Architecture — YipitData Tech-News ARR Pipeline

## 1. Overview

We model the exercise as a small, local **medallion (bronze / silver / gold)** batch
pipeline. Messy, synthetic news data is cleaned and enriched, then materialized into
dimensions and a fact table that make **company ARR observations over time** the primary
queryable grain, while preserving full lineage back to source records.

Two source files enter the pipeline:

- `tech_news.csv` (750 articles) — the event/natural source of ARR observations.
- `company_metadata.json` (21 companies) — enrichment/master reference.

## 2. Data model (gold)

```
dim_company (one row per canonical company)
  company_id            <- surrogate, deterministic hash of canonical name
  company_name, industry, founded_year, headquarters,
  employee_count, is_public, stock_ticker

dim_article (one row per article; includes invalid & unmatched records)
  article_id            <- surrogate, deterministic hash of source article id
  source_article_id, title, company_name_raw, company_name_resolved,
  company_id (FK -> dim_company, nullable),
  published_date, published_year, published_quarter, published_month,
  category_raw, category_standardized,
  revenue_raw, revenue_currency, revenue_is_range,
  arr_usd, arr_parse_status,
  summary, url, author, word_count,
  + enriched: industry, founded_year, headquarters, employee_count,
    is_public, stock_ticker, company_age, company_size_category

fact_arr_observation (one row per VALID ARR observation)
  arr_observation_id    <- surrogate, deterministic hash of (company, article, date)
  company_id (FK), article_id (FK -> source lineage),
  observation_date, arr_usd, arr_raw_value, arr_currency, arr_is_range
```

### Grain and semantics

- **`fact_arr_observation.grain`**: one row per **valid** ARR observation, identified by
  `(company_id, article_id, observation_date)`. This is the grain that powers
  "ARR for a company over time".
- **ARR interpretation**: `arr_usd` is the normalized, FX-converted (integer USD)
  value of the article's `revenue`. The original string is kept in `arr_raw_value`
  and the original currency in `arr_currency` for audit, so we never lose the source.
- **Invalid / missing ARR** are **not** modeled as observations. They remain in
  `dim_article` with `arr_usd = NULL` and are surfaced in `parse_errors.csv`. This
  directly satisfies *"Do not silently treat missing or undisclosed values as valid ARR."*
- **Idempotency**: all surrogate keys are deterministic SHA-1 hashes of natural keys.
  The pipeline performs a full rebuild from raw on each run, so re-runs cannot create
  duplicate modeled records.

### Deriving common views

- **Latest ARR per company**
  ```sql
  WITH ranked AS (
    SELECT company_id, observation_date, arr_usd,
           ROW_NUMBER() OVER (PARTITION BY company_id ORDER BY observation_date DESC) rn
    FROM fact_arr_observation)
  SELECT * FROM ranked WHERE rn = 1
  ```
- **Quarterly ARR per company**: partition by `(company_id, published_year, published_quarter)`
  and take the observation closest to quarter end (last in the quarter).

### Lineage

`fact_arr_observation.article_id → dim_article.article_id` recovers the full source
article, whose `revenue_raw` and `source_article_id` connect back to the raw CSV row in
`bronze_articles.csv` (which stores the source file hash and ingest timestamp).

## 3. Cleaning & enrichment rules

### Revenue → ARR (see also `tests/test_revenue.py`)

Supports raw integers, `M`/`B`/`T` suffixes, word scales (`million`/`billion`), bare
numbers with a currency code (`500M USD`), `$`/`€`/`£`/`¥`, and ranges
(`$10M - $20M` → **midpoint**). FX: **EUR ×1.1, GBP ×1.27, JPY /150** (rates in
`src/config.py`). The parser returns an integer `arr_usd`, the original currency, and a
`parse_status` (`valid`, `valid_range`, `invalid_missing`, `unparseable`).

Missing tokens (`N/A`, `Not disclosed`, `null`, empty, `-`, …) → `arr_usd = NULL`.

### Date normalization (see `tests/test_dates.py`)

Handles ISO (`2022-02-17`), ISO+timezone (`2021-09-11T00:00:00Z`), US slash
(`02/23/2023`), dash (`23-08-2023`, `06-21-2024`), full month (`October 19, 2022`) and
day-abbreviated-month (`21 Feb 2020`). Extracts `year` / `quarter` / `month`.

**Ambiguity rule** (documented): for both `/` and `-` numeric dates we disambiguate by
which token is a valid month (1–12). If the leading token is a valid month we read it as
`MM-DD-YYYY`; otherwise, if the second token is a valid month, we read `DD-MM-YYYY`.
This matches the source data (e.g. `06-21-2024` = June 21 US; `23-08-2023` = Aug 23 EU).
All 750 source dates parse; any residual invalid date is flagged in `parse_errors.csv`.

### Category standardization (see `tests/test_categories.py`)

19 raw categories map to a 6-group taxonomy in `src/config.py` (`CATEGORY_MAP`):
`AI_ML`, `CLOUD_COMPUTING`, `FINTECH`, `SOFTWARE`, `CYBERSECURITY`, `DATA_ANALYTICS`.
Unknown values become `UNKNOWN` (never silently dropped).

### Company identity (see `tests/test_companies.py`)

- **Exact** canonical match; then **case-insensitive**; then **fuzzy** (`rapidfuzz`,
  token-sort ratio ≥ 88) with an explicit `ALIAS_MAP` for deterministic variations
  (`AWS`/`Amazon Web Services (AWS)` → Amazon Web Services; `Azure`/`Microsoft Azure` →
  Microsoft; `Nvidia`/`NVIDIA Corporation` → NVIDIA; `Open AI`/`OpenAI Inc.` → OpenAI;
  `Data Robot` → DataRobot; `Facebook AI Research` → Meta AI; and so on).
- Companies with **no metadata entry** (`Cohere`, `xAI`, `Mistral AI`, `Perplexity AI`,
  `Hugging Face` — 25 articles) are **resolved to `None`** and exported in
  `unmatched_companies.csv`, rather than being dropped or over-matched.

### Metadata enrichment

- `company_age = published_year - founded_year` (may be negative given synthetic
  metadata; kept as-is and visible for audit).
- `company_size_category` from `employee_count`: **Small** < 10k, **Medium** 10k–30k,
  **Large** > 30k.
- Industry is carried through for AI filtering.

## 4. AI-articles export (`ai_articles_enriched.csv`)

Includes every article where:
- **category is AI/ML** (`category_standardized == 'AI_ML'`) **OR** **company industry is
  AI/ML**, **and**
- `published_year ∈ [2022, 2024]`, **and**
- `arr_usd > 50,000,000`.

Columns are exactly those in the brief, plus `embedding` (JSON list) and (bonus)
`top_similar_articles` (top-3 most similar article ids, excluding self).

## 5. Semantic search & hybrid querying

`src/embeddings.py` embeds **title + summary** with `sentence-transformers`
`all-MiniLM-L6-v2` (384-d), normalizes vectors, and:

- `find_similar_articles(query_text, top_k=5)` → cosine search over all articles.
- `top_similar_for_each(...)` → top-3 nearest neighbours excluding self.

`src/duckdb_store.py` persists the modeled tables **and** embeddings (as `DOUBLE[]`) to
`yipitdata.duckdb` and exposes `hybrid_search(query, ..., categories=, years=, arr_min=)`.
Hybrid ranking applies the SQL filters first, then ranks the filtered rows by cosine
similarity to a query embedding.

Reusable artifacts: `embeddings/embeddings.npy` + `embeddings/embedding_ids.json`
(rows aligned to `silver_articles`).

## 6. Reliability beyond a local batch

The design already anticipates incremental, frequent arrivals:

- **Append/Rebuild**: today a full rebuild per run is cheap (750 rows) and guarantees
  idempotency. For production we would switch to **incremental ingestion**: track a
  `source_hash` / `max(source_article_id)` watermark per batch in `bronze`, then upsert
  deterministic keys into silver/gold (`INSERT ... ON CONFLICT DO UPDATE`).
- **Schema changes / backfills**: because lineage is retained in bronze and every
  surrogate key is deterministic, a schema change is applied by re-running the cleaning
  code on bronze and rebuilding silver/gold — no manual reconciliation of keys. Old
  `parse_errors`/obsolete rows can be dropped by regenerating from bronze.
- **Deployment**: the cleaning functions are pure and unit-testable; the orchestrator is
  a single entry point, so it can be wrapped as a scheduled job (Airflow/Dagster) or a
  managed serverless function. DuckDB gives a portable analytical store; swapping to a
  warehouse would reuse the same table definitions.
- **Data quality**: every ambiguity (dates, currencies, unmatched companies, failed
  parses) is either resolved by an explicit, documented rule or exported for audit —
  nothing is silently discarded.
