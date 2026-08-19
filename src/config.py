"""Central configuration for the YipitData pipeline.

All tunable constants live here so the rest of the code stays declarative
and easy to reason about (FX rates, thresholds, taxonomy, paths).
"""
from __future__ import annotations

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
OUTPUT_DIR = DATA_DIR / "output"

ARTICLES_RAW = os.environ.get("ARTICLES_RAW", str(RAW_DIR / "tech_news.csv"))
COMPANY_METADATA_RAW = os.environ.get(
    "COMPANY_METADATA_RAW", str(RAW_DIR / "company_metadata.json")
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Per-table modeled CSV exports (one per warehouse table).
BRONZE_ARTICLES_CSV = OUTPUT_DIR / "bronze_articles.csv"
BRONZE_COMPANIES_CSV = OUTPUT_DIR / "bronze_companies.csv"
SILVER_ARTICLES_CSV = OUTPUT_DIR / "silver_articles.csv"
SILVER_COMPANIES_CSV = OUTPUT_DIR / "silver_companies.csv"
DIM_COMPANY_CSV = OUTPUT_DIR / "dim_company.csv"
DIM_ARTICLE_CSV = OUTPUT_DIR / "dim_article.csv"
FACT_ARR_OBSERVATION_CSV = OUTPUT_DIR / "fact_arr_observation.csv"
UNMATCHED_COMPANIES_CSV = OUTPUT_DIR / "unmatched_companies.csv"
PARSE_ERRORS_CSV = OUTPUT_DIR / "parse_errors.csv"
AI_ARTICLES_ENRICHED_CSV = OUTPUT_DIR / "ai_articles_enriched.csv"

# Optional persisted DuckDB file.
DUCKDB_PATH = OUTPUT_DIR / "yipitdata.duckdb"

# Optional serialized embedding artifacts (numpy archive + id order).
EMBEDDINGS_DIR = OUTPUT_DIR / "embeddings"
EMBEDDING_IDS_PATH = EMBEDDINGS_DIR / "embedding_ids.json"
EMBEDDING_NPY_PATH = EMBEDDINGS_DIR / "embeddings.npy"

EMBEDDINGS_MODEL = os.environ.get("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
EMBEDDING_DIM = 384  # all-MiniLM-L6-v2 output dimension

# ---------------------------------------------------------------------------
# FX conversion (per the assignment)
# ---------------------------------------------------------------------------
FX_TO_USD = {
    "USD": 1.0,
    "EUR": 1.1,    # EUR -> USD: multiply by 1.1
    "GBP": 1.27,   # GBP -> USD: multiply by 1.27
    "JPY": 1 / 150,  # JPY -> USD: divide by 150
}

# ---------------------------------------------------------------------------
# Company size thresholds
# ---------------------------------------------------------------------------
SIZE_SMALL_MAX = 10_000          # fewer than 10,000 -> Small
SIZE_MEDIUM_MAX = 30_000         # 10,000 through 30,000 -> Medium
SIZE_MEDIUM_LABEL = "Medium"

# ---------------------------------------------------------------------------
# Date ambiguity: we interpret slash dates as US MM/DD/YYYY when the leading
# token is a valid month (1-12); this mirrors the sample values where
# '02/23/2023' clearly means Feb 23. Dash-separated numeric dates such as
# '23-08-2023' are treated as DD-MM-YYYY (EU style).
# ---------------------------------------------------------------------------
US_SLASH_FORMAT = "%m/%d/%Y"
EU_SLASH_FORMAT = "%d/%m/%Y"

# ---------------------------------------------------------------------------
# Category taxonomy: source category -> standardized group.
# ---------------------------------------------------------------------------
CATEGORY_MAP = {
    "Artificial Intelligence": "AI_ML",
    "Machine Learning": "AI_ML",
    "AI/ML": "AI_ML",
    "AI & ML": "AI_ML",
    "Cloud Computing": "CLOUD_COMPUTING",
    "Cloud": "CLOUD_COMPUTING",
    "Cloud Services": "CLOUD_COMPUTING",
    "Financial Technology": "FINTECH",
    "FinTech": "FINTECH",
    "Finance": "FINTECH",
    "SaaS": "SOFTWARE",
    "Software": "SOFTWARE",
    "Enterprise Software": "SOFTWARE",
    "Cybersecurity": "CYBERSECURITY",
    "Security": "CYBERSECURITY",
    "InfoSec": "CYBERSECURITY",
    "Data Analytics": "DATA_ANALYTICS",
    "Analytics": "DATA_ANALYTICS",
    "Big Data": "DATA_ANALYTICS",
}

# Categories that count as AI / Machine Learning for the AI export filter.
AI_CATEGORIES = {"AI_ML"}

# Industries that count as AI / ML for the AI export filter.
AI_INDUSTRIES = {"AI/ML", "AI_ML"}

# ---------------------------------------------------------------------------
# AI article export filters
# ---------------------------------------------------------------------------
AI_YEAR_FROM = 2022
AI_YEAR_TO = 2024
AI_ARR_MIN_USD = 50_000_000

# ---------------------------------------------------------------------------
# ARR interpretation / missing handling
# ---------------------------------------------------------------------------
MISSING_REVENUE_TOKENS = {
    "n/a", "na", "not disclosed", "undisclosed", "null", "none", "-", "--",
    "nan", "missing",
}

# Values that are unparseable/undisclosed are never treated as valid ARR.
