"""Medallion pipeline orchestration (bronze -> silver -> gold).

Design goals:
  - **Lineage**: bronze tables hold the raw source plus a source file hash and
    an ingestion timestamp; every downstream artifact references article_id and
    company_id so records can be traced back to source.
  - **Idempotency**: all surrogate keys are deterministic hashes of natural
    keys. Re-running the pipeline fully rebuilds from raw, so repeated runs
    never create duplicate modeled records.
  - **Small, local, batch-friendly**: uses pandas DataFrames in memory and
    writes one CSV per table.

Grain of the gold model:
  - ``dim_company``: one row per canonical (resolved) company.
  - ``dim_article``: one row per article, with the raw and clean fields and a
    foreign key to the (possibly None) resolved company.
  - ``fact_arr_observation``: one row per **valid** ARR observation, keyed by
    (company_id, article_id, observation_date). Invalid/missing ARR values are
    excluded here but retained in ``dim_article`` (arr_usd NULL) and surfaced
    via ``parse_errors.csv`` for audit.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
from dataclasses import dataclass

import pandas as pd

from . import config
from .cleaning import categories as cat_mod
from .cleaning import companies as comp_mod
from .cleaning.dates import parse_date
from .cleaning.revenue import parse_revenue


def _sha1(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def _company_id(canonical: str) -> str:
    return "COMP-" + _sha1(canonical)[:16].upper()


def _article_id(article_id: str) -> str:
    return "ART-" + _sha1(str(article_id))[:16].upper()


def _arr_obs_id(company_id: str, article_id: str, obs_date: str) -> str:
    return "ARR-" + _sha1(f"{company_id}|{article_id}|{obs_date}")[:24].upper()


def _file_hash(path) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


@dataclass
class PipelineResult:
    bronze_articles: pd.DataFrame
    bronze_companies: pd.DataFrame
    silver_articles: pd.DataFrame
    companies: pd.DataFrame        # dim_company
    dim_article: pd.DataFrame
    fact_arr: pd.DataFrame         # fact_arr_observation
    unmatched: pd.DataFrame
    parse_errors: pd.DataFrame


def run_pipeline() -> PipelineResult:
    """Run the full bronze -> silver -> gold build and return all tables."""
    bronze_articles, bronze_companies = load_bronze()
    metadata = json.load(open(config.COMPANY_METADATA_RAW))

    silver_articles, errors = clean_articles(bronze_articles, metadata)

    dim_company, dim_article, fact = build_gold(silver_articles, metadata)

    # Unmatched companies (present in articles, absent from metadata).
    resolved = silver_articles["company_name_resolved"].notna()
    unseen = silver_articles.loc[~resolved, "company_name_raw"]
    unmatched = (
        unseen.value_counts()
        .rename("article_count")
        .reset_index()
        .rename(columns={"index": "company_name_raw"})
    )
    if unmatched.empty:
        unmatched = pd.DataFrame(columns=["company_name_raw", "article_count"])
    else:
        unmatched["article_count"] = unmatched["article_count"].astype("int64")

    return PipelineResult(
        bronze_articles=bronze_articles,
        bronze_companies=bronze_companies,
        silver_articles=silver_articles,
        companies=dim_company,
        dim_article=dim_article,
        fact_arr=fact,
        unmatched=unmatched,
        parse_errors=errors,
    )


# ---------------------------------------------------------------------------
# Bronze: load raw sources as-is, preserving provenance.
# ---------------------------------------------------------------------------
def load_bronze() -> tuple[pd.DataFrame, pd.DataFrame]:
    now = dt.datetime.now(dt.timezone.utc).isoformat()
    articles = pd.read_csv(config.ARTICLES_RAW, dtype=str).fillna("")
    # Preserve natural article ids but also expose the raw source.
    articles.columns = [c.strip() for c in articles.columns]
    bronze_articles = articles.copy()
    bronze_articles["source_file"] = config.ARTICLES_RAW
    bronze_articles["source_hash"] = _file_hash(config.ARTICLES_RAW)
    bronze_articles["ingested_at"] = now

    meta = json.load(open(config.COMPANY_METADATA_RAW))
    records = []
    for name, fields in meta.items():
        rec = {"company_name": name}
        rec.update(fields)
        records.append(rec)
    bronze_companies = pd.DataFrame(records)
    bronze_companies["source_file"] = config.COMPANY_METADATA_RAW
    bronze_companies["source_hash"] = _file_hash(config.COMPANY_METADATA_RAW)
    bronze_companies["ingested_at"] = now
    return bronze_articles, bronze_companies


# ---------------------------------------------------------------------------
# Silver: clean and enrich articles.
# ---------------------------------------------------------------------------
def clean_articles(article_frame: pd.DataFrame, metadata: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    resolver = comp_mod.CompanyResolver(metadata)
    out = []
    errors = []

    for _, row in article_frame.iterrows():
        raw_company = (row.get("company_name") or "").strip()
        canonical = resolver.resolve(raw_company)

        pd_ = parse_date(row.get("published_date"))
        rev = parse_revenue(row.get("revenue"))

        article_id = _article_id(row.get("article_id"))
        company_id = _company_id(canonical) if canonical else None
        obs_date = pd_.date.strftime("%Y-%m-%d") if pd_.date else None

        out.append({
            "article_id": article_id,
            "source_article_id": row.get("article_id"),
            "title": row.get("title"),
            "company_name_raw": raw_company,
            "company_name_resolved": canonical,
            "company_id": company_id,
            "published_date": obs_date,
            "published_year": pd_.year,
            "published_quarter": pd_.quarter,
            "published_month": pd_.month,
            "category_raw": row.get("category"),
            "category_standardized": cat_mod.standardize_category(row.get("category")),
            "revenue_raw": row.get("revenue"),
            "revenue_currency": rev.currency_original,
            "revenue_is_range": rev.is_range,
            "arr_usd": rev.arr_usd,
            "arr_parse_status": rev.parse_status,
            "summary": row.get("summary"),
            "url": row.get("url"),
            "author": row.get("author"),
            "word_count": row.get("word_count"),
        })

        # Surface any parse issue for audit.
        if rev.parse_status not in ("valid", "valid_range"):
            errors.append({
                "article_id": article_id,
                "source_article_id": row.get("article_id"),
                "company_name": canonical or raw_company,
                "field": "revenue",
                "raw_value": row.get("revenue"),
                "issue": rev.parse_status,
            })
        if pd_.parse_status != "valid":
            errors.append({
                "article_id": article_id,
                "source_article_id": row.get("article_id"),
                "company_name": canonical or raw_company,
                "field": "published_date",
                "raw_value": row.get("published_date"),
                "issue": pd_.parse_status,
            })

    silver = pd.DataFrame(out)
    errors_df = pd.DataFrame(errors)
    return silver, errors_df


# ---------------------------------------------------------------------------
# Gold: dimensional model.
# ---------------------------------------------------------------------------
def build_gold(silver_articles: pd.DataFrame, metadata: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # dim_company: one row per canonical company found in metadata.
    comp_rows = []
    for name, fields in metadata.items():
        comp_rows.append({
            "company_id": _company_id(name),
            "company_name": name,
            "industry": fields.get("industry"),
            "founded_year": fields.get("founded_year"),
            "headquarters": fields.get("headquarters"),
            "employee_count": fields.get("employee_count"),
            "is_public": fields.get("is_public"),
            "stock_ticker": fields.get("stock_ticker"),
        })
    dim_company = pd.DataFrame(comp_rows)
    dim_company["founded_year"] = pd.to_numeric(dim_company["founded_year"], errors="coerce").astype("Int64")
    dim_company["employee_count"] = pd.to_numeric(dim_company["employee_count"], errors="coerce").astype("Int64")

    # dim_article: all articles (including unmatched / invalid) for lineage.
    dim_article = silver_articles.copy()

    # Dim company enrichments computed at article grain: company_age and
    # company_size_category are properties that depend on the article date.
    # We compute them on dim_article (joined to company fields) below.
    dim_article = dim_article.merge(
        dim_company.drop(columns=["company_name"]), on="company_id", how="left",
    )
    dim_article["company_age"] = dim_article.apply(
        lambda r: comp_mod.compute_company_age(r["founded_year"], r["published_year"]), axis=1,
    )
    dim_article["company_size_category"] = dim_article["employee_count"].apply(
        comp_mod.compute_size_category
    )

    # Coerce integer metadata columns to nullable Int64 so they do not render
    # as floats in exports.
    for col in ("founded_year", "employee_count", "company_age"):
        vals = dim_article[col]
        if hasattr(vals, "astype"):
            dim_article[col] = pd.to_numeric(vals, errors="coerce").astype("Int64")
    dim_article["is_public"] = dim_article["is_public"].map(
        lambda v: bool(v) if v is not None and str(v).lower() != "nan" else None
    )

    # fact_arr_observation: one row per valid ARR observation.
    valid = silver_articles[
        silver_articles["arr_usd"].notna()
        & silver_articles["company_id"].notna()
        & silver_articles["published_date"].notna()
    ].copy()
    fact = valid[["company_id", "article_id", "published_date", "arr_usd",
                  "revenue_raw", "revenue_currency", "revenue_is_range"]].copy()
    fact["arr_observation_id"] = [
        _arr_obs_id(cid, aid, d)
        for cid, aid, d in zip(fact["company_id"], fact["article_id"], fact["published_date"])
    ]
    fact["arr_usd"] = fact["arr_usd"].astype("int64")
    # Rename for the model's ARR interpretation.
    fact = fact.rename(columns={
        "published_date": "observation_date",
        "revenue_raw": "arr_raw_value",
        "revenue_currency": "arr_currency",
        "revenue_is_range": "arr_is_range",
    })
    fact = fact[["arr_observation_id", "company_id", "article_id", "observation_date",
                 "arr_usd", "arr_raw_value", "arr_currency", "arr_is_range"]]

    return dim_company, dim_article, fact
