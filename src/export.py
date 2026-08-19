"""Export helpers: write each modeled table to its own CSV and build the
required ``ai_articles_enriched.csv`` downstream export."""
from __future__ import annotations

import json

import pandas as pd

from . import config
from .cleaning.categories import is_ai_category


def build_ai_articles_enriched(dim_article: pd.DataFrame,
                               all_article_ids: list[str] | None,
                               embeddings,
                               top_similar: list[list[str]] | None) -> pd.DataFrame:
    """Build ``ai_articles_enriched`` filtered to the assignment's criteria.

    ``all_article_ids`` is the ordered id list the ``embeddings`` rows are
    aligned to (in practice ``result.silver_articles["article_id"]``). This
    guarantees embeddings are mapped to the correct article rather than the
    (reordered) filtered subset.
    """
    df = dim_article.copy()

    # Industry AI/ML: metadata industry in AI set.
    df["_industry_ai"] = df["industry"].isin(config.AI_INDUSTRIES)
    df["_category_ai"] = df["category_standardized"].apply(is_ai_category)

    match = (
        (df["_category_ai"] | df["_industry_ai"])
        & df["published_year"].between(config.AI_YEAR_FROM, config.AI_YEAR_TO)
        & (df["arr_usd"] > config.AI_ARR_MIN_USD)
    )
    out = df[match].copy()

    cols = [
        "article_id",
        "title",
        "company_name",
        "published_date",
        "category",
        "arr_usd",
        "summary",
        "url",
        "industry",
        "founded_year",
        "headquarters",
        "employee_count",
        "is_public",
        "stock_ticker",
        "company_age",
        "company_size_category",
        "embedding",
    ]

    # The enriched export uses the raw company/category names.
    out["company_name"] = out["company_name_resolved"].fillna(out["company_name_raw"])
    out["category"] = out["category_raw"]
    out["arr_usd"] = out["arr_usd"].astype("int64")

    # Attach embeddings and top_similar if provided. Map by article_id so the
    # alignment between the filtered subset and the full embeddings is correct.
    if embeddings is not None and all_article_ids is not None:
        id_to_emb = dict(zip(all_article_ids, embeddings))
        out["embedding"] = out["article_id"].map(
            lambda a: json.dumps([float(x) for x in id_to_emb[a]])
        )
    else:
        out["embedding"] = None
    if top_similar is not None:
        if all_article_ids is not None:
            id_to_top = dict(zip(all_article_ids, top_similar))
        else:
            id_to_top = dict(zip(out["article_id"], top_similar))
        out["top_similar_articles"] = out["article_id"].map(lambda a: json.dumps(id_to_top[a]))
    else:
        out["top_similar_articles"] = None

    out = out[cols + (["top_similar_articles"] if top_similar is not None else [])]
    return out.reset_index(drop=True)


def export_all(result, embeddings=None, top_similar=None) -> None:
    """Write every modeled table and the required CSVs to data/output/."""
    bronze_articles, bronze_companies, silver_articles, companies, unmatched, errors = (
        result.bronze_articles,
        result.bronze_companies,
        result.silver_articles,
        result.companies,
        result.unmatched,
        result.parse_errors,
    )

    bronze_articles.to_csv(config.BRONZE_ARTICLES_CSV, index=False)
    bronze_companies.to_csv(config.BRONZE_COMPANIES_CSV, index=False)
    silver_articles.to_csv(config.SILVER_ARTICLES_CSV, index=False)
    companies.to_csv(config.DIM_COMPANY_CSV, index=False)
    result.dim_article.to_csv(config.DIM_ARTICLE_CSV, index=False)
    result.fact_arr.to_csv(config.FACT_ARR_OBSERVATION_CSV, index=False)
    unmatched.to_csv(config.UNMATCHED_COMPANIES_CSV, index=False)
    errors.to_csv(config.PARSE_ERRORS_CSV, index=False)

    ai = build_ai_articles_enriched(result.dim_article,
                                    list(silver_articles["article_id"]),
                                    embeddings, top_similar)
    ai.to_csv(config.AI_ARTICLES_ENRICHED_CSV, index=False)
    return ai
