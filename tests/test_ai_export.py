"""Tests for the AI articles export filter logic."""
import json

import pandas as pd
import pytest

from src.export import build_ai_articles_enriched
import src.pipeline as pipeline


@pytest.fixture(scope="module")
def result():
    return pipeline.run_pipeline()


def test_ai_export_includes_expected_columns(result):
    ai = build_ai_articles_enriched(result.dim_article, None, None, None)
    expected = {
        "article_id", "title", "company_name", "published_date", "category",
        "arr_usd", "summary", "url", "industry", "founded_year",
        "headquarters", "employee_count", "is_public", "stock_ticker",
        "company_age", "company_size_category", "embedding",
    }
    assert expected <= set(ai.columns)


def test_ai_export_year_window(result):
    ai = build_ai_articles_enriched(result.dim_article, None, None, None)
    years = pd.to_datetime(ai["published_date"]).dt.year
    assert years.between(2022, 2024).all()


def test_ai_export_arr_threshold(result):
    ai = build_ai_articles_enriched(result.dim_article, None, None, None)
    assert (ai["arr_usd"] > 50_000_000).all()


def test_ai_export_is_ai(result):
    """Every row must be AI by category OR by industry."""
    ai = build_ai_articles_enriched(result.dim_article, None, None, None)
    cat_ai = ai["category"].str.contains("AI|Machine Learning|Intelligence", case=False, na=False)
    ind_ai = ai["industry"].isin({"AI/ML"})
    assert (cat_ai | ind_ai).all()


def test_ai_export_embedding_when_provided(result):
    ids = list(result.silver_articles["article_id"])
    fake = [[0.0] * 384 for _ in ids]
    ai = build_ai_articles_enriched(result.dim_article, ids, fake, None)
    assert ai["embedding"].notna().all()
    first = json.loads(ai["embedding"].iloc[0])
    assert len(first) == 384


def test_ai_export_top_similar(result):
    ids = list(result.silver_articles["article_id"])
    fake = [[0.0] * 384 for _ in ids]
    top = [["A", "B", "C"] for _ in ids]
    ai = build_ai_articles_enriched(result.dim_article, ids, fake, top)
    assert "top_similar_articles" in ai.columns
