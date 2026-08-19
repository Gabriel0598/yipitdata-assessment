"""Tests for the pipeline end-to-end and idempotency guarantees."""
import pandas as pd
import pytest

import src.pipeline as pipeline


@pytest.fixture(scope="module")
def result():
    return pipeline.run_pipeline()


def test_row_counts(result):
    assert len(result.bronze_articles) == 750
    assert len(result.silver_articles) == 750
    assert len(result.companies) == 21
    assert len(result.fact_arr) == 538
    assert len(result.unmatched) == 5


def test_every_article_has_unique_id(result):
    ids = result.dim_article["article_id"]
    assert ids.is_unique


def test_fact_arr_grain_is_one_per_valid_observation(result):
    key = ["company_id", "article_id", "observation_date"]
    assert result.fact_arr.duplicated(subset=key).sum() == 0


def test_invalid_arr_excluded_from_fact_but_kept_in_dim(result):
    invalid_in_dim = result.dim_article["arr_usd"].isna().sum()
    assert invalid_in_dim == 192
    # All fact rows are valid (non-null arr_usd).
    assert result.fact_arr["arr_usd"].notna().all()


def test_arr_usd_is_integer(result):
    assert result.fact_arr["arr_usd"].dtype.kind in "iu"


def test_unmatched_companies_listed(result):
    names = set(result.unmatched["company_name_raw"])
    assert {"Cohere", "xAI", "Mistral AI", "Perplexity AI", "Hugging Face"} <= names


def test_dim_company_contains_all_metadata(result):
    meta = result.companies
    assert set(meta["company_id"]).__len__() == 21
    assert "company_id" in meta.columns
    assert "industry" in meta.columns


def test_companies_industry_enrichment(result):
    # dim_article should carry industry from metadata.
    assert "industry" in result.dim_article.columns


@pytest.mark.parametrize("column", [
    "company_age", "company_size_category", "industry", "founded_year",
])
def test_model_has_enrichment_columns(result, column):
    assert column in result.dim_article.columns


def test_parse_errors_records_every_invalid_revenue(result):
    n_invalid = result.parse_errors[result.parse_errors["field"] == "revenue"].shape[0]
    assert n_invalid == 192


def test_idempotent_rebuild():
    """Re-running the pipeline yields identical outputs (no duplicates)."""
    r1 = pipeline.run_pipeline()
    r2 = pipeline.run_pipeline()
    pd.testing.assert_frame_equal(
        r1.fact_arr.sort_values("arr_observation_id").reset_index(drop=True),
        r2.fact_arr.sort_values("arr_observation_id").reset_index(drop=True),
    )
    pd.testing.assert_frame_equal(
        r1.dim_article.sort_values("article_id").reset_index(drop=True),
        r2.dim_article.sort_values("article_id").reset_index(drop=True),
    )
    assert len(r1.fact_arr) == len(r2.fact_arr)
