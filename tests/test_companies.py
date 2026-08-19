"""Tests for company identity resolution, sizes and enrichment."""
import json

import pytest

from src.cleaning.companies import (CompanyResolver, compute_company_age,
                                    compute_size_category)
from src import config


@pytest.fixture(scope="module")
def resolver():
    meta = json.load(open(config.COMPANY_METADATA_RAW))
    return CompanyResolver(meta), meta


def test_exact_match(resolver):
    r, _ = resolver
    assert r.resolve("OpenAI") == "OpenAI"
    assert r.resolve("Databricks") == "Databricks"


@pytest.mark.parametrize("raw,expected", [
    ("AWS", "Amazon Web Services"),
    ("Amazon Web Services (AWS)", "Amazon Web Services"),
    ("Azure", "Microsoft"),
    ("Microsoft Azure", "Microsoft"),
    ("DeepMind", "Google DeepMind"),
    ("Google Deepmind", "Google DeepMind"),
    ("Nvidia", "NVIDIA"),
    ("NVIDIA Corporation", "NVIDIA"),
    ("Open AI", "OpenAI"),
    ("OpenAI Inc.", "OpenAI"),
    ("Data Robot", "DataRobot"),
    ("Mongo DB", "MongoDB"),
    ("CloudFlare", "Cloudflare"),
    ("Palantir Technologies", "Palantir"),
    ("The Boring Company / SpaceX", "SpaceX"),
    ("Facebook AI Research", "Meta AI"),
])
def test_alias_mapping(resolver, raw, expected):
    r, _ = resolver
    assert r.resolve(raw) == expected


@pytest.mark.parametrize("raw", ["Cohere", "xAI", "Mistral AI", "Perplexity AI", "Hugging Face"])
def test_known_unmatched_stays_unmatched(resolver, raw):
    r, _ = resolver
    assert r.resolve(raw) is None


def test_all_metadata_companies_resolvable(resolver):
    r, meta = resolver
    for name in meta:
        assert r.resolve(name) == name


def test_size_categories():
    assert compute_size_category(9_999) == "Small"
    assert compute_size_category(10_000) == "Medium"
    assert compute_size_category(30_000) == "Medium"
    assert compute_size_category(30_001) == "Large"
    assert compute_size_category(None) is None


def test_company_age():
    assert compute_company_age(2015, 2023) == 8
    assert compute_company_age(None, 2023) is None
    assert compute_company_age(2015, None) is None
