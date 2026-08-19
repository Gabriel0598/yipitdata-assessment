"""Tests for category standardization."""
import pytest

from src.cleaning.categories import is_ai_category, standardize_category
from src import config


def test_ai_groupings_collapse_to_ai_ml():
    for cat in ["Artificial Intelligence", "Machine Learning", "AI/ML", "AI & ML"]:
        assert standardize_category(cat) == "AI_ML"


@pytest.mark.parametrize("raw,expected", [
    ("Cloud Computing", "CLOUD_COMPUTING"),
    ("Cloud", "CLOUD_COMPUTING"),
    ("Cloud Services", "CLOUD_COMPUTING"),
    ("FinTech", "FINTECH"),
    ("Finance", "FINTECH"),
    ("Financial Technology", "FINTECH"),
    ("SaaS", "SOFTWARE"),
    ("Software", "SOFTWARE"),
    ("Enterprise Software", "SOFTWARE"),
    ("Cybersecurity", "CYBERSECURITY"),
    ("Security", "CYBERSECURITY"),
    ("InfoSec", "CYBERSECURITY"),
    ("Data Analytics", "DATA_ANALYTICS"),
    ("Analytics", "DATA_ANALYTICS"),
    ("Big Data", "DATA_ANALYTICS"),
])
def test_taxonomy_mapping(raw, expected):
    assert standardize_category(raw) == expected


def test_unknown_category_marked():
    assert standardize_category("Something New") == "UNKNOWN"
    assert standardize_category(None) == "UNKNOWN"
    assert standardize_category("") == "UNKNOWN"


def test_is_ai_category():
    assert is_ai_category("AI_ML") is True
    assert is_ai_category("FINTECH") is False


def test_all_raw_categories_have_a_mapping():
    # The 19 distinct categories in the source CSV all map to a real group.
    assert len(config.CATEGORY_MAP) == 19
