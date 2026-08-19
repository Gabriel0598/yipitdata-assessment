"""Company identity resolution and metadata enrichment.

Two jobs:
  1. Resolve the article's ``company_name`` to a canonical identity present in
     ``company_metadata.json`` via an explicit alias map and fuzzy matching.
  2. Validate which article companies could not be matched so they are
     exported (``unmatched_companies.csv``) instead of being silently dropped.
"""
from __future__ import annotations

from rapidfuzz import fuzz

from .. import config


def _normalize(name: str) -> str:
    return name.strip().lower()


def _norm_compact(name: str) -> str:
    """Lowercase, strip punctuation/whitespace for fuzzy comparison."""
    return "".join(ch for ch in _normalize(name) if ch.isalnum())


# Explicit alias map: raw article company name -> canonical metadata key.
# These handle the deterministic variations present in the source data.
ALIAS_MAP = {
    "AWS": "Amazon Web Services",
    "Amazon Web Services (AWS)": "Amazon Web Services",
    "Amazon Websevices": "Amazon Web Services",
    "Azure": "Microsoft",
    "Microsoft Azure": "Microsoft",
    "DeepMind": "Google DeepMind",
    "Google Deepmind": "Google DeepMind",
    "Nvidia": "NVIDIA",
    "NVIDIA Corporation": "NVIDIA",
    "Open AI": "OpenAI",
    "OpenAI Inc.": "OpenAI",
    "Databricks Inc.": "Databricks",
    "Snowflake Inc.": "Snowflake",
    "CloudFlare": "Cloudflare",
    "Data Robot": "DataRobot",
    "Mongo DB": "MongoDB",
    "Meta AI Research": "Meta AI",
    "Facebook AI Research": "Meta AI",
    "Stripe Inc.": "Stripe",
    "Palantir Technologies": "Palantir",
    "The Boring Company / SpaceX": "SpaceX",
}

# Companies present in the articles but with no metadata entry. They cannot be
# resolved and must be surfaced (not silently dropped). Kept explicit so the
# fuzzy matcher does not accidentally over-match them.
OBSERVED_UNMATCHED = {
    "Cohere",
    "xAI",
    "Mistral AI",
    "Perplexity AI",
    "Hugging Face",
    "Microsoft Azure",
    "Azure",
}

FUZZY_THRESHOLD = 88.0


class CompanyResolver:
    """Resolves raw article company names to canonical metadata identities."""

    def __init__(self, metadata: dict):
        # metadata: {canonical_name: {..fields..}}
        self.metadata = metadata
        self.canonical_names = sorted(metadata.keys())
        # precompute compact norms for the canonical set
        self._norm_index = {c: _norm_compact(c) for c in self.canonical_names}

    def resolve(self, raw_name: str) -> str | None:
        """Return the canonical name, or None if it cannot be resolved."""
        key = raw_name.strip()
        if not key:
            return None

        # 0) explicit alias map
        if key in ALIAS_MAP:
            alias_target = ALIAS_MAP[key]
            if alias_target in self.metadata:
                return alias_target
            # alias points at a canonical name we have -> validated above

        # Known unmatched (no metadata) names short-circuit.
        if key in OBSERVED_UNMATCHED:
            return None

        # 1) exact match
        if key in self.metadata:
            return key
        # case-insensitive exact
        for canon in self.canonical_names:
            if canon.lower() == key.lower():
                return canon

        # case-insensitive left-embed: "Amazon Web Services (AWS)" -> "..."
        key_low = key.lower()

        # 2) fuzzy match against canonical names
        compact = _norm_compact(key)
        best, best_score = None, 0.0
        for canon in self.canonical_names:
            if _norm_compact(canon) == compact:
                return canon
            score = fuzz.token_sort_ratio(key_low, canon.lower())
            if score > best_score:
                best, best_score = canon, score

        if best is not None and best_score >= FUZZY_THRESHOLD:
            return best
        return None


def compute_size_category(employee_count: int | None) -> str | None:
    """Small <10k, Medium 10k-30k, Large >30k."""
    if employee_count is None:
        return None
    if employee_count < config.SIZE_SMALL_MAX:
        return "Small"
    if employee_count <= config.SIZE_MEDIUM_MAX:
        return config.SIZE_MEDIUM_LABEL
    return "Large"


def compute_company_age(founded_year: int | None, article_year: int | None) -> int | None:
    if founded_year is None or article_year is None:
        return None
    return article_year - founded_year
