"""Revenue / ARR cleaning.

Parses the messy ``revenue`` column into a normalized integer USD value that
is interpreted as a reported ARR observation for the article's company as of
the article's published date.

Supported formats (golden cases covered in tests):
  - Raw integers with thousands separators: ``$5,200,000,000``, ``£244,094,488``,
    ``€1,254,545,455``, ``¥360,000,000,000``, ``$790,000,000``
  - Suffix scales: ``$980.0M``, ``$16.800B``, ``75000.0M USD`` (trailing zeros are
    decimal noise, i.e. ``16.800B`` -> 16.8B)
  - Word scales: ``$1.480 billion``, ``5.2 billion``, ``$135.0 million``
  - Bars/bare numbers with currency code: ``500M USD``, ``5.2B``
  - Ranges: ``$38475.0M - $42525.0M`` -> midpoint
  - Missing / null / "N/A" / "Not disclosed" -> NULL (never a valid ARR)

Return value is a :class:`ParsedRevenue` dataclass carrying the raw value,
the original currency, whether it was a range, and a ``parse_status`` so the
pipeline can retain lineage and surface failed parses instead of silently
swallowing them.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from .. import config

_MULTIPLIERS = {
    "THOUSAND": 1_000,
    "MILLION": 1_000_000,
    "BILLION": 1_000_000_000,
    "TRILLION": 1_000_000_000_000,
    # single letter suffixes
    "K": 1_000,
    "M": 1_000_000,
    "B": 1_000_000_000,
    "T": 1_000_000_000_000,
}

_WORD_PATTERN = re.compile(
    r"(thousand|million|billion|trillion)", re.IGNORECASE
)
# A number (with optional decimal / millions separators) immediately followed
# by a single-letter scale suffix that is not part of a currency code.
_SUFFIX_PATTERN = re.compile(r"([0-9][0-9.,]*)\s*([KMBT])\b", re.IGNORECASE)
_CURRENCY_PATTERN = re.compile(r"\b(USD|EUR|GBP|JPY)\b", re.IGNORECASE)
_SYMBOL_CURRENCY = {"¥": "JPY", "€": "EUR", "£": "GBP", "$": "USD"}
_RANGE_SPLIT = re.compile(r"\s*(?:-|to|–|—)\s*")


@dataclass
class ParsedRevenue:
    arr_usd: int | None = None
    currency_original: str | None = None
    is_range: bool = False
    parse_status: str = "valid"
    raw_value: str = ""
    note: str | None = None


def _is_missing(raw: str) -> bool:
    return raw.lower() in config.MISSING_REVENUE_TOKENS


def _currency_of(raw: str) -> str:
    """Best-effort currency detection from symbol or ISO code (default USD)."""
    for sym, cur in _SYMBOL_CURRENCY.items():
        if sym in raw:
            return cur
    m = _CURRENCY_PATTERN.search(raw)
    if m:
        return m.group(1).upper()
    return "USD"


def _parse_amount(raw: str) -> float | None:
    """Parse a single non-range amount into a USD amount (already converted).

    Returns ``None`` when no numeric value can be recognized.
    """
    work = raw.strip()
    if not work:
        return None

    # Strip currency codes ('USD', 'EUR'...) and symbols; keep number + scale.
    work = _CURRENCY_PATTERN.sub("", work)
    for sym in _SYMBOL_CURRENCY:
        work = work.replace(sym, "")

    scale = 1.0

    # Word scale first (e.g. '1.480 billion').
    mw = _WORD_PATTERN.search(work)
    if mw:
        scale = _MULTIPLIERS[mw.group(1).upper()]
        work = _WORD_PATTERN.sub("", work)

    # Single-letter suffix scale (e.g. '980.0M', '75000.0M').
    ms = _SUFFIX_PATTERN.search(work)
    if ms:
        scale = _MULTIPLIERS[ms.group(2).upper()]
        # Remove the suffix letter so it is not treated as a number char.
        work = work.replace(ms.group(2), "", 1)

    # Extract the leading numeric token. Remove thousands separators: since
    # currency codes/symbols are gone and decimal dots only appear with a
    # scale, strip commas and (European) multi-dot separators conservatively.
    token_match = re.search(r"[0-9][0-9.,]*", work)
    if not token_match:
        return None
    num_str = token_match.group(0).replace(",", "")

    # If there is more than one dot, it is a European thousands separator.
    if num_str.count(".") > 1:
        num_str = num_str.replace(".", "")

    try:
        value = float(num_str) * scale
    except ValueError:
        return None

    # FX conversion: EUR *1.1, GBP *1.27, USD *1.0, JPY /150 (rate 1/150).
    value = value * config.FX_TO_USD[_currency_of(raw)]
    return value


def parse_revenue(raw: str | None) -> ParsedRevenue:
    """Parse a raw revenue string into a normalized USD ARR value."""
    if raw is None:
        return ParsedRevenue(raw_value=raw, parse_status="invalid_missing")

    raw = raw.strip()
    if raw == "" or _is_missing(raw):
        return ParsedRevenue(raw_value=raw, parse_status="invalid_missing")

    currency = _currency_of(raw)

    # ---- Ranges: take the midpoint ----
    parts = _RANGE_SPLIT.split(raw)
    if len(parts) == 2:
        lo = _parse_amount(parts[0])
        hi = _parse_amount(parts[1])
        if lo is not None and hi is not None:
            mid = (lo + hi) / 2.0
            return ParsedRevenue(
                arr_usd=int(round(mid)),
                currency_original=currency,
                is_range=True,
                parse_status="valid_range",
                raw_value=raw,
            )

    amount = _parse_amount(raw)
    if amount is None:
        return ParsedRevenue(raw_value=raw, parse_status="unparseable")

    return ParsedRevenue(
        arr_usd=int(round(amount)),
        currency_original=currency,
        parse_status="valid",
        raw_value=raw,
    )
