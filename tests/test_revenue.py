"""Golden-case tests for the revenue/ARR parser."""
import pytest

from src.cleaning.revenue import parse_revenue


@pytest.mark.parametrize("raw,expected", [
    # Raw integers with thousands separators
    ("$5,200,000,000", 5_200_000_000),
    ("$790,000,000", 790_000_000),
    ("$4,200,000,000", 4_200_000_000),
    ("$45,000,000", 45_000_000),
    ("$54,000,000,000", 54_000_000_000),
    # Suffix scales (M / B)
    ("$980.0M", 980_000_000),
    ("$16.800B", 16_800_000_000),
    ("$170.0M", 170_000_000),
    ("$1.030B", 1_030_000_000),
    ("$12.600B", 12_600_000_000),
    ("75000.0M USD", 75_000_000_000),
    ("2300.0M USD", 2_300_000_000),
    ("295.0M USD", 295_000_000),
    ("500M USD", 500_000_000),
    ("5.2B", 5_200_000_000),
    # Word scales
    ("$1.480 billion", 1_480_000_000),
    ("$0.930 billion", 930_000_000),
    ("$55.000 billion", 55_000_000_000),
    ("$135.0 million", 135_000_000),
    ("$7000.0 million", 7_000_000_000),
    ("5.2 billion", 5_200_000_000),
    ("$82000.0 million", 82_000_000_000),
    # Currency conversions: EUR *1.1
    ("€1,254,545,455", 1_380_000_000),
    ("€36,818,181,818", 40_500_000_000),
    ("€795,454,545", 875_000_000),
    # Currency conversions: GBP *1.27
    ("£244,094,488", 310_000_000),
    ("£299,212,598", 379_999_999),
    ("£114,173,228", 145_000_000),
    # Currency conversions: JPY /150
    ("¥360,000,000,000", 2_400_000_000),
    ("¥2,400,000,000,000", 16_000_000_000),
    ("¥255,000,000,000", 1_700_000_000),
])
def test_parse_valid_values(raw, expected):
    parsed = parse_revenue(raw)
    assert parsed.parse_status == "valid"
    assert parsed.arr_usd == expected


@pytest.mark.parametrize("raw,expected", [
    ("$38475.0M - $42525.0M", (38_475_000_000 + 42_525_000_000) // 2),
    ("$57950.0M - $64050.0M", (57_950_000_000 + 64_050_000_000) // 2),
    ("$1330.0M - $1470.0M", (1_330_000_000 + 1_470_000_000) // 2),
    ("$10M - $20M", 15_000_000),
])
def test_parse_ranges_take_midpoint(raw, expected):
    parsed = parse_revenue(raw)
    assert parsed.parse_status == "valid_range"
    assert parsed.is_range is True
    assert parsed.arr_usd == expected


@pytest.mark.parametrize("raw", [
    None, "", "   ", "N/A", "n/a", "Not disclosed", "UNDISCLOSED",
    "null", "-", "missing",
])
def test_parse_missing_never_valid(raw):
    parsed = parse_revenue(raw)
    assert parsed.parse_status == "invalid_missing"
    assert parsed.arr_usd is None


@pytest.mark.parametrize("raw", ["garbage", "unknown format here", "$abc"])
def test_parse_unparseable(raw):
    parsed = parse_revenue(raw)
    assert parsed.parse_status == "unparseable"
    assert parsed.arr_usd is None
