"""Tests for date normalization and ambiguity handling."""
import pytest

from src.cleaning.dates import parse_date


@pytest.mark.parametrize("raw,expected", [
    # ISO
    ("2022-02-17", "2022-02-17"),
    ("2023-12-13", "2023-12-13"),
    # ISO with timezone
    ("2021-09-11T00:00:00Z", "2021-09-11"),
    # US slash (MM/DD/YYYY)
    ("02/23/2023", "2023-02-23"),
    ("12/28/2021", "2021-12-28"),
    ("04/30/2020", "2020-04-30"),
    # EU slash fallback (leading token not a valid month -> DD/MM/YYYY)
    ("13/05/2023", "2023-05-13"),
    # Dash dates (mixed US/EU resolved by which token is a valid month)
    ("23-08-2023", "2023-08-23"),
    ("06-21-2024", "2024-06-21"),
    ("10-28-2024", "2024-10-28"),
    ("05-15-2023", "2023-05-15"),
    # Full month text
    ("October 19, 2022", "2022-10-19"),
    ("February 25, 2020", "2020-02-25"),
    ("September 07, 2021", "2021-09-07"),
    # Day before month name
    ("21 Feb 2020", "2020-02-21"),
    ("27 Sep 2020", "2020-09-27"),
])
def test_parse_valid_dates(raw, expected):
    p = parse_date(raw)
    assert p.parse_status == "valid"
    assert p.date.strftime("%Y-%m-%d") == expected


def test_year_quarter_month_extracted():
    p = parse_date("2023-12-13")
    assert (p.year, p.quarter, p.month) == (2023, 4, 12)
    p2 = parse_date("2022-03-01")
    assert (p2.year, p2.quarter, p2.month) == (2022, 1, 3)


@pytest.mark.parametrize("raw", [None, "", "not a date", "garbage", "0000-99-99"])
def test_invalid_dates(raw):
    p = parse_date(raw)
    assert p.parse_status in ("invalid", "invalid_missing")
    assert p.date is None
