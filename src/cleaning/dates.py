"""Date normalization.

Handles the multiple date formats present in ``published_date``:

  - ISO:            ``2022-02-17``, ``2021-09-11T00:00:00Z``
  - US slash:       ``02/23/2023`` (MM/DD/YYYY)
  - EU-style dash:  ``23-08-2023`` (DD-MM-YYYY)
  - Month text:     ``October 19, 2022``, ``January 5, 2021``
  - Day-abbr month: ``21 Feb 2020``

Ambiguity handling (documented in the README / architecture doc):
  - Two tokens with a **slash** are interpreted as US ``MM/DD/YYYY`` when the
    leading token is a valid month (1-12); this matches the sample data where
    ``02/23/2023`` clearly means Feb 23. We only fall back to EU ``DD/MM/YYYY``
    when the leading token is *not* a valid month but the second token is
    (edge case that does not occur in the provided data).
  - Two tokens with a **dash** are interpreted as EU ``DD-MM-YYYY``.
  - A leading token > 12 in a slash date is treated as a day (EU style).

Returned value is a :class:`ParsedDate` with the ``datetime`` plus the derived
``year``, ``quarter`` and ``month`` columns used for filtering/analysis.
"""
from __future__ import annotations

import datetime as dt
import re
from dataclasses import dataclass

_MONTHS = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "sept": 9, "oct": 10, "nov": 11, "dec": 12,
}
_ISO = re.compile(r"^\d{4}-\d{2}-\d{2}")
_SLASH = re.compile(r"^(\d{1,2})/(\d{1,2})/(\d{4})$")
_DASH = re.compile(r"^(\d{1,2})-(\d{1,2})-(\d{4})$")
_TEXT = re.compile(r"^([A-Za-z]+)\s+(\d{1,2}),\s*(\d{4})$")
_DAY_ABBR = re.compile(r"^(\d{1,2})\s+([A-Za-z]{3,})\s+(\d{4})$")
_ABBR_MONTH = re.compile(r"^([A-Za-z]{3,})\s+(\d{1,2}),?\s*(\d{4})$")


@dataclass
class ParsedDate:
    date: dt.datetime | None = None
    year: int | None = None
    quarter: int | None = None
    month: int | None = None
    parse_status: str = "valid"
    raw_value: str = ""
    note: str | None = None


def _month_number(name: str) -> int | None:
    key = name.strip().lower()
    if key in _MONTHS:
        return _MONTHS[key]
    if key.startswith("sept"):
        return _MONTHS["sept"]
    # fall back to the first three letters
    return _MONTHS.get(key[:3])


def parse_date(raw: str | None) -> ParsedDate:
    p = ParsedDate(raw_value=raw)
    if raw is None:
        p.parse_status = "invalid_missing"
        p.note = "missing value"
        return p
    raw = raw.strip()
    p.raw_value = raw
    if raw == "":
        p.parse_status = "invalid_missing"
        p.note = "empty value"
        return p

    try:
        # ISO (with optional time component).
        if _ISO.match(raw):
            date = dt.datetime.strptime(raw[:10], "%Y-%m-%d")

        # Slash date (US MM/DD/YYYY by default).
        elif (m := _SLASH.match(raw)):
            a, b, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
            if 1 <= a <= 12 and not (1 <= b <= 12 and a > b):
                # leading token is a month -> MM/DD/YYYY
                date = dt.datetime(y, a, b)
            elif 1 <= b <= 12:
                # leading token is the day -> DD/MM/YYYY (EU)
                date = dt.datetime(y, b, a)
            else:
                raise ValueError(f"unresolvable slash date {raw!r}")

        # Dash-separated numeric date: ambiguous -- resolve by which token is
        # a valid month. Sample data mixes US "06-21-2024" (MM-DD) and EU-style
        # "23-08-2023" (DD-MM), so we use the same rule as slash dates.
        elif (m := _DASH.match(raw)):
            a, b, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
            if 1 <= a <= 12 and not (1 <= b <= 12 and a > b):
                date = dt.datetime(y, a, b)      # MM-DD-YYYY
            elif 1 <= b <= 12:
                date = dt.datetime(y, b, a)      # DD-MM-YYYY
            else:
                raise ValueError(f"unresolvable dash date {raw!r}")

        # Full month text: "October 19, 2022"
        elif (m := _TEXT.match(raw)):
            mo = _month_number(m.group(1))
            if mo is None:
                raise ValueError(f"unknown month {m.group(1)!r}")
            date = dt.datetime(int(m.group(3)), mo, int(m.group(2)))

        # Abbreviated month first: "Oct 19, 2022" / "Oct 19 2022"
        elif (m := _ABBR_MONTH.match(raw)):
            mo = _month_number(m.group(1))
            if mo is None:
                raise ValueError(f"unknown month {m.group(1)!r}")
            date = dt.datetime(int(m.group(3)), mo, int(m.group(2)))

        # Day before month name: "21 Feb 2020"
        elif (m := _DAY_ABBR.match(raw)):
            mo = _month_number(m.group(2))
            if mo is None:
                raise ValueError(f"unknown month {m.group(2)!r}")
            date = dt.datetime(int(m.group(3)), mo, int(m.group(1)))

        else:
            raise ValueError(f"unrecognized date format {raw!r}")

    except ValueError as exc:
        p.date = None
        p.parse_status = "invalid"
        p.note = str(exc)
        return p

    p.date = date
    p.year = date.year
    p.month = date.month
    p.quarter = (date.month - 1) // 3 + 1
    p.parse_status = "valid"
    return p
