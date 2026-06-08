"""Robust parsing of free-form (PubMed-style) reference publication dates.

PubMed ``date_published`` values are messy: ISO dates, year-only (``2019``),
year+month (``2019-03``, ``Mar 2019``, ``2019 Mar``), month ranges
(``2019 Mar-Apr``), seasons (``Summer 2019``), and more. This module parses
them to a ``datetime`` for comparison, resolving missing components to the
earliest instant (year -> Jan 1, year+month -> the 1st).
"""

import logging
import re
from datetime import datetime
from typing import Optional

from dateutil import parser as _dateutil_parser

logger = logging.getLogger(__name__)

# Plausible publication years (1500-2199); avoids matching stray numbers.
_YEAR_RE = re.compile(r"\b(1[5-9]\d{2}|20\d{2}|21\d{2})\b")
# Collapse an alphabetic month range like "Mar-Apr" / "Mar / Apr" to its first
# month so dateutil reads a single date. Restricted to alpha tokens so it never
# touches numeric ISO dates ("2019-03-15").
_MONTH_RANGE_RE = re.compile(r"(?i)\b([a-z]{3,9})\s*[-/]\s*[a-z]{3,9}\b")


def parse_reference_date(date_str: Optional[str]) -> Optional[datetime]:
    """Best-effort parse of a free-form reference date.

    Returns a ``datetime`` (missing components default to the earliest instant),
    or ``None`` when the input is empty or contains no plausible 4-digit year.
    """
    if not date_str:
        return None
    s = str(date_str).strip()
    if not s:
        return None

    # Fast path: ISO date / datetime ("2019-03-15", "2019-03-15T10:00:00").
    for fmt in ("%Y-%m-%d", "%Y/%m/%d"):
        try:
            return datetime.strptime(s[:10], fmt)
        except ValueError:
            pass
    # Year + numeric month ("2019-03").
    try:
        return datetime.strptime(s[:7], "%Y-%m")
    except ValueError:
        pass

    # Free-form: require a plausible year, then let dateutil read whatever month/
    # day it can, anchoring missing components to Jan 1 of that year.
    match = _YEAR_RE.search(s)
    if not match:
        return None
    year = int(match.group(1))
    cleaned = _MONTH_RANGE_RE.sub(r"\1", s)
    try:
        return _dateutil_parser.parse(cleaned, default=datetime(year, 1, 1), fuzzy=True)
    except (ValueError, OverflowError, TypeError):
        # Year is known even if the rest is unparseable.
        return datetime(year, 1, 1)
