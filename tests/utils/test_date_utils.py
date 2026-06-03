"""Tests for robust reference-date parsing (training date filter)."""

from datetime import datetime

from utils.date_utils import parse_reference_date


def test_iso_date():
    assert parse_reference_date("2019-03-15") == datetime(2019, 3, 15)


def test_iso_datetime_drops_time():
    assert parse_reference_date("2019-03-15T10:30:00") == datetime(2019, 3, 15)


def test_year_month():
    assert parse_reference_date("2019-03") == datetime(2019, 3, 1)


def test_year_only():
    assert parse_reference_date("2019") == datetime(2019, 1, 1)


def test_month_abbrev_then_year():
    assert parse_reference_date("Mar 2019") == datetime(2019, 3, 1)


def test_year_then_month_abbrev():
    assert parse_reference_date("2019 Mar") == datetime(2019, 3, 1)


def test_full_month_name():
    assert parse_reference_date("March 2019") == datetime(2019, 3, 1)


def test_pubmed_month_range_takes_first_month():
    # "2019 Mar-Apr" -> first month of the range, day defaults to the 1st
    assert parse_reference_date("2019 Mar-Apr") == datetime(2019, 3, 1)


def test_season_falls_back_to_year():
    # "Summer 2019" has no parseable month -> earliest instant of the year
    assert parse_reference_date("Summer 2019") == datetime(2019, 1, 1)


def test_none_and_empty_return_none():
    assert parse_reference_date(None) is None
    assert parse_reference_date("") is None
    assert parse_reference_date("   ") is None


def test_string_without_year_returns_none():
    assert parse_reference_date("no date here") is None
    assert parse_reference_date("Mar") is None
