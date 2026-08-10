"""
test_ingest.py

Unit tests for the ingest module.

The tests cover two things: that a valid export loads correctly, and that
each kind of malformed input is rejected with a clear error rather than
producing silently wrong data. The second group matters more, because a
simulation built on quietly corrupted input would produce plausible looking
results that are wrong.
"""

import os
import sys

import pandas as pd
import pytest

# Allow the tests to import from src/ when pytest is run from the project root.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from ingest import load_trades, TradeDataError, REQUIRED_COLUMNS


FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures")

# Known facts about the real export, used as expected values below.
EXPECTED_TRADE_COUNT = 736
EXPECTED_STARTING_BALANCE = 100000.00


def fixture(name):
    """Build the path to a test fixture file."""
    return os.path.join(FIXTURES, name)


# ---------------------------------------------------------------------------
# Loading a valid export
# ---------------------------------------------------------------------------

def test_loads_expected_number_of_trades():
    trades, _ = load_trades(fixture("sample_trades.csv"))
    assert len(trades) == EXPECTED_TRADE_COUNT


def test_returns_only_the_required_columns():
    trades, _ = load_trades(fixture("sample_trades.csv"))
    assert list(trades.columns) == REQUIRED_COLUMNS


def test_timestamps_are_parsed_as_datetimes():
    """
    If the dates were left as text, sorting would be alphabetical rather than
    chronological, which would silently corrupt every equity curve.
    """
    trades, _ = load_trades(fixture("sample_trades.csv"))
    assert pd.api.types.is_datetime64_any_dtype(trades["Open time"])
    assert pd.api.types.is_datetime64_any_dtype(trades["Close time"])


def test_profit_and_balance_are_numeric():
    trades, _ = load_trades(fixture("sample_trades.csv"))
    assert pd.api.types.is_numeric_dtype(trades["Profit/Loss"])
    assert pd.api.types.is_numeric_dtype(trades["Balance"])


def test_derives_starting_balance_from_first_trade():
    _, starting_balance = load_trades(fixture("sample_trades.csv"))
    assert starting_balance == pytest.approx(EXPECTED_STARTING_BALANCE)


def test_explicit_starting_balance_overrides_derived_value():
    _, starting_balance = load_trades(
        fixture("sample_trades.csv"), starting_balance=50000
    )
    assert starting_balance == 50000


def test_trades_are_sorted_by_close_time():
    trades, _ = load_trades(fixture("sample_trades.csv"))
    close_times = trades["Close time"]
    assert close_times.is_monotonic_increasing


def test_sample_type_column_is_retained():
    """
    The in-sample / out-of-sample labelling is needed later to compare
    optimised periods against held-back ones, so it must survive ingest.
    """
    trades, _ = load_trades(fixture("sample_trades.csv"))
    assert "Sample type" in trades.columns
    assert trades["Sample type"].notna().all()


# ---------------------------------------------------------------------------
# Rejecting malformed input
# ---------------------------------------------------------------------------

def test_missing_column_raises_error():
    with pytest.raises(TradeDataError) as error:
        load_trades(fixture("missing_column.csv"))
    assert "Profit/Loss" in str(error.value)


def test_non_numeric_profit_raises_error():
    with pytest.raises(TradeDataError) as error:
        load_trades(fixture("bad_profit.csv"))
    assert "Profit/Loss" in str(error.value)


def test_bad_date_format_raises_error():
    with pytest.raises(TradeDataError) as error:
        load_trades(fixture("bad_date.csv"))
    assert "Close time" in str(error.value)


def test_file_with_no_trades_raises_error():
    with pytest.raises(TradeDataError):
        load_trades(fixture("empty.csv"))


def test_missing_file_raises_error():
    with pytest.raises(TradeDataError) as error:
        load_trades(fixture("does_not_exist.csv"))
    assert "not found" in str(error.value).lower()
