"""
ingest.py

Loads and validates a StrategyQuant X / MetaTrader 5 trade list export and
converts it into a clean internal format for the simulation engine.

This module owns all knowledge of the export file format. Nothing downstream
should need to know that the file is semicolon delimited or that dates use
dots, because format quirks are isolated here (see FR1 to FR4).
"""

import pandas as pd


# The exact datetime format SQX writes. Dots between date parts, not slashes
# or dashes, so pandas cannot infer it reliably and we state it explicitly.
SQX_DATETIME_FORMAT = "%Y.%m.%d %H:%M:%S"

# Columns the rest of the pipeline depends on. The export contains more than
# this, but these are the ones without which the simulation cannot run.
REQUIRED_COLUMNS = [
    "Open time",
    "Close time",
    "Type",
    "Profit/Loss",
    "Balance",
    "Sample type",
]

# Columns kept after cleaning. Everything else in the export (MAE, MFE,
# Comment, Time in trade and so on) is dropped because it is not used by the
# simulation and carrying it adds no value.
KEPT_COLUMNS = REQUIRED_COLUMNS


class TradeDataError(ValueError):
    """
    Raised when a trade export cannot be loaded or fails validation.

    A dedicated exception type means calling code (and the tests) can
    distinguish a genuine data problem from an unrelated ValueError raised
    somewhere inside pandas.
    """


def load_trades(filepath, starting_balance=None):
    """
    Load a trade export and return the cleaned trades plus the starting balance.

    Args:
        filepath: path to the CSV export.
        starting_balance: optional override. If not supplied it is derived
            from the file (see below).

    Returns:
        A tuple of (trades, starting_balance) where trades is a pandas
        DataFrame sorted by close time.

    Raises:
        TradeDataError: if the file cannot be read, is missing required
            columns, contains no trades, or has values of the wrong type.
    """
    trades = _read_file(filepath)
    _validate_columns(trades)
    trades = _parse_and_validate_types(trades)
    trades = _sort_by_close_time(trades)

    if starting_balance is None:
        starting_balance = _derive_starting_balance(trades)

    return trades[KEPT_COLUMNS].reset_index(drop=True), starting_balance


def _read_file(filepath):
    """
    Read the raw CSV.

    SQX writes semicolon separated values with every field wrapped in double
    quotes, so the pandas defaults (comma separated) would put the entire row
    into a single column.
    """
    try:
        trades = pd.read_csv(filepath, sep=";", quotechar='"')
    except FileNotFoundError:
        raise TradeDataError(f"Trade file not found: {filepath}")
    except pd.errors.EmptyDataError:
        raise TradeDataError(f"Trade file is empty: {filepath}")
    except Exception as error:
        # Anything else pandas throws is re-raised in our own type so callers
        # only ever have to handle TradeDataError.
        raise TradeDataError(f"Could not read trade file {filepath}: {error}")

    if trades.empty:
        raise TradeDataError(f"Trade file contains no trades: {filepath}")

    return trades


def _validate_columns(trades):
    """
    Check every required column is present.

    Reporting all missing columns at once, rather than failing on the first
    one, means a user fixing a bad export sees the full problem immediately.
    """
    missing = [column for column in REQUIRED_COLUMNS if column not in trades.columns]

    if missing:
        raise TradeDataError(
            f"Missing required columns: {missing}. "
            f"Found columns: {list(trades.columns)}"
        )


def _parse_and_validate_types(trades):
    """
    Convert timestamps and numeric fields, failing clearly on bad values.

    errors="coerce" turns anything unparseable into NaT/NaN rather than
    throwing, which lets us count the bad rows and report how many there are
    instead of surfacing a raw pandas error.
    """
    trades = trades.copy()

    for column in ["Open time", "Close time"]:
        trades[column] = pd.to_datetime(
            trades[column], format=SQX_DATETIME_FORMAT, errors="coerce"
        )
        bad_rows = trades[column].isna().sum()
        if bad_rows > 0:
            raise TradeDataError(
                f"Column '{column}' has {bad_rows} value(s) that do not match "
                f"the expected format {SQX_DATETIME_FORMAT}"
            )

    for column in ["Profit/Loss", "Balance"]:
        trades[column] = pd.to_numeric(trades[column], errors="coerce")
        bad_rows = trades[column].isna().sum()
        if bad_rows > 0:
            raise TradeDataError(
                f"Column '{column}' has {bad_rows} non-numeric value(s)"
            )

    return trades


def _sort_by_close_time(trades):
    """
    Sort trades into the order they closed.

    Order matters because an equity curve is a running total. The export is
    usually already ordered, but relying on that would be an unstated
    assumption, so we enforce it.
    """
    return trades.sort_values("Close time")


def _derive_starting_balance(trades):
    """
    Work out the account balance before the first trade.

    The export records the balance after each trade but never states the
    opening balance, so it is recovered by subtracting the first trade's
    profit or loss from the balance recorded against that trade.
    """
    first_trade = trades.iloc[0]
    return float(first_trade["Balance"] - first_trade["Profit/Loss"])
