"""
compare.py

Builds the buy and hold benchmark and compares simulated strategy paths
against it.

Two questions are asked of every simulated path (FR9 to FR11):
  1. Did it finish with a higher return than simply holding the market?
  2. Did it achieve a better return per unit of drawdown?

Both are reported because they can disagree. A strategy can beat the market
on raw return while taking considerably more risk to get there, and reporting
only the first would hide that.
"""

import numpy as np
import pandas as pd


class ComparisonError(ValueError):
    """Raised when a benchmark or comparison cannot be computed."""


def max_drawdown(equity, starting_balance):
    """
    Largest peak to trough decline, as a fraction of the running peak.

    Drawdown is measured from the highest balance reached so far, not from
    the opening balance, because that is the loss actually experienced by
    someone holding the position: the distance fallen from the best point
    they had already seen.

    Works on a single equity curve or on a 2D array of many curves at once,
    in which case one drawdown per row is returned.

    Args:
        equity: 1D curve or 2D array of curves (one per row).
        starting_balance: balance before the first trade. Included as the
            initial peak so that an immediate loss is counted properly.

    Returns:
        A float, or a 1D array of floats, in the range 0 to 1.
    """
    equity = np.asarray(equity, dtype=float)

    if equity.size == 0:
        raise ComparisonError("Cannot compute drawdown of an empty curve")

    if starting_balance <= 0:
        raise ComparisonError("Starting balance must be positive")

    # Prepend the opening balance. Without it, a curve that only ever falls
    # would report its first value as the peak and understate the loss.
    opening = np.full(equity.shape[:-1] + (1,), starting_balance)
    equity = np.concatenate([opening, equity], axis=-1)

    running_peak = np.maximum.accumulate(equity, axis=-1)
    drawdowns = (running_peak - equity) / running_peak

    return drawdowns.max(axis=-1)


def total_return(equity, starting_balance):
    """
    Overall return as a fraction of the opening balance.

    Returns a float for a single curve, or one value per row for a 2D array.
    """
    equity = np.asarray(equity, dtype=float)
    final_balance = equity[..., -1]
    return (final_balance - starting_balance) / starting_balance


def return_to_drawdown(total_return_value, max_drawdown_value):
    """
    Return earned per unit of drawdown suffered.

    A strategy returning 50% with a 10% drawdown scores 5.0; the same return
    with a 50% drawdown scores 1.0. This is what exposes a strategy that beats
    the market only by taking far more risk.

    A drawdown of zero would divide by zero. It is treated as infinite skill
    only when the return is positive, and as zero otherwise, which avoids
    rewarding a flat strategy that simply never lost anything.
    """
    total_return_value = np.asarray(total_return_value, dtype=float)
    max_drawdown_value = np.asarray(max_drawdown_value, dtype=float)

    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(
            max_drawdown_value > 0,
            total_return_value / max_drawdown_value,
            np.where(total_return_value > 0, np.inf, 0.0),
        )

    return ratio


def buy_and_hold_curve(prices, starting_balance):
    """
    Equity curve for putting the whole opening balance into the market at the
    start and holding to the end.

    The position is unleveraged and never changes size, which is the point:
    it is the simplest possible alternative to running a strategy at all.

    Args:
        prices: sequence of prices in chronological order.
        starting_balance: amount invested at the first price.
    """
    prices = np.asarray(prices, dtype=float)

    if prices.size < 2:
        raise ComparisonError("Need at least two prices to build a benchmark")

    if not np.isfinite(prices).all() or (prices <= 0).any():
        raise ComparisonError("Prices must all be positive and finite")

    # Scaling by the ratio to the first price converts the price series into
    # the value of a fixed holding bought at the start.
    return starting_balance * (prices / prices[0])


def prices_from_trades(trades):
    """
    Recover an approximate price series from the trade log.

    The log records the market price whenever a trade opened or closed, so
    interleaving those in time order gives a sampled view of the market.

    This is a fallback for when no proper price history is supplied. It is an
    approximation, and specifically an optimistic one for the benchmark's
    risk: because it only observes the market at trade times, any decline that
    began and ended between two trades is invisible, so the benchmark's
    drawdown will be understated. Where accuracy matters, supply a real price
    series instead.
    """
    opens = trades[["Open time", "Open price"]].rename(
        columns={"Open time": "time", "Open price": "price"}
    )
    closes = trades[["Close time", "Close price"]].rename(
        columns={"Close time": "time", "Close price": "price"}
    )

    combined = pd.concat([opens, closes], ignore_index=True).sort_values("time")

    return combined["price"].to_numpy(dtype=float)


def compare_to_benchmark(paths, benchmark_curve, starting_balance):
    """
    Compare every simulated path against the benchmark on both criteria.

    Args:
        paths: 2D array of simulated equity curves, one per row.
        benchmark_curve: the buy and hold equity curve.
        starting_balance: opening balance, shared by both.

    Returns:
        A dictionary containing, for the benchmark, its return, drawdown and
        ratio; for the simulations, the same three as arrays; and two boolean
        arrays recording whether each path beat the benchmark on return and
        on return per unit of drawdown.
    """
    paths = np.asarray(paths, dtype=float)

    if paths.ndim != 2:
        raise ComparisonError("Simulated paths must be a 2D array")

    benchmark_return = float(total_return(benchmark_curve, starting_balance))
    benchmark_drawdown = float(max_drawdown(benchmark_curve, starting_balance))
    benchmark_ratio = float(return_to_drawdown(benchmark_return, benchmark_drawdown))

    path_returns = total_return(paths, starting_balance)
    path_drawdowns = max_drawdown(paths, starting_balance)
    path_ratios = return_to_drawdown(path_returns, path_drawdowns)

    return {
        "benchmark_return": benchmark_return,
        "benchmark_drawdown": benchmark_drawdown,
        "benchmark_ratio": benchmark_ratio,
        "path_returns": path_returns,
        "path_drawdowns": path_drawdowns,
        "path_ratios": path_ratios,
        "beats_on_return": path_returns > benchmark_return,
        "beats_on_ratio": path_ratios > benchmark_ratio,
    }


# Approximate number of trading days in a year, used to annualise.
TRADING_DAYS_PER_YEAR = 252


def sharpe_ratio(returns, periods_per_year=None, risk_free_rate=0.0):
    """
    Mean return divided by the standard deviation of returns.

    Where the return-to-drawdown ratio penalises the single worst decline,
    the Sharpe ratio penalises volatility throughout. The two can disagree: a
    strategy with consistently choppy returns but no severe decline scores
    poorly here and well there, so both are worth reporting.

    Annualisation is applied by scaling with the square root of the number of
    periods per year. Lo (2002) shows that this scaling is only valid when
    returns are independent and identically distributed, an assumption that
    real trade sequences frequently violate, so an annualised figure computed
    this way should be read as an approximation rather than a precise
    quantity. That caveat is the reason a distribution of outcomes is
    reported alongside any single summary statistic.

    Args:
        returns: per-trade or per-period fractional returns. A 2D array is
            treated as one series per row.
        periods_per_year: supply to annualise; leave as None for the
            per-period figure.
        risk_free_rate: return available without taking risk, expressed per
            period. Defaults to zero.

    Returns:
        A float, or one value per row for a 2D input.
    """
    returns = np.asarray(returns, dtype=float)

    if returns.shape[-1] < 2:
        raise ComparisonError("Need at least two returns to compute a Sharpe ratio")

    excess = returns - risk_free_rate
    mean = excess.mean(axis=-1)
    # ddof=1 gives the sample standard deviation, appropriate because these
    # returns are a sample of the strategy's behaviour rather than the whole
    # population of trades it could ever produce.
    deviation = excess.std(axis=-1, ddof=1)

    # Zero volatility is handled the same way as zero drawdown above: a
    # positive return earned with no variability at all is unbounded, while a
    # flat or losing series earns nothing. Consistency between the two
    # risk-adjusted measures matters here, since they are reported together.
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(
            deviation > 0,
            mean / deviation,
            np.where(mean > 0, np.inf, 0.0),
        )

    if periods_per_year is not None:
        ratio = ratio * np.sqrt(periods_per_year)

    return ratio


def returns_from_curve(equity, starting_balance):
    """
    Recover the per-step fractional returns from an equity curve.

    Lets the risk-adjusted measures be computed for simulated paths, which
    are produced as balances rather than as returns.
    """
    equity = np.asarray(equity, dtype=float)
    opening = np.full(equity.shape[:-1] + (1,), starting_balance)
    full = np.concatenate([opening, equity], axis=-1)

    if (full[..., :-1] <= 0).any():
        raise ComparisonError("Equity curve reached zero or below")

    return np.diff(full, axis=-1) / full[..., :-1]
