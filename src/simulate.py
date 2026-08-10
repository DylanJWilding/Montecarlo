"""
simulate.py

The Monte Carlo engine. Takes the profit and loss of each trade and produces
a large set of alternative equity paths by resampling that sequence.

The purpose is to separate a strategy's genuine edge from the luck of the
particular order its trades happened to occur in. A single backtest is one
realisation; this module generates many (see FR5 to FR8).
"""

import numpy as np


# Resampling modes.
#
# WITH_REPLACEMENT draws trades at random from the original set, so a trade
# can appear several times in one path or not at all. The total profit
# therefore differs between paths, which produces a distribution of final
# outcomes. This is the mode needed to ask whether the strategy is likely to
# beat a benchmark, so it is the default.
#
# WITHOUT_REPLACEMENT reshuffles the original trades into a new order. Every
# path contains exactly the same trades, so every path ends at exactly the
# same balance. It answers a narrower question: given these trades, how much
# did the order affect the drawdown along the way.
WITH_REPLACEMENT = "with_replacement"
WITHOUT_REPLACEMENT = "without_replacement"
VALID_MODES = (WITH_REPLACEMENT, WITHOUT_REPLACEMENT)

DEFAULT_SIMULATIONS = 10_000


class SimulationError(ValueError):
    """Raised when a simulation cannot be run with the given inputs."""


def run_simulation(
    profits,
    starting_balance,
    n_simulations=DEFAULT_SIMULATIONS,
    mode=WITH_REPLACEMENT,
    seed=None,
    compound=False,
):
    """
    Generate simulated equity paths by resampling the trade sequence.

    Args:
        profits: sequence of per-trade profit and loss values.
        starting_balance: account balance before the first trade.
        n_simulations: number of alternative paths to generate.
        mode: WITH_REPLACEMENT or WITHOUT_REPLACEMENT.
        seed: optional integer. Supplying one makes the run reproducible.
        compound: whether the values supplied are fractional returns to be
            compounded (True) or fixed monetary amounts to be added (False).
            Use True whenever the strategy sizes its positions as a
            percentage of the account, because in that case a trade's
            monetary value depends on the balance at the time it was taken
            and cannot meaningfully be transplanted to a different point in
            a different sequence. See the note on _accumulate below.

    Returns:
        A 2D numpy array of shape (n_simulations, n_trades) where each row is
        one simulated equity path and each column is the account balance
        after that many trades.

    Raises:
        SimulationError: if the inputs are empty, non-numeric, or invalid.
    """
    profits = _validate_profits(profits)
    _validate_simulation_count(n_simulations)
    _validate_mode(mode)

    # A seeded generator gives reproducible output, which the requirements
    # need (FR7) and which also makes the results testable. Passing None
    # produces a fresh random stream each run.
    rng = np.random.default_rng(seed)

    resampled = _resample(profits, n_simulations, mode, rng)

    return _accumulate(resampled, starting_balance, compound)


def _accumulate(resampled, starting_balance, compound):
    """
    Turn resampled trade outcomes into equity paths.

    Two accumulation methods are needed because the two position sizing
    regimes produce different kinds of number.

    Under fixed monetary risk every trade stakes roughly the same amount
    regardless of the balance, so a trade's result is a sum of money that
    means the same thing anywhere in the sequence. Those are added.

    Under percentage risk the amount staked scales with the account, so a
    given trade's monetary result is only meaningful relative to the balance
    at the time. Moving a large late-sequence loss to the start of a path
    would apply a loss that could never have occurred at that balance. The
    trade is therefore expressed as a fraction of the account and compounded,
    which preserves its meaning wherever it lands.

    Both are computed across the whole array at once rather than per
    simulation, for the performance reasons noted above (NFR1).
    """
    if compound:
        return starting_balance * np.cumprod(1.0 + resampled, axis=1)

    return starting_balance + np.cumsum(resampled, axis=1)


def to_returns(profits, balances):
    """
    Express each trade as a fraction of the account balance before it.

    Needed to simulate a strategy that sizes positions as a percentage of
    equity. The balance before a trade is recovered by subtracting that
    trade's result from the balance recorded against it.
    """
    profits = np.asarray(profits, dtype=float)
    balances = np.asarray(balances, dtype=float)

    if profits.shape != balances.shape:
        raise SimulationError("Profits and balances must be the same length")

    equity_before = balances - profits

    if (equity_before <= 0).any():
        raise SimulationError("Account balance reached zero or below")

    return profits / equity_before


def equity_curve(profits, starting_balance, compound=False):
    """
    Build the actual historical equity curve, with no resampling.

    This is the single path the strategy really produced. It is needed so the
    real result can be positioned within the simulated distribution (FR14).
    """
    profits = _validate_profits(profits)

    if compound:
        return starting_balance * np.cumprod(1.0 + profits)

    return starting_balance + np.cumsum(profits)


def _resample(profits, n_simulations, mode, rng):
    """
    Produce the resampled trade sequences for every simulation.

    Both branches build the entire (n_simulations, n_trades) array in one
    operation rather than simulation by simulation.
    """
    n_trades = len(profits)

    if mode == WITH_REPLACEMENT:
        # Draw random positions into the original trade array. Because the
        # same position can be drawn more than once, totals vary between
        # paths.
        indices = rng.integers(0, n_trades, size=(n_simulations, n_trades))
    else:
        # Generate a separate random permutation for each simulation.
        # Sorting a row of random numbers yields the order that would sort
        # them, which is a uniformly random permutation, and argsort applies
        # this across every row at once.
        indices = rng.random((n_simulations, n_trades)).argsort(axis=1)

    return profits[indices]


def _validate_profits(profits):
    """Convert the trade profits to a numeric array and reject bad input."""
    profits = np.asarray(profits, dtype=float)

    if profits.ndim != 1:
        raise SimulationError("Trade profits must be a one dimensional sequence")

    if profits.size == 0:
        raise SimulationError("Cannot simulate: no trades supplied")

    if not np.isfinite(profits).all():
        raise SimulationError("Trade profits contain missing or infinite values")

    return profits


def _validate_simulation_count(n_simulations):
    if not isinstance(n_simulations, (int, np.integer)) or isinstance(n_simulations, bool):
        raise SimulationError("Number of simulations must be an integer")

    if n_simulations < 1:
        raise SimulationError(
            f"Number of simulations must be at least 1, got {n_simulations}"
        )


def _validate_mode(mode):
    if mode not in VALID_MODES:
        raise SimulationError(
            f"Unknown resampling mode '{mode}'. Valid modes: {list(VALID_MODES)}"
        )
