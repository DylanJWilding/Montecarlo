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
):
    """
    Generate simulated equity paths by resampling the trade sequence.

    Args:
        profits: sequence of per-trade profit and loss values.
        starting_balance: account balance before the first trade.
        n_simulations: number of alternative paths to generate.
        mode: WITH_REPLACEMENT or WITHOUT_REPLACEMENT.
        seed: optional integer. Supplying one makes the run reproducible.

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

    # Each path is the running total of its own resampled trades, offset by
    # the opening balance. Doing this with a cumulative sum across the whole
    # array at once is far faster than looping over each simulation, which
    # matters because this is the performance bottleneck (NFR1).
    return starting_balance + np.cumsum(resampled, axis=1)


def equity_curve(profits, starting_balance):
    """
    Build the actual historical equity curve, with no resampling.

    This is the single path the strategy really produced. It is needed so the
    real result can be positioned within the simulated distribution (FR14).
    """
    profits = _validate_profits(profits)
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
