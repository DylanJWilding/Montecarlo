"""
test_simulate.py

Unit tests for the Monte Carlo engine.

Testing a random process needs care: assertions have to hold for any valid
random draw, not just the one that happened. The approach here is to test
the properties that must always be true (output shape, reproducibility under
a seed, and the mathematical invariants of each resampling mode) rather than
comparing against specific random numbers.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from simulate import (
    run_simulation,
    equity_curve,
    SimulationError,
    WITH_REPLACEMENT,
    WITHOUT_REPLACEMENT,
)


# A small, hand checkable set of trades used throughout.
PROFITS = np.array([100.0, -50.0, 200.0, -25.0, 75.0])
STARTING_BALANCE = 1000.0


# ---------------------------------------------------------------------------
# Output shape and basic behaviour
# ---------------------------------------------------------------------------

def test_output_has_one_row_per_simulation_and_one_column_per_trade():
    paths = run_simulation(PROFITS, STARTING_BALANCE, n_simulations=50, seed=1)
    assert paths.shape == (50, len(PROFITS))


def test_first_column_is_starting_balance_plus_one_trade():
    """
    Every path begins with the opening balance adjusted by whichever trade was
    drawn first, so the first column can only take one of the original trade
    values as its offset.
    """
    paths = run_simulation(PROFITS, STARTING_BALANCE, n_simulations=100, seed=1)
    first_moves = paths[:, 0] - STARTING_BALANCE
    assert np.isin(first_moves, PROFITS).all()


def test_both_modes_run():
    for mode in (WITH_REPLACEMENT, WITHOUT_REPLACEMENT):
        paths = run_simulation(
            PROFITS, STARTING_BALANCE, n_simulations=10, mode=mode, seed=1
        )
        assert paths.shape == (10, len(PROFITS))


# ---------------------------------------------------------------------------
# Reproducibility (FR7)
# ---------------------------------------------------------------------------

def test_same_seed_produces_identical_results():
    first = run_simulation(PROFITS, STARTING_BALANCE, n_simulations=100, seed=42)
    second = run_simulation(PROFITS, STARTING_BALANCE, n_simulations=100, seed=42)
    assert np.array_equal(first, second)


def test_different_seeds_produce_different_results():
    first = run_simulation(PROFITS, STARTING_BALANCE, n_simulations=100, seed=1)
    second = run_simulation(PROFITS, STARTING_BALANCE, n_simulations=100, seed=2)
    assert not np.array_equal(first, second)


# ---------------------------------------------------------------------------
# Invariants of each resampling mode
#
# These are the tests that confirm the two modes are doing conceptually
# different things, which is the reasoning behind offering both.
# ---------------------------------------------------------------------------

def test_without_replacement_always_ends_at_the_same_balance():
    """
    Reshuffling the same trades cannot change their total, so every path must
    finish at the same final balance as the real strategy. Only the route
    there differs.
    """
    paths = run_simulation(
        PROFITS, STARTING_BALANCE, n_simulations=200,
        mode=WITHOUT_REPLACEMENT, seed=1,
    )
    expected_final = STARTING_BALANCE + PROFITS.sum()
    assert np.allclose(paths[:, -1], expected_final)


def test_without_replacement_uses_each_trade_exactly_once():
    paths = run_simulation(
        PROFITS, STARTING_BALANCE, n_simulations=50,
        mode=WITHOUT_REPLACEMENT, seed=1,
    )
    # Recover the individual trades from the running totals.
    trades_per_path = np.diff(paths, prepend=STARTING_BALANCE, axis=1)
    for path_trades in trades_per_path:
        assert np.allclose(np.sort(path_trades), np.sort(PROFITS))


def test_with_replacement_produces_varying_final_balances():
    """
    Drawing trades with replacement means totals differ between paths, which
    is what creates the distribution of outcomes needed to compare against a
    benchmark.
    """
    paths = run_simulation(
        PROFITS, STARTING_BALANCE, n_simulations=500,
        mode=WITH_REPLACEMENT, seed=1,
    )
    assert len(np.unique(paths[:, -1])) > 1


def test_with_replacement_mean_final_balance_approaches_the_real_total():
    """
    Resampling with replacement is unbiased: over many simulations the average
    final balance should converge on the strategy's actual final balance.
    This is a correctness check against a value that can be derived rather
    than observed (NFR4).
    """
    paths = run_simulation(
        PROFITS, STARTING_BALANCE, n_simulations=20_000,
        mode=WITH_REPLACEMENT, seed=7,
    )
    expected = STARTING_BALANCE + PROFITS.sum()
    assert paths[:, -1].mean() == pytest.approx(expected, rel=0.05)


# ---------------------------------------------------------------------------
# The real (unresampled) equity curve
# ---------------------------------------------------------------------------

def test_equity_curve_matches_hand_calculation():
    curve = equity_curve(PROFITS, STARTING_BALANCE)
    expected = np.array([1100.0, 1050.0, 1250.0, 1225.0, 1300.0])
    assert np.allclose(curve, expected)


# ---------------------------------------------------------------------------
# Invalid input
# ---------------------------------------------------------------------------

def test_empty_trade_list_raises_error():
    with pytest.raises(SimulationError):
        run_simulation([], STARTING_BALANCE)


def test_zero_simulations_raises_error():
    with pytest.raises(SimulationError):
        run_simulation(PROFITS, STARTING_BALANCE, n_simulations=0)


def test_unknown_mode_raises_error():
    with pytest.raises(SimulationError) as error:
        run_simulation(PROFITS, STARTING_BALANCE, mode="sideways")
    assert "sideways" in str(error.value)


def test_missing_values_raise_error():
    with pytest.raises(SimulationError):
        run_simulation([100.0, np.nan, 50.0], STARTING_BALANCE)
