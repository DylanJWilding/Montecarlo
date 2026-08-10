"""
test_compare.py

Unit tests for the benchmark and comparison logic.

Drawdown is the easiest thing in this project to get subtly wrong, so most of
these tests use small curves where the correct answer can be worked out by
hand and checked against.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from compare import (
    max_drawdown,
    total_return,
    return_to_drawdown,
    buy_and_hold_curve,
    compare_to_benchmark,
    ComparisonError,
)


START = 1000.0


# ---------------------------------------------------------------------------
# Drawdown
# ---------------------------------------------------------------------------

def test_rising_curve_has_no_drawdown():
    curve = np.array([1100.0, 1200.0, 1300.0])
    assert max_drawdown(curve, START) == pytest.approx(0.0)


def test_drawdown_measured_from_running_peak_not_from_start():
    """
    Rises to 2000 then falls to 1500. Measured from the peak that is a 25%
    drawdown. Measured from the 1000 opening balance it would look like a
    gain, which is why the peak is the correct reference point.
    """
    curve = np.array([1500.0, 2000.0, 1500.0])
    assert max_drawdown(curve, START) == pytest.approx(0.25)


def test_immediate_loss_is_counted_against_starting_balance():
    """
    A curve that only ever falls has no peak of its own, so the opening
    balance has to serve as the initial peak or the loss is missed entirely.
    """
    curve = np.array([900.0, 800.0, 700.0])
    assert max_drawdown(curve, START) == pytest.approx(0.30)


def test_largest_of_several_drawdowns_is_reported():
    # Falls 1200 to 1080 (10%), recovers, then falls 2000 to 1400 (30%).
    curve = np.array([1200.0, 1080.0, 2000.0, 1400.0])
    assert max_drawdown(curve, START) == pytest.approx(0.30)


def test_drawdown_computed_per_row_for_many_curves():
    curves = np.array([
        [1500.0, 2000.0, 1500.0],   # 25%
        [1100.0, 1200.0, 1300.0],   # 0%
        [900.0, 800.0, 700.0],      # 30%
    ])
    result = max_drawdown(curves, START)
    assert result == pytest.approx([0.25, 0.0, 0.30])


def test_empty_curve_raises_error():
    with pytest.raises(ComparisonError):
        max_drawdown(np.array([]), START)


# ---------------------------------------------------------------------------
# Return and ratio
# ---------------------------------------------------------------------------

def test_total_return_is_fraction_of_starting_balance():
    assert total_return(np.array([1100.0, 1500.0]), START) == pytest.approx(0.5)


def test_total_return_can_be_negative():
    assert total_return(np.array([900.0, 700.0]), START) == pytest.approx(-0.3)


def test_ratio_divides_return_by_drawdown():
    assert return_to_drawdown(0.5, 0.1) == pytest.approx(5.0)


def test_ratio_penalises_the_same_return_earned_with_more_drawdown():
    """The property that makes this metric worth reporting alongside return."""
    assert return_to_drawdown(0.5, 0.5) < return_to_drawdown(0.5, 0.1)


def test_zero_drawdown_with_profit_is_infinite():
    assert np.isinf(return_to_drawdown(0.5, 0.0))


def test_zero_drawdown_without_profit_is_not_rewarded():
    """A flat strategy that never lost anything has not demonstrated skill."""
    assert return_to_drawdown(0.0, 0.0) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Buy and hold benchmark
# ---------------------------------------------------------------------------

def test_benchmark_tracks_the_price_series():
    """Price doubling should double the account, since the holding is fixed."""
    curve = buy_and_hold_curve([100.0, 150.0, 200.0], START)
    assert curve == pytest.approx([1000.0, 1500.0, 2000.0])


def test_benchmark_return_matches_price_change():
    curve = buy_and_hold_curve([100.0, 120.0], START)
    assert total_return(curve, START) == pytest.approx(0.2)


def test_benchmark_rejects_negative_prices():
    with pytest.raises(ComparisonError):
        buy_and_hold_curve([100.0, -50.0], START)


def test_benchmark_needs_at_least_two_prices():
    with pytest.raises(ComparisonError):
        buy_and_hold_curve([100.0], START)


# ---------------------------------------------------------------------------
# Comparing simulations against the benchmark
# ---------------------------------------------------------------------------

def test_comparison_flags_which_paths_beat_the_benchmark_on_return():
    benchmark = buy_and_hold_curve([100.0, 110.0], START)   # +10%
    paths = np.array([
        [1000.0, 1200.0],   # +20%, beats it
        [1000.0, 1050.0],   # +5%, does not
    ])
    result = compare_to_benchmark(paths, benchmark, START)
    assert list(result["beats_on_return"]) == [True, False]


def test_a_path_can_win_on_return_but_lose_on_ratio():
    """
    The case that justifies reporting both criteria: a path finishes ahead but
    only by enduring a far deeper decline along the way.
    """
    benchmark = buy_and_hold_curve([100.0, 105.0, 110.0], START)  # +10%, no drawdown
    paths = np.array([[1000.0, 500.0, 1200.0]])  # +20% but a 50% drawdown
    result = compare_to_benchmark(paths, benchmark, START)
    assert result["beats_on_return"][0]
    assert not result["beats_on_ratio"][0]


def test_comparison_returns_one_value_per_path():
    benchmark = buy_and_hold_curve([100.0, 110.0], START)
    paths = np.array([[1000.0, 1200.0], [1000.0, 900.0], [1000.0, 1100.0]])
    result = compare_to_benchmark(paths, benchmark, START)
    assert len(result["path_returns"]) == 3
    assert len(result["path_drawdowns"]) == 3
    assert len(result["path_ratios"]) == 3


def test_comparison_rejects_one_dimensional_input():
    benchmark = buy_and_hold_curve([100.0, 110.0], START)
    with pytest.raises(ComparisonError):
        compare_to_benchmark(np.array([1000.0, 1200.0]), benchmark, START)
