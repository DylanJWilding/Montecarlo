"""
test_metrics.py

Unit tests for the reporting and aggregation layer.

The figures produced here are the ones the user actually reads, so the tests
concentrate on the properties that would mislead someone if they were wrong:
that probabilities match a hand count, that confidence intervals behave
sensibly as the number of simulations changes, and that infinite values from
the risk-adjusted measures do not corrupt a summary.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from metrics import (
    summarise_distribution,
    probability_with_interval,
    percentile_of_actual,
    report_to_rows,
    MetricsError,
)


# ---------------------------------------------------------------------------
# Summarising a distribution
# ---------------------------------------------------------------------------

def test_summary_reports_median_and_percentiles():
    values = np.arange(1, 101)          # 1 to 100
    summary = summarise_distribution(values)
    assert summary["median"] == pytest.approx(50.5)
    assert summary["p5"] == pytest.approx(np.percentile(values, 5))
    assert summary["p95"] == pytest.approx(np.percentile(values, 95))


def test_summary_excludes_infinite_values_and_counts_them():
    """
    An infinite risk-adjusted score is legitimate: it means a path suffered no
    drawdown at all. Including it would make the mean infinite and destroy the
    summary, so such values are set aside and reported separately.
    """
    values = np.array([1.0, 2.0, 3.0, np.inf])
    summary = summarise_distribution(values)
    assert summary["mean"] == pytest.approx(2.0)
    assert summary["excluded_non_finite"] == 1


def test_summary_of_empty_input_raises_error():
    with pytest.raises(MetricsError):
        summarise_distribution(np.array([]))


def test_summary_of_all_infinite_values_raises_error():
    with pytest.raises(MetricsError):
        summarise_distribution(np.array([np.inf, np.inf]))


# ---------------------------------------------------------------------------
# Probabilities and confidence intervals
# ---------------------------------------------------------------------------

def test_probability_matches_a_hand_count():
    successes = np.array([True, True, True, False])
    assert probability_with_interval(successes)["probability"] == pytest.approx(0.75)


def test_interval_brackets_the_estimate():
    result = probability_with_interval(np.random.default_rng(1).random(1000) < 0.6)
    assert result["lower"] < result["probability"] < result["upper"]


def test_more_simulations_narrow_the_interval():
    """
    The property that makes the interval worth reporting: it is a statement
    about how much the estimate would move if the simulation were re-run, and
    that uncertainty falls as the number of simulations rises.
    """
    rng = np.random.default_rng(1)
    small = probability_with_interval(rng.random(100) < 0.5)
    large = probability_with_interval(rng.random(10_000) < 0.5)
    small_width = small["upper"] - small["lower"]
    large_width = large["upper"] - large["lower"]
    assert large_width < small_width


def test_interval_stays_within_zero_and_one():
    """
    The normal approximation can produce bounds outside the valid range when
    the proportion is extreme, so the result is clipped.
    """
    result = probability_with_interval(np.ones(50, dtype=bool))
    assert result["lower"] >= 0.0
    assert result["upper"] <= 1.0


def test_unsupported_confidence_level_raises_error():
    with pytest.raises(MetricsError):
        probability_with_interval(np.array([True, False]), confidence=0.42)


def test_probability_from_no_simulations_raises_error():
    with pytest.raises(MetricsError):
        probability_with_interval(np.array([], dtype=bool))


# ---------------------------------------------------------------------------
# Locating the real result within the simulated distribution
# ---------------------------------------------------------------------------

def test_actual_at_the_middle_of_the_distribution():
    simulated = np.arange(0, 100)
    assert percentile_of_actual(50, simulated) == pytest.approx(50.0)


def test_actual_above_every_simulation_is_the_top_percentile():
    assert percentile_of_actual(1000, np.arange(0, 100)) == pytest.approx(100.0)


def test_actual_below_every_simulation_is_the_bottom_percentile():
    assert percentile_of_actual(-1, np.arange(0, 100)) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Flattening for display and export
# ---------------------------------------------------------------------------

def test_nested_report_is_flattened_to_labelled_rows():
    report = {"benchmark": {"return": 0.5}, "actual": {"return": 0.7}}
    rows = dict(report_to_rows(report))
    assert rows["benchmark.return"] == 0.5
    assert rows["actual.return"] == 0.7
