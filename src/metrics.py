"""
metrics.py

Turns the per-path comparison results into the figures actually reported to
the user (FR12 to FR14).

The comparison module answers "did this particular simulated path beat the
benchmark?" once per path. This module answers "how often, and how confident
can we be in that number?" across all of them, and locates the strategy's
real historical result within the simulated distribution.
"""

import numpy as np

from compare import (
    max_drawdown,
    total_return,
    return_to_drawdown,
    sharpe_ratio,
    returns_from_curve,
)


class MetricsError(ValueError):
    """Raised when summary figures cannot be computed."""


# Percentiles reported for every distribution. The 5th and 95th bracket the
# bulk of outcomes without being dominated by the extreme tails, and the
# median is preferred to the mean because compounded return distributions are
# skewed, which drags the mean away from the typical outcome.
REPORTED_PERCENTILES = (5, 25, 50, 75, 95)


def summarise_distribution(values, percentiles=REPORTED_PERCENTILES):
    """
    Describe a set of simulated outcomes.

    Infinite values can arise legitimately from the risk-adjusted measures
    when a path happened to suffer no drawdown at all, so they are excluded
    from the summary and counted separately rather than being allowed to make
    the mean meaningless.
    """
    values = np.asarray(values, dtype=float)

    if values.size == 0:
        raise MetricsError("Cannot summarise an empty set of results")

    finite = values[np.isfinite(values)]

    if finite.size == 0:
        raise MetricsError("No finite values to summarise")

    summary = {
        "mean": float(finite.mean()),
        "median": float(np.median(finite)),
        "std": float(finite.std(ddof=1)) if finite.size > 1 else 0.0,
        "min": float(finite.min()),
        "max": float(finite.max()),
        "excluded_non_finite": int(values.size - finite.size),
    }

    for percentile in percentiles:
        summary[f"p{percentile}"] = float(np.percentile(finite, percentile))

    return summary


def probability_with_interval(successes, confidence=0.95):
    """
    Proportion of simulations that met a condition, with a confidence interval.

    The interval is the point that is easy to overlook. A reported probability
    of 86 per cent is an estimate produced from a finite number of simulations,
    not an exact quantity, and it would differ slightly if the simulation were
    re-run with a different random seed. The interval states how much of that
    wobble to expect, which is what makes the number honest.

    A normal approximation to the binomial is used. It is reliable here
    because the number of simulations is large, but it degrades when the
    proportion sits very close to zero or one, so the interval is clipped to
    remain within the valid range.

    Args:
        successes: boolean array, one entry per simulation.
        confidence: confidence level, defaulting to 95 per cent.

    Returns:
        Dictionary with the probability, the interval bounds, the standard
        error, and the number of simulations behind the estimate.
    """
    successes = np.asarray(successes, dtype=bool)

    if successes.size == 0:
        raise MetricsError("Cannot compute a probability from no simulations")

    n = successes.size
    probability = float(successes.mean())

    # z score for the requested two-sided confidence level. Standard values
    # are used rather than pulling in a statistics dependency for one number.
    z_scores = {0.90: 1.645, 0.95: 1.960, 0.99: 2.576}
    z = z_scores.get(round(confidence, 2))
    if z is None:
        raise MetricsError(
            f"Unsupported confidence level {confidence}. "
            f"Supported: {sorted(z_scores)}"
        )

    standard_error = np.sqrt(probability * (1 - probability) / n)
    margin = z * standard_error

    return {
        "probability": probability,
        "lower": float(max(0.0, probability - margin)),
        "upper": float(min(1.0, probability + margin)),
        "standard_error": float(standard_error),
        "n_simulations": int(n),
        "confidence": confidence,
    }


def percentile_of_actual(actual_value, simulated_values):
    """
    Where the real historical result sits within the simulated distribution.

    This is the figure that directly addresses the project's central question.
    A real backtest sitting at the 95th percentile of what the same trades
    could have produced in a different order suggests the historical result
    owed a good deal to a favourable sequence. One sitting near the middle
    suggests it was typical of the strategy rather than lucky.
    """
    simulated_values = np.asarray(simulated_values, dtype=float)
    finite = simulated_values[np.isfinite(simulated_values)]

    if finite.size == 0:
        raise MetricsError("No finite simulated values to compare against")

    return float((finite < actual_value).mean() * 100)


def build_report(
    comparison,
    actual_curve,
    starting_balance,
    periods_per_year=None,
    confidence=0.95,
):
    """
    Assemble the complete set of reported figures.

    Args:
        comparison: the dictionary returned by compare_to_benchmark.
        actual_curve: the strategy's real historical equity curve.
        starting_balance: opening balance.
        periods_per_year: supply to annualise the Sharpe ratios.
        confidence: confidence level for the probability intervals.

    Returns:
        A nested dictionary holding the benchmark figures, the strategy's real
        figures, summaries of each simulated distribution, the probability of
        beating the benchmark on each criterion, and the percentile position
        of the real result within the simulations.
    """
    actual_curve = np.asarray(actual_curve, dtype=float)

    actual_return = float(total_return(actual_curve, starting_balance))
    actual_drawdown = float(max_drawdown(actual_curve, starting_balance))
    actual_ratio = float(return_to_drawdown(actual_return, actual_drawdown))
    actual_sharpe = float(
        sharpe_ratio(
            returns_from_curve(actual_curve, starting_balance), periods_per_year
        )
    )

    path_sharpes = comparison.get("path_sharpes")

    report = {
        "benchmark": {
            "return": comparison["benchmark_return"],
            "max_drawdown": comparison["benchmark_drawdown"],
            "return_to_drawdown": comparison["benchmark_ratio"],
        },
        "actual": {
            "return": actual_return,
            "max_drawdown": actual_drawdown,
            "return_to_drawdown": actual_ratio,
            "sharpe": actual_sharpe,
        },
        "simulated": {
            "return": summarise_distribution(comparison["path_returns"]),
            "max_drawdown": summarise_distribution(comparison["path_drawdowns"]),
            "return_to_drawdown": summarise_distribution(comparison["path_ratios"]),
        },
        "probability_of_beating_benchmark": {
            "on_return": probability_with_interval(
                comparison["beats_on_return"], confidence
            ),
            "on_return_to_drawdown": probability_with_interval(
                comparison["beats_on_ratio"], confidence
            ),
        },
        "actual_result_percentile": {
            "return": percentile_of_actual(
                actual_return, comparison["path_returns"]
            ),
            "max_drawdown": percentile_of_actual(
                actual_drawdown, comparison["path_drawdowns"]
            ),
        },
    }

    if path_sharpes is not None:
        report["simulated"]["sharpe"] = summarise_distribution(path_sharpes)
        report["actual_result_percentile"]["sharpe"] = percentile_of_actual(
            actual_sharpe, path_sharpes
        )
        if "benchmark_sharpe" in comparison:
            report["benchmark"]["sharpe"] = comparison["benchmark_sharpe"]
            report["probability_of_beating_benchmark"]["on_sharpe"] = (
                probability_with_interval(
                    np.asarray(path_sharpes) > comparison["benchmark_sharpe"],
                    confidence,
                )
            )

    return report


def report_to_rows(report):
    """
    Flatten the report into label and value pairs for display or export.

    Keeping the flattening separate from the calculation means the same
    figures can be shown on screen and written to a file without either
    format dictating how they are computed.
    """
    rows = []
    for section, contents in report.items():
        if isinstance(contents, dict):
            for key, value in contents.items():
                if isinstance(value, dict):
                    for inner_key, inner_value in value.items():
                        rows.append((f"{section}.{key}.{inner_key}", inner_value))
                else:
                    rows.append((f"{section}.{key}", value))
        else:
            rows.append((section, contents))
    return rows
