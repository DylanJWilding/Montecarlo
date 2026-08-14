"""
convergence.py

Evaluation: how many simulations are enough?

The tool reports probabilities estimated from a finite number of simulated
paths. Too few and the figure moves noticeably between runs; too many and the
tool stops feeling interactive. The default has to be chosen somewhere, and
the point of this script is to choose it from evidence rather than by picking
a round number.

Method: run the same input at increasing simulation counts, repeating each
count several times with different random seeds, and measure how much the
reported probability varies across those repeats. The spread of results at a
given count is a direct measure of how much the answer would wobble if the
user re-ran the tool.

Run from the project root:  python evaluation/convergence.py
"""

import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from ingest import load_trades
from simulate import run_simulation, equity_curve, to_returns
from compare import buy_and_hold_curve, prices_from_trades, compare_to_benchmark


DATA = ROOT / "tests" / "fixtures" / "sample_trades.csv"
OUTPUT = ROOT / "evaluation" / "results"

SIMULATION_COUNTS = [100, 250, 500, 1_000, 2_500, 5_000, 10_000, 25_000, 50_000]
REPEATS = 12          # independent seeds per count
CONFIDENCE_TARGET = 0.005   # half a percentage point of spread


def load_inputs():
    """Load the trade data and build everything the simulation needs."""
    trades, starting_balance = load_trades(DATA)

    raw = pd.read_csv(DATA, sep=";", quotechar='"')
    for column in ("Open time", "Close time"):
        raw[column] = pd.to_datetime(raw[column], format="%Y.%m.%d %H:%M:%S")

    returns = to_returns(
        trades["Profit/Loss"].to_numpy(), trades["Balance"].to_numpy()
    )
    benchmark = buy_and_hold_curve(prices_from_trades(raw), starting_balance)

    span_years = (
        trades["Close time"].iloc[-1] - trades["Open time"].iloc[0]
    ).days / 365.25

    return {
        "returns": returns,
        "starting_balance": starting_balance,
        "benchmark": benchmark,
        "periods_per_year": len(trades) / span_years,
    }


def measure_convergence(inputs):
    """
    Record the reported probabilities at each simulation count, across
    several seeds, so the variation between repeats can be measured.
    """
    records = []

    for count in SIMULATION_COUNTS:
        for repeat in range(REPEATS):
            started = time.perf_counter()

            paths = run_simulation(
                inputs["returns"],
                inputs["starting_balance"],
                n_simulations=count,
                seed=1000 + repeat,
                compound=True,
            )
            comparison = compare_to_benchmark(
                paths,
                inputs["benchmark"],
                inputs["starting_balance"],
                periods_per_year=inputs["periods_per_year"],
            )

            records.append({
                "n_simulations": count,
                "seed": 1000 + repeat,
                "p_return": comparison["beats_on_return"].mean(),
                "p_ratio": comparison["beats_on_ratio"].mean(),
                "p_sharpe": comparison["beats_on_sharpe"].mean(),
                "seconds": time.perf_counter() - started,
            })

        print(f"  completed {count:,} simulations x {REPEATS} seeds")

    return pd.DataFrame(records)


def summarise(results):
    """
    For each simulation count, how much did the answer vary between seeds?

    The range (highest minus lowest across repeats) is reported alongside the
    standard deviation because it is the figure a user would actually notice:
    it is the worst disagreement they could see between two runs.
    """
    summary = results.groupby("n_simulations").agg(
        p_return_mean=("p_return", "mean"),
        p_return_std=("p_return", "std"),
        p_return_range=("p_return", lambda values: values.max() - values.min()),
        p_ratio_mean=("p_ratio", "mean"),
        p_ratio_std=("p_ratio", "std"),
        p_ratio_range=("p_ratio", lambda values: values.max() - values.min()),
        p_sharpe_mean=("p_sharpe", "mean"),
        p_sharpe_std=("p_sharpe", "std"),
        p_sharpe_range=("p_sharpe", lambda values: values.max() - values.min()),
        mean_seconds=("seconds", "mean"),
    ).reset_index()

    return summary


def recommend(summary):
    """
    Smallest count at which every reported probability is stable to within
    the target spread. Stated as a rule applied to the data rather than a
    judgement, so the recommendation can be reproduced.
    """
    stable = summary[
        (summary["p_return_range"] <= CONFIDENCE_TARGET)
        & (summary["p_ratio_range"] <= CONFIDENCE_TARGET)
        & (summary["p_sharpe_range"] <= CONFIDENCE_TARGET)
    ]

    if stable.empty:
        return None

    return int(stable["n_simulations"].iloc[0])


def plot_convergence(results, summary, recommended):
    """Two panels: the spread of estimates, and how fast that spread falls."""
    figure, (top, bottom) = plt.subplots(2, 1, figsize=(9, 8), sharex=True)

    # Panel 1: every individual run, showing the scatter narrowing.
    for label, column, colour in (
        ("Beats on return", "p_return", "#b3541e"),
        ("Beats on return/drawdown", "p_ratio", "#1f3d63"),
        ("Beats on Sharpe", "p_sharpe", "#2e7d5b"),
    ):
        top.scatter(
            results["n_simulations"], results[column],
            alpha=0.35, s=18, color=colour, label=label,
        )
        means = summary.set_index("n_simulations")[f"{column}_mean"]
        top.plot(means.index, means.values, color=colour, linewidth=1.2)

    top.set_xscale("log")
    top.set_ylabel("Reported probability")
    top.set_title("Estimates from repeated runs at each simulation count")
    top.legend(frameon=False, fontsize=9)
    top.spines[["top", "right"]].set_visible(False)

    # Panel 2: the spread itself, which is what the decision rests on.
    for label, column, colour in (
        ("Beats on return", "p_return_range", "#b3541e"),
        ("Beats on return/drawdown", "p_ratio_range", "#1f3d63"),
        ("Beats on Sharpe", "p_sharpe_range", "#2e7d5b"),
    ):
        bottom.plot(
            summary["n_simulations"], summary[column],
            marker="o", markersize=4, color=colour, label=label,
        )

    bottom.axhline(
        CONFIDENCE_TARGET, color="#999999", linestyle="--", linewidth=1,
        label=f"Target spread ({CONFIDENCE_TARGET:.1%})",
    )
    if recommended is not None:
        bottom.axvline(
            recommended, color="#333333", linestyle=":", linewidth=1.2,
            label=f"Chosen default ({recommended:,})",
        )

    bottom.set_xscale("log")
    bottom.set_yscale("log")
    bottom.set_xlabel("Number of simulations")
    bottom.set_ylabel("Spread across repeats")
    bottom.set_title("How much the answer moves between runs")
    bottom.legend(frameon=False, fontsize=9)
    bottom.spines[["top", "right"]].set_visible(False)

    figure.tight_layout()
    figure.savefig(OUTPUT / "convergence.png", dpi=150)
    print(f"  chart written to {OUTPUT / 'convergence.png'}")


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)

    print("Convergence analysis")
    print(f"  {len(SIMULATION_COUNTS)} counts x {REPEATS} seeds")

    inputs = load_inputs()
    results = measure_convergence(inputs)
    summary = summarise(results)
    recommended = recommend(summary)

    results.to_csv(OUTPUT / "convergence_raw.csv", index=False)
    summary.to_csv(OUTPUT / "convergence_summary.csv", index=False)
    plot_convergence(results, summary, recommended)

    print("\n  n_sims   spread(return)  spread(ret/DD)  spread(Sharpe)   time")
    for _, row in summary.iterrows():
        print(
            f"  {int(row['n_simulations']):>7,}  "
            f"{row['p_return_range']:>13.2%}  "
            f"{row['p_ratio_range']:>13.2%}  "
            f"{row['p_sharpe_range']:>13.2%}  "
            f"{row['mean_seconds']:>6.2f}s"
        )

    if recommended is None:
        print(
            f"\n  No tested count held every estimate within "
            f"{CONFIDENCE_TARGET:.1%}. Report the trend and choose on the "
            f"time/stability trade-off."
        )
    else:
        seconds = float(
            summary.loc[
                summary["n_simulations"] == recommended, "mean_seconds"
            ].iloc[0]
        )
        print(
            f"\n  Recommended default: {recommended:,} simulations "
            f"({seconds:.2f}s per run)"
        )


if __name__ == "__main__":
    main()
