"""
performance.py

Evaluation: how does the tool scale, and where does it stop being practical?

The requirements set an interactive response target (NFR1). This script
measures execution time and peak memory against the number of simulations and
against the length of the trade history, so the target can be shown to be met
and the point at which it fails can be stated honestly rather than left
undiscovered.

It also compares the vectorised implementation against a straightforward loop,
which turns a design decision made during the build into a measured result.

Run from the project root:  python evaluation/performance.py
"""

import sys
import time
import tracemalloc
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from ingest import load_trades
from simulate import run_simulation, to_returns
from compare import buy_and_hold_curve, prices_from_trades, compare_to_benchmark


DATA = ROOT / "tests" / "fixtures" / "sample_trades.csv"
OUTPUT = ROOT / "evaluation" / "results"

SIMULATION_COUNTS = [1_000, 5_000, 10_000, 25_000, 50_000, 100_000]
TRADE_COUNTS = [100, 250, 500, 736, 1_500, 3_000]
INTERACTIVE_TARGET_SECONDS = 10.0
REPEATS = 3


def load_returns():
    trades, starting_balance = load_trades(DATA)
    returns = to_returns(
        trades["Profit/Loss"].to_numpy(), trades["Balance"].to_numpy()
    )
    return returns, starting_balance


def time_run(function, repeats=REPEATS):
    """
    Best of several repeats, with peak memory.

    The minimum is used rather than the mean because a slow repeat reflects
    interference from other work on the machine, not the cost of the
    computation itself. Peak memory is captured on a single run since it does
    not vary between repeats.
    """
    tracemalloc.start()
    function()
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    timings = []
    for _ in range(repeats):
        started = time.perf_counter()
        function()
        timings.append(time.perf_counter() - started)

    return min(timings), peak_bytes / 1e6


def scaling_with_simulations(returns, starting_balance):
    """Time and memory as the number of simulated paths grows."""
    records = []

    for count in SIMULATION_COUNTS:
        seconds, megabytes = time_run(
            lambda c=count: run_simulation(
                returns, starting_balance, n_simulations=c, seed=1, compound=True
            )
        )
        records.append({
            "n_simulations": count,
            "seconds": seconds,
            "peak_mb": megabytes,
            "within_target": seconds <= INTERACTIVE_TARGET_SECONDS,
        })
        print(f"  {count:>7,} simulations: {seconds:>6.2f}s, {megabytes:>7.0f} MB")

    return pd.DataFrame(records)


def scaling_with_trade_count(returns, starting_balance):
    """
    Time as the trade history lengthens, at a fixed number of simulations.

    Longer histories are produced by resampling the real returns, so the
    statistical character of the input is preserved while its length changes.
    """
    records = []
    rng = np.random.default_rng(0)

    for count in TRADE_COUNTS:
        if count <= len(returns):
            subset = returns[:count]
        else:
            subset = rng.choice(returns, size=count, replace=True)

        seconds, megabytes = time_run(
            lambda s=subset: run_simulation(
                s, starting_balance, n_simulations=10_000, seed=1, compound=True
            )
        )
        records.append({
            "n_trades": count,
            "seconds": seconds,
            "peak_mb": megabytes,
        })
        print(f"  {count:>7,} trades:      {seconds:>6.2f}s, {megabytes:>7.0f} MB")

    return pd.DataFrame(records)


def simulate_with_loop(returns, starting_balance, n_simulations, seed):
    """
    The obvious implementation: build each simulated path one at a time.

    Kept only as a baseline for the comparison below. It produces the same
    result as the vectorised version but constructs it simulation by
    simulation in Python rather than as a single array operation.
    """
    rng = np.random.default_rng(seed)
    n_trades = len(returns)
    paths = np.empty((n_simulations, n_trades))

    for row in range(n_simulations):
        drawn = returns[rng.integers(0, n_trades, size=n_trades)]
        paths[row] = starting_balance * np.cumprod(1.0 + drawn)

    return paths


def vectorised_versus_loop(returns, starting_balance):
    """
    Quantify the design decision to vectorise.

    Only the smaller counts are run through the loop, because at larger ones
    it takes long enough to make the point without needing to be measured.
    """
    records = []

    for count in [1_000, 5_000, 10_000, 25_000]:
        vector_seconds, _ = time_run(
            lambda c=count: run_simulation(
                returns, starting_balance, n_simulations=c, seed=1, compound=True
            )
        )
        loop_seconds, _ = time_run(
            lambda c=count: simulate_with_loop(returns, starting_balance, c, 1),
            repeats=1,
        )

        records.append({
            "n_simulations": count,
            "vectorised_seconds": vector_seconds,
            "loop_seconds": loop_seconds,
            "speedup": loop_seconds / vector_seconds,
        })
        print(
            f"  {count:>7,}: vectorised {vector_seconds:>6.2f}s, "
            f"loop {loop_seconds:>6.2f}s, {loop_seconds / vector_seconds:>5.1f}x faster"
        )

    return pd.DataFrame(records)


def full_pipeline_timing(returns, starting_balance):
    """
    Time the complete analysis, not just the simulation.

    The interactive target applies to what the user waits for, which includes
    building the benchmark and computing every comparison, so timing the
    simulation alone would understate it.
    """
    trades, _ = load_trades(DATA)
    raw = pd.read_csv(DATA, sep=";", quotechar='"')
    for column in ("Open time", "Close time"):
        raw[column] = pd.to_datetime(raw[column], format="%Y.%m.%d %H:%M:%S")
    benchmark = buy_and_hold_curve(prices_from_trades(raw), starting_balance)

    span_years = (
        trades["Close time"].iloc[-1] - trades["Open time"].iloc[0]
    ).days / 365.25
    periods = len(trades) / span_years

    records = []
    for count in [10_000, 25_000, 50_000]:
        def whole_analysis(c=count):
            paths = run_simulation(
                returns, starting_balance, n_simulations=c, seed=1, compound=True
            )
            return compare_to_benchmark(
                paths, benchmark, starting_balance, periods_per_year=periods
            )

        seconds, megabytes = time_run(whole_analysis, repeats=2)
        records.append({
            "n_simulations": count,
            "seconds": seconds,
            "peak_mb": megabytes,
            "within_target": seconds <= INTERACTIVE_TARGET_SECONDS,
        })
        print(f"  {count:>7,} simulations: {seconds:>6.2f}s, {megabytes:>7.0f} MB")

    return pd.DataFrame(records)


def plot_performance(by_simulations, by_trades, comparison, pipeline):
    figure, axes = plt.subplots(2, 2, figsize=(11, 8))

    # Time and memory against simulation count.
    left = axes[0][0]
    left.plot(by_simulations["n_simulations"], by_simulations["seconds"],
              marker="o", color="#1f3d63", label="Simulation only")
    left.plot(pipeline["n_simulations"], pipeline["seconds"],
              marker="s", color="#b3541e", label="Full analysis")
    left.axhline(INTERACTIVE_TARGET_SECONDS, color="#999999", linestyle="--",
                 linewidth=1, label=f"Target ({INTERACTIVE_TARGET_SECONDS:.0f}s)")
    left.set_xscale("log")
    left.set_xlabel("Number of simulations")
    left.set_ylabel("Seconds")
    left.set_title("Execution time")
    left.legend(frameon=False, fontsize=8)
    left.spines[["top", "right"]].set_visible(False)

    right = axes[0][1]
    right.plot(by_simulations["n_simulations"], by_simulations["peak_mb"],
               marker="o", color="#2e7d5b")
    right.set_xscale("log")
    right.set_xlabel("Number of simulations")
    right.set_ylabel("Peak memory (MB)")
    right.set_title("Memory use")
    right.spines[["top", "right"]].set_visible(False)

    lower_left = axes[1][0]
    lower_left.plot(by_trades["n_trades"], by_trades["seconds"],
                    marker="o", color="#1f3d63")
    lower_left.set_xlabel("Number of trades in history")
    lower_left.set_ylabel("Seconds")
    lower_left.set_title("Scaling with history length (10,000 simulations)")
    lower_left.spines[["top", "right"]].set_visible(False)

    lower_right = axes[1][1]
    width = 0.35
    positions = np.arange(len(comparison))
    lower_right.bar(positions - width / 2, comparison["vectorised_seconds"],
                    width, label="Vectorised", color="#1f3d63")
    lower_right.bar(positions + width / 2, comparison["loop_seconds"],
                    width, label="Loop", color="#c8b8a0")
    lower_right.set_xticks(positions)
    lower_right.set_xticklabels(
        [f"{int(n / 1000)}k" for n in comparison["n_simulations"]]
    )
    lower_right.set_xlabel("Number of simulations")
    lower_right.set_ylabel("Seconds")
    lower_right.set_title("Effect of vectorising")
    lower_right.legend(frameon=False, fontsize=8)
    lower_right.spines[["top", "right"]].set_visible(False)

    figure.tight_layout()
    figure.savefig(OUTPUT / "performance.png", dpi=150)
    print(f"\n  chart written to {OUTPUT / 'performance.png'}")


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    returns, starting_balance = load_returns()

    print("Performance: scaling with simulation count")
    by_simulations = scaling_with_simulations(returns, starting_balance)

    print("\nPerformance: scaling with trade history length")
    by_trades = scaling_with_trade_count(returns, starting_balance)

    print("\nPerformance: vectorised against a loop")
    comparison = vectorised_versus_loop(returns, starting_balance)

    print("\nPerformance: complete analysis")
    pipeline = full_pipeline_timing(returns, starting_balance)

    by_simulations.to_csv(OUTPUT / "performance_simulations.csv", index=False)
    by_trades.to_csv(OUTPUT / "performance_trades.csv", index=False)
    comparison.to_csv(OUTPUT / "performance_vectorisation.csv", index=False)
    pipeline.to_csv(OUTPUT / "performance_pipeline.csv", index=False)
    plot_performance(by_simulations, by_trades, comparison, pipeline)

    breached = pipeline[~pipeline["within_target"]]
    if breached.empty:
        print(
            f"\n  Every configuration tested completed within the "
            f"{INTERACTIVE_TARGET_SECONDS:.0f}s interactive target."
        )
    else:
        first = int(breached["n_simulations"].iloc[0])
        print(f"\n  Interactive target first breached at {first:,} simulations.")

    peak = by_simulations["peak_mb"].max()
    print(f"  Peak memory across all configurations: {peak:,.0f} MB")


if __name__ == "__main__":
    main()
