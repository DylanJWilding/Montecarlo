"""
sensitivity.py

Evaluation: how much does the answer depend on position sizing?

The strategy under test risks a fixed percentage of the account on each trade.
That percentage is chosen by whoever trades the strategy, not by the strategy
itself, and it is not obvious in advance how much the choice matters.

This script re-runs the whole comparison across a range of sizing regimes. It
was written after noticing during development that the conclusion appeared to
change with sizing, and its purpose is to establish whether that impression
holds and how large the effect is.

The method rescales each trade into a multiple of the amount risked and
compounds it at the chosen percentage. That assumes a trade's outcome relative
to its risk is unchanged at a different position size, which holds reasonably
for a strategy using fixed stop distances but ignores slippage and available
liquidity at larger sizes. The assumption is stated because the results depend
on it.

Run from the project root:  python evaluation/sensitivity.py
"""

import sys
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
from compare import (
    buy_and_hold_curve,
    prices_from_trades,
    compare_to_benchmark,
    max_drawdown,
    total_return,
    return_to_drawdown,
)


DATA = ROOT / "tests" / "fixtures" / "sample_trades.csv"
OUTPUT = ROOT / "evaluation" / "results"

RISK_LEVELS = [0.0025, 0.005, 0.0075, 0.01, 0.015, 0.02, 0.03, 0.05]
BASELINE_RISK = 0.01          # the level used for the primary results
N_SIMULATIONS = 25_000
SEED = 42


def load_inputs():
    trades, starting_balance = load_trades(DATA)

    raw = pd.read_csv(DATA, sep=";", quotechar='"')
    for column in ("Open time", "Close time"):
        raw[column] = pd.to_datetime(raw[column], format="%Y.%m.%d %H:%M:%S")

    profits = trades["Profit/Loss"].to_numpy()
    balances = trades["Balance"].to_numpy()
    returns = to_returns(profits, balances)

    # Each trade as a multiple of the amount actually risked. Dividing the
    # observed fractional return by the risk level the export used recovers
    # the outcome in units of risk, which can then be re-applied at any other
    # level.
    observed_risk = estimate_risk_level(raw)
    risk_multiples = returns / observed_risk

    benchmark = buy_and_hold_curve(prices_from_trades(raw), starting_balance)
    span_years = (
        trades["Close time"].iloc[-1] - trades["Open time"].iloc[0]
    ).days / 365.25

    return {
        "risk_multiples": risk_multiples,
        "observed_risk": observed_risk,
        "starting_balance": starting_balance,
        "benchmark": benchmark,
        "periods_per_year": len(trades) / span_years,
    }


def estimate_risk_level(raw):
    """
    Recover the fraction of the account risked per trade from the export.

    Trades closed at a stop loss lost close to the full amount risked, so the
    typical such loss, expressed against the balance before it, gives the
    risk level. The median is used rather than the mean because it is not
    disturbed by the occasional trade that slipped beyond its stop.

    The raw export is used rather than the cleaned trade table, because the
    ingest stage discards the exit-reason column: it is not needed by the
    simulation and is only required here.
    """
    stopped = raw[raw["Close type"] == "SL"]

    if stopped.empty:
        raise ValueError("No stop-loss exits found; cannot infer risk level")

    balance_before = stopped["Balance"] - stopped["Profit/Loss"]
    return float((-stopped["Profit/Loss"] / balance_before).median())


def evaluate_at_risk(inputs, risk):
    """Run the whole comparison with the strategy sized at the given risk."""
    returns = inputs["risk_multiples"] * risk
    starting_balance = inputs["starting_balance"]

    actual = equity_curve(returns, starting_balance, compound=True)
    paths = run_simulation(
        returns, starting_balance,
        n_simulations=N_SIMULATIONS, seed=SEED, compound=True,
    )
    comparison = compare_to_benchmark(
        paths, inputs["benchmark"], starting_balance,
        periods_per_year=inputs["periods_per_year"],
    )

    actual_return = float(total_return(actual, starting_balance))
    actual_drawdown = float(max_drawdown(actual, starting_balance))

    return {
        "risk_per_trade": risk,
        "actual_return": actual_return,
        "actual_max_drawdown": actual_drawdown,
        "actual_return_to_drawdown": float(
            return_to_drawdown(actual_return, actual_drawdown)
        ),
        "p_beats_on_return": float(comparison["beats_on_return"].mean()),
        "p_beats_on_ratio": float(comparison["beats_on_ratio"].mean()),
        "p_beats_on_sharpe": float(comparison["beats_on_sharpe"].mean()),
        "median_sim_return": float(np.median(comparison["path_returns"])),
        "median_sim_drawdown": float(np.median(comparison["path_drawdowns"])),
        "p95_sim_drawdown": float(np.percentile(comparison["path_drawdowns"], 95)),
        "benchmark_return": comparison["benchmark_return"],
        "benchmark_drawdown": comparison["benchmark_drawdown"],
    }


def plot_sensitivity(results, benchmark_return, benchmark_drawdown):
    figure, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    risk_percent = results["risk_per_trade"] * 100

    # Panel 1: does it beat the benchmark, on each criterion?
    left = axes[0]
    for label, column, colour in (
        ("On return", "p_beats_on_return", "#b3541e"),
        ("On return/drawdown", "p_beats_on_ratio", "#1f3d63"),
        ("On Sharpe", "p_beats_on_sharpe", "#2e7d5b"),
    ):
        left.plot(risk_percent, results[column] * 100,
                  marker="o", color=colour, label=label)
    left.axhline(50, color="#999999", linestyle="--", linewidth=1)
    left.axvline(BASELINE_RISK * 100, color="#333333", linestyle=":",
                 linewidth=1.2, label="Primary case")
    left.set_xlabel("Risk per trade (% of account)")
    left.set_ylabel("Simulations beating benchmark (%)")
    left.set_title("Does it beat buy and hold?")
    left.legend(frameon=False, fontsize=8)
    left.spines[["top", "right"]].set_visible(False)

    # Panel 2: return, against the benchmark's.
    middle = axes[1]
    middle.plot(risk_percent, results["actual_return"] * 100,
                marker="o", color="#1f3d63", label="Strategy")
    middle.axhline(benchmark_return * 100, color="#b3541e", linestyle="--",
                   linewidth=1.2, label="Buy and hold")
    middle.axvline(BASELINE_RISK * 100, color="#333333", linestyle=":", linewidth=1.2)
    middle.set_yscale("log")
    middle.set_xlabel("Risk per trade (% of account)")
    middle.set_ylabel("Total return (%)")
    middle.set_title("Return")
    middle.legend(frameon=False, fontsize=8)
    middle.spines[["top", "right"]].set_visible(False)

    # Panel 3: the cost of that return.
    right = axes[2]
    right.plot(risk_percent, results["actual_max_drawdown"] * 100,
               marker="o", color="#1f3d63", label="Strategy (actual)")
    right.plot(risk_percent, results["p95_sim_drawdown"] * 100,
               marker="^", markersize=4, color="#8fa8c8", linestyle="-.",
               label="Simulated 95th percentile")
    right.axhline(benchmark_drawdown * 100, color="#b3541e", linestyle="--",
                  linewidth=1.2, label="Buy and hold")
    right.axvline(BASELINE_RISK * 100, color="#333333", linestyle=":", linewidth=1.2)
    right.set_xlabel("Risk per trade (% of account)")
    right.set_ylabel("Maximum drawdown (%)")
    right.set_title("Risk taken")
    right.legend(frameon=False, fontsize=8)
    right.spines[["top", "right"]].set_visible(False)

    figure.tight_layout()
    figure.savefig(OUTPUT / "sensitivity.png", dpi=150)
    print(f"\n  chart written to {OUTPUT / 'sensitivity.png'}")


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)

    inputs = load_inputs()
    print("Position sizing sensitivity")
    print(f"  risk level found in the export: {inputs['observed_risk']:.3%}")
    print(f"  {N_SIMULATIONS:,} simulations at each level\n")

    results = pd.DataFrame(
        [evaluate_at_risk(inputs, risk) for risk in RISK_LEVELS]
    )

    benchmark_return = results["benchmark_return"].iloc[0]
    benchmark_drawdown = results["benchmark_drawdown"].iloc[0]

    print(f"  Benchmark: {benchmark_return:.1%} return, "
          f"{benchmark_drawdown:.1%} maximum drawdown\n")
    print("  risk    return    maxDD   ret/DD   P(beat return)  P(ret/DD)  P(Sharpe)")
    print("  " + "-" * 72)
    for _, row in results.iterrows():
        print(
            f"  {row['risk_per_trade']:>4.2%}  "
            f"{row['actual_return']:>7.1%}  "
            f"{row['actual_max_drawdown']:>6.1%}  "
            f"{row['actual_return_to_drawdown']:>6.2f}  "
            f"{row['p_beats_on_return']:>13.1%}  "
            f"{row['p_beats_on_ratio']:>9.1%}  "
            f"{row['p_beats_on_sharpe']:>9.1%}"
        )

    results.to_csv(OUTPUT / "sensitivity.csv", index=False)
    plot_sensitivity(results, benchmark_return, benchmark_drawdown)

    # Where, if anywhere, does the verdict on return change?
    beats = results[results["p_beats_on_return"] > 0.5]
    if beats.empty:
        print(
            "\n  At no tested risk level did the strategy beat the benchmark "
            "on return in the majority of simulations."
        )
    else:
        crossover = beats["risk_per_trade"].iloc[0]
        print(
            f"\n  The strategy first beats the benchmark on return in the "
            f"majority of simulations at {crossover:.2%} risk per trade."
        )

    print(
        "  Note that the Sharpe ratio is unchanged by sizing, since scaling "
        "every return by the same factor scales mean and standard deviation "
        "equally. Any variation shown is simulation noise."
    )


if __name__ == "__main__":
    main()
