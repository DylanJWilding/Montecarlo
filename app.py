"""
app.py

The user interface.

Deliberately contains no analysis of its own: every figure shown here is
produced by the core modules, which have no knowledge of this file. That
separation is what allows the statistical work to be tested in isolation
(NFR5), and it means the interface can be changed without any risk of
altering a result.

The presentation follows the decision recorded during design: the simulated
equity paths are drawn overlaid on a single chart with the real backtest
picked out among them. That form was chosen over percentile bands or
histograms because it communicates the central idea without requiring the
reader to know any statistics. They see many versions of the same strategy
and can tell at once that the single curve they would normally be shown is
only one of them.
"""

import io
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent / "src"))

from ingest import load_trades, TradeDataError
from simulate import (
    run_simulation,
    equity_curve,
    to_returns,
    WITH_REPLACEMENT,
    WITHOUT_REPLACEMENT,
    SimulationError,
)
from compare import (
    buy_and_hold_curve,
    prices_from_trades,
    compare_to_benchmark,
    ComparisonError,
)
from metrics import build_report, report_to_rows


# Drawing every path would produce an unreadable block of colour and a very
# large image, so a random subset is plotted. The statistics are always
# computed on the full set; this limit affects only what is drawn.
MAX_PATHS_DRAWN = 300

st.set_page_config(page_title="Lucky or Good?", layout="wide")


def main():
    st.title("Lucky or Good?")
    st.caption(
        "Testing whether an algorithmic trading strategy really beats buy and hold, "
        "or whether its backtest just got a favourable run of trades."
    )

    settings = sidebar_controls()

    if settings["file"] is None:
        show_introduction()
        return

    try:
        analyse(settings)
    except (TradeDataError, SimulationError, ComparisonError) as error:
        st.error(str(error))


def sidebar_controls():
    """Collect the input file and simulation settings (FR16)."""
    with st.sidebar:
        st.header("Input")
        uploaded = st.file_uploader("Strategy trade export (CSV)", type="csv")

        st.header("Simulation")
        n_simulations = st.select_slider(
            "Number of simulations",
            options=[1_000, 5_000, 10_000, 25_000, 50_000],
            value=10_000,
            help="More simulations give a steadier estimate but take longer.",
        )
        mode_label = st.radio(
            "Resampling method",
            ["Draw trades at random (with replacement)",
             "Reshuffle the same trades (without replacement)"],
            help=(
                "Drawing at random varies the total profit, which is what "
                "produces a range of possible outcomes to compare against the "
                "market. Reshuffling keeps the same trades so every run ends "
                "at the same balance, and only the path there changes."
            ),
        )
        compound = st.checkbox(
            "Strategy risks a percentage of the account",
            value=True,
            help=(
                "Tick this if position sizes grow with the account. Trades are "
                "then treated as percentage returns and compounded, because a "
                "given trade's cash result only means something relative to "
                "the balance it was taken on."
            ),
        )
        seed = st.number_input(
            "Random seed", min_value=0, max_value=999_999, value=42,
            help="The same seed always produces the same result.",
        )

    return {
        "file": uploaded,
        "n_simulations": n_simulations,
        "mode": WITH_REPLACEMENT if mode_label.startswith("Draw") else WITHOUT_REPLACEMENT,
        "compound": compound,
        "seed": int(seed),
    }


def show_introduction():
    """Shown before a file is uploaded, explaining the idea from scratch."""
    st.info("Upload a strategy's trade history in the sidebar to begin.")

    st.subheader("What this does")
    st.markdown(
        """
A **backtest** shows what a trading strategy would have made on past market
data. It produces a single line, the **equity curve**, tracing the account
balance over time. Traders routinely judge a strategy on that one line.

The problem is that the line shows only one ordering of trades, and the order
was largely luck. The same trades arriving in a different sequence would have
produced a different-looking result, sometimes a much worse one.

This tool takes the strategy's actual trades and rebuilds the account
thousands of times, each with the trades drawn in a different order. That
produces a range of results the strategy could plausibly have produced, rather
than the single one it happened to.

Each of those is then compared against **buy and hold**: simply purchasing the
market at the start and holding to the end. If a strategy cannot beat that, its
complexity has earned nothing.
        """
    )


def analyse(settings):
    """Run the full pipeline and display the results."""
    raw = read_raw(settings["file"])
    trades, starting_balance = load_trades(io.BytesIO(settings["file"].getvalue()))

    profits = trades["Profit/Loss"].to_numpy()
    balances = trades["Balance"].to_numpy()

    # Under percentage sizing the simulation works in fractional returns; under
    # fixed sizing it works in cash. See the note in simulate._accumulate.
    values = to_returns(profits, balances) if settings["compound"] else profits

    span_years = (
        trades["Close time"].iloc[-1] - trades["Open time"].iloc[0]
    ).days / 365.25
    trades_per_year = len(trades) / span_years if span_years > 0 else None

    with st.spinner(f"Running {settings['n_simulations']:,} simulations..."):
        paths = run_simulation(
            values,
            starting_balance,
            n_simulations=settings["n_simulations"],
            mode=settings["mode"],
            seed=settings["seed"],
            compound=settings["compound"],
        )
        actual = equity_curve(values, starting_balance, compound=settings["compound"])
        benchmark = buy_and_hold_curve(prices_from_trades(raw), starting_balance)
        comparison = compare_to_benchmark(
            paths, benchmark, starting_balance, periods_per_year=trades_per_year
        )
        report = build_report(
            comparison, actual, starting_balance, periods_per_year=trades_per_year
        )

    show_headline(report)
    show_chart(paths, actual, benchmark, starting_balance)
    show_comparison_table(report)
    show_where_the_real_result_sits(report)
    show_export(report, settings)

    st.caption(
        "These figures are probabilities estimated from historical data. They "
        "describe what the strategy might have done, not what it will do."
    )


def read_raw(uploaded_file):
    """Read the upload again with the price columns needed for the benchmark."""
    raw = pd.read_csv(io.BytesIO(uploaded_file.getvalue()), sep=";", quotechar='"')
    for column in ("Open time", "Close time"):
        raw[column] = pd.to_datetime(raw[column], format="%Y.%m.%d %H:%M:%S")
    return raw


def show_headline(report):
    """The answer, stated in words before any chart or table (FR15)."""
    st.subheader("Did it beat buy and hold?")

    probabilities = report["probability_of_beating_benchmark"]
    criteria = [
        ("on_return", "On profit", "made more money"),
        ("on_return_to_drawdown", "On profit per unit of loss", "earned more per unit of decline suffered"),
        ("on_sharpe", "On steadiness of returns", "produced steadier returns"),
    ]

    columns = st.columns(len([c for c in criteria if c[0] in probabilities]))
    for column, (key, label, phrase) in zip(columns, criteria):
        if key not in probabilities:
            continue
        result = probabilities[key]
        with column:
            st.metric(label, f"{result['probability']:.1%}")
            st.caption(
                f"of simulations {phrase} than buy and hold "
                f"(95% confidence: {result['lower']:.1%} to {result['upper']:.1%})"
            )


def show_chart(paths, actual, benchmark, starting_balance):
    """The overlaid-paths chart, as decided during design."""
    st.subheader("Every way this strategy could have gone")

    figure, axes = plt.subplots(figsize=(11, 5.5))

    drawn = paths[:MAX_PATHS_DRAWN]
    trade_numbers = np.arange(1, paths.shape[1] + 1)

    for path in drawn:
        axes.plot(trade_numbers, path, color="#8fa8c8", alpha=0.06, linewidth=0.6)

    axes.plot(trade_numbers, actual, color="#1f3d63", linewidth=2.0,
              label="The backtest that actually happened")

    # The benchmark is measured over price observations rather than trades, so
    # it is stretched onto the same axis to make the comparison visible.
    benchmark_x = np.linspace(1, paths.shape[1], len(benchmark))
    axes.plot(benchmark_x, benchmark, color="#b3541e", linewidth=2.0,
              linestyle="--", label="Buy and hold")

    axes.axhline(starting_balance, color="#999999", linewidth=0.8, linestyle=":")
    axes.set_xlabel("Number of trades")
    axes.set_ylabel("Account balance")
    axes.legend(loc="upper left", frameon=False)
    axes.spines[["top", "right"]].set_visible(False)
    axes.set_yscale("log")

    st.pyplot(figure)
    st.caption(
        f"Each faint line is one simulated run of the same trades in a different "
        f"order ({min(len(paths), MAX_PATHS_DRAWN)} of {len(paths):,} shown). "
        f"A logarithmic scale is used so that proportional changes are "
        f"comparable at every balance."
    )


def show_comparison_table(report):
    st.subheader("The numbers")

    rows = {
        "Total return": ("return", "{:.1%}"),
        "Worst decline from a peak": ("max_drawdown", "{:.1%}"),
        "Return per unit of decline": ("return_to_drawdown", "{:.2f}"),
        "Sharpe ratio (annualised)": ("sharpe", "{:.2f}"),
    }

    table = []
    for label, (key, fmt) in rows.items():
        if key not in report["actual"]:
            continue
        table.append({
            "": label,
            "The real backtest": fmt.format(report["actual"][key]),
            "Buy and hold": (
                fmt.format(report["benchmark"][key])
                if key in report["benchmark"] else "n/a"
            ),
            "Simulated median": (
                fmt.format(report["simulated"][key]["median"])
                if key in report["simulated"] else "n/a"
            ),
            "Simulated range (5th to 95th)": (
                f"{fmt.format(report['simulated'][key]['p5'])} to "
                f"{fmt.format(report['simulated'][key]['p95'])}"
                if key in report["simulated"] else "n/a"
            ),
        })

    st.dataframe(pd.DataFrame(table), hide_index=True, use_container_width=True)


def show_where_the_real_result_sits(report):
    """
    Locates the historical backtest within the simulated range, which is the
    most direct answer to whether it was lucky.
    """
    st.subheader("Was the real backtest lucky?")

    percentiles = report["actual_result_percentile"]
    return_percentile = percentiles["return"]

    if return_percentile >= 80:
        verdict = (
            f"The real backtest returned more than {return_percentile:.0f}% of the "
            "simulated runs. It sits towards the fortunate end of what these trades "
            "could have produced, so the historical figure likely flatters the strategy."
        )
    elif return_percentile <= 20:
        verdict = (
            f"The real backtest returned more than only {return_percentile:.0f}% of the "
            "simulated runs. The historical result was on the unlucky side, and the "
            "strategy may be better than its backtest suggests."
        )
    else:
        verdict = (
            f"The real backtest returned more than {return_percentile:.0f}% of the "
            "simulated runs, placing it around the middle. The historical result looks "
            "typical of these trades rather than notably lucky or unlucky."
        )

    st.write(verdict)
    st.caption(
        f"Its worst decline was deeper than {percentiles['max_drawdown']:.0f}% of "
        "simulated runs."
    )


def show_export(report, settings):
    """Export results with the settings needed to reproduce them (FR17)."""
    st.subheader("Export")

    payload = {
        "settings": {
            "n_simulations": settings["n_simulations"],
            "mode": settings["mode"],
            "compound": settings["compound"],
            "seed": settings["seed"],
        },
        "results": report,
    }

    columns = st.columns(2)
    with columns[0]:
        st.download_button(
            "Download results (JSON)",
            data=json.dumps(payload, indent=2, default=float),
            file_name="simulation_results.json",
            mime="application/json",
        )
    with columns[1]:
        frame = pd.DataFrame(report_to_rows(report), columns=["metric", "value"])
        st.download_button(
            "Download results (CSV)",
            data=frame.to_csv(index=False),
            file_name="simulation_results.csv",
            mime="text/csv",
        )

    st.caption(
        "Exports include the random seed and settings, so any result here can "
        "be reproduced exactly."
    )


if __name__ == "__main__":
    main()
