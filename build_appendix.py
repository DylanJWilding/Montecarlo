"""
build_appendix.py

Assembles Appendix A (source code) into a single text file with clear
headings for each module, ready to paste into Word.

Run from the project root:  python build_appendix.py
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "appendix_a_source_code.txt"

# Order matters: the reader should meet the modules in pipeline order,
# then the tests, then the evaluation scripts.
FILES = [
    ("Core modules", [
        "src/ingest.py",
        "src/simulate.py",
        "src/compare.py",
        "src/metrics.py",
    ]),
    ("User interface", [
        "app.py",
    ]),
    ("Test suite", [
        "tests/test_ingest.py",
        "tests/test_simulate.py",
        "tests/test_compare.py",
        "tests/test_metrics.py",
    ]),
    ("Evaluation scripts", [
        "evaluation/convergence.py",
        "evaluation/performance.py",
        "evaluation/sensitivity.py",
    ]),
]

RULE = "=" * 78


def main():
    parts = []
    total_lines = 0
    missing = []

    parts.append(RULE)
    parts.append("APPENDIX A: SOURCE CODE")
    parts.append(RULE)
    parts.append("")
    parts.append("All code presented in this appendix was written by the author.")
    parts.append("No machine-generated or third-party code is included.")
    parts.append("")

    for section, paths in FILES:
        parts.append("")
        parts.append(RULE)
        parts.append(section.upper())
        parts.append(RULE)

        for rel in paths:
            path = ROOT / rel
            if not path.exists():
                missing.append(rel)
                continue

            text = path.read_text(encoding="utf-8")
            line_count = len(text.splitlines())
            total_lines += line_count

            parts.append("")
            parts.append("-" * 78)
            parts.append(f"  {rel}   ({line_count} lines)")
            parts.append("-" * 78)
            parts.append("")
            parts.append(text.rstrip())

    parts.append("")
    parts.append(RULE)
    parts.append(f"Total: {total_lines} lines across "
                 f"{sum(len(p) for _, p in FILES) - len(missing)} files.")
    parts.append(RULE)

    OUTPUT.write_text("\n".join(parts), encoding="utf-8")

    print(f"Written to {OUTPUT.name}")
    print(f"  {total_lines} lines of code")
    if missing:
        print("\n  NOT FOUND (check the paths in FILES):")
        for rel in missing:
            print(f"    {rel}")


if __name__ == "__main__":
    main()
