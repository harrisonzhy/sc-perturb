import argparse
import sys

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def load_metric_series(path: str, metric: str) -> pd.Series:
    """Load a numeric metric column from a CSV/TSV, dropping non-numeric values."""
    try:
        df = pd.read_csv(path, sep=None, engine="python")
    except Exception as e:
        print(f"Error reading file '{path}': {e}", file=sys.stderr)
        sys.exit(1)

    if metric not in df.columns:
        print(
            f"Error: metric column '{metric}' not found in '{path}'. "
            f"Columns present: {list(df.columns)}",
            file=sys.stderr,
        )
        sys.exit(1)

    s = pd.to_numeric(df[metric], errors="coerce").dropna()
    if s.empty:
        print(
            f"Error: metric column '{metric}' in '{path}' has no valid numeric values.",
            file=sys.stderr,
        )
        sys.exit(1)

    return s


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--baseline",
        "-b",
        required=True,
        help="Path to the baseline CSV/TSV file with a metric column.",
    )
    parser.add_argument(
        "--path",
        "-p",
        required=True,
        help="Path to the CSV/TSV file for the new data with the same metric column.",
    )
    parser.add_argument(
        "--metric",
        "-m",
        required=True,
        help="Name of the metric column to use (e.g. precision_at_N, overlap_at_50).",
    )
    parser.add_argument(
        "--title",
        "-t",
        help="Title of the plot. If omitted, a default based on the metric name is used.",
    )
    parser.add_argument(
        "--output",
        "-o",
        default="distribution.png",
        help="Optional path to save the plot (e.g. distribution.png). If omitted, the plot is just shown.",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=None,
        help="Optionally limit to the top N values (after sorting) from each distribution.",
    )
    args = parser.parse_args()

    # Load metric values
    base_series = load_metric_series(args.baseline, args.metric)
    new_series = load_metric_series(args.path, args.metric)

    # Sort values from highest to lowest
    baseline_vals = np.sort(base_series.values)[::-1]
    new_vals = np.sort(new_series.values)[::-1]

    # Optionally limit to top N
    if args.top is not None and args.top > 0:
        baseline_vals = baseline_vals[: args.top]
        new_vals = new_vals[: args.top]

    # X-axes are just ranks (1 = highest value)
    x_base = np.arange(1, len(baseline_vals) + 1)
    x_new = np.arange(1, len(new_vals) + 1)

    plt.figure(figsize=(12, 6))

    # Colors (keeps baseline blue-ish like the original script)
    baseline_color = "#046bb3"
    new_color = "#90ee90"

    plt.plot(x_base, baseline_vals, label="Baseline", linewidth=2, color=baseline_color)
    plt.plot(x_new, new_vals, label="New", linewidth=2, color=new_color, alpha=0.8)

    plt.xlabel("Ordered rank (1 is highest)")
    plt.ylabel(args.metric)
    plt.title(args.title or f"Distribution of {args.metric} (sorted descending)")

    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()

    if args.output:
        plt.savefig(args.output, dpi=300, bbox_inches="tight")
        print(f"Saved plot to {args.output}")
    else:
        plt.show()


if __name__ == "__main__":
    main()

