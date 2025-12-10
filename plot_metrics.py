import argparse
import sys

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--baseline",
        "-b",
        help="Path to the baseline CSV/TSV file with columns including 'perturbation' and a metric.",
    )

    parser.add_argument(
        "--path",
        "-p",
        help="Path to the CSV/TSV file with columns including 'perturbation' and a metric.",
    )
    parser.add_argument(
        "--metric",
        "-m",
        help="Name of the metric column to plot (e.g. precision_at_N, overlap_at_50).",
    )
    parser.add_argument(
        "--title",
        "-t",
        help="Title of plot",
    )
    parser.add_argument(
        "--output",
        "-o",
        default="plot.png",
        help="Optional path to save the plot (e.g. plot.png). If omitted, the plot is just shown.",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=None,
        help="Optionally limit to the top N perturbations by metric value.",
    )
    args = parser.parse_args()

    try:
        df = pd.read_csv(args.path, sep=None, engine="python")
        df_baseline = pd.read_csv(args.baseline, sep=None, engine="python")
    except Exception as e:
        print(f"Error reading file: {e}", file=sys.stderr)
        sys.exit(1)

    # Basic checks
    if "perturbation" not in df.columns or "perturbation" not in df_baseline.columns:
        print("Error: 'perturbation' column not found in the file.", file=sys.stderr)
        print(f"Columns present: {list(df.columns)}", file=sys.stderr)
        sys.exit(1)

    if args.metric not in df.columns or args.metric not in df_baseline.columns:
        print(f"Error: metric column '{args.metric}' not found.", file=sys.stderr)
        print(f"Columns present: {list(df.columns)}", file=sys.stderr)
        sys.exit(1)

    # Select relevant columns
    sub = df[["perturbation", args.metric]].copy()
    sub_baseline = df_baseline[["perturbation", args.metric]].copy()

    # Convert to numeric (just in case), coerce non-numeric to NaN
    sub[args.metric] = pd.to_numeric(sub[args.metric], errors="coerce")
    sub_baseline[args.metric] = pd.to_numeric(sub_baseline[args.metric], errors="coerce")

    # Drop NaNs
    sub = sub.dropna(subset=[args.metric])
    sub_baseline = sub_baseline.dropna(subset=[args.metric])

    key = "perturbation"

    # Sort by metric descending
    sub = sub.sort_values(by=args.metric, ascending=False)
    if args.top is not None and args.top > 0:
        sub = sub.head(args.top)
    sub_baseline = sub_baseline.loc[sub.index]

    # Optionally keep only top N
    if args.top is not None and args.top > 0:
        sub = sub.head(args.top)

    sub_baseline = sub_baseline.set_index(key).loc[sub[key]].reset_index()

    baseline = sub_baseline[args.metric].values
    improved = sub[args.metric].values
    delta = improved - baseline

    x = np.arange(len(sub))
    w = 0.6

    plt.figure(figsize=(20, 8))

    blue = "#046bb3"
    green = "#90ee90"
    red = "#ff9999"

    # Baseline bars (solid light blue)
    plt.bar(
        x,
        baseline,
        width=w,
        color=blue,
        label="Baseline"
    )

    # Stacked delta bars
    delta_colors = [green if d >= 0 else red for d in delta]

    # Plot using baseline as the original for all deltas
    plt.bar(
        x,
        delta,
        width=w,
        bottom=baseline,
        color=delta_colors,
    )

    plt.xticks(x, sub["perturbation"], rotation=90)
    plt.xlabel("Perturbation")
    plt.ylabel(args.metric)
    plt.title(args.title)

    # Legend
    from matplotlib.patches import Patch
    legend_patches = [
        Patch(color=blue, label="Baseline"),
        Patch(color=green, label="Improvement"),
        Patch(color=red, label="Regression")
    ]
    plt.legend(handles=legend_patches)
    plt.tight_layout()

    if args.output:
        plt.savefig(args.output, dpi=300, bbox_inches="tight")
        print(f"Saved plot to {args.output}")
    else:
        plt.show()

if __name__ == "__main__":
    main()

