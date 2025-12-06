import argparse
import sys

import pandas as pd
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
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

    # Read CSV/TSV (auto-detect separator)
    try:
        df = pd.read_csv(args.path, sep=None, engine="python")
    except Exception as e:
        print(f"Error reading file {args.path}: {e}", file=sys.stderr)
        sys.exit(1)

    # Basic checks
    if "perturbation" not in df.columns:
        print("Error: 'perturbation' column not found in the file.", file=sys.stderr)
        print(f"Columns present: {list(df.columns)}", file=sys.stderr)
        sys.exit(1)

    if args.metric not in df.columns:
        print(f"Error: metric column '{args.metric}' not found.", file=sys.stderr)
        print(f"Columns present: {list(df.columns)}", file=sys.stderr)
        sys.exit(1)

    # Select relevant columns
    sub = df[["perturbation", args.metric]].copy()

    # Convert to numeric (just in case), coerce non-numeric to NaN
    sub[args.metric] = pd.to_numeric(sub[args.metric], errors="coerce")

    # Drop NaNs
    sub = sub.dropna(subset=[args.metric])

    # Sort by metric descending
    sub = sub.sort_values(by=args.metric, ascending=False)

    # Optionally keep only top N
    if args.top is not None and args.top > 0:
        sub = sub.head(args.top)

    # Plot
    plt.figure(figsize=(10, 6))
    plt.bar(sub["perturbation"], sub[args.metric])
    plt.xticks(rotation=90)
    plt.xlabel("Perturbation")
    plt.ylabel(args.metric)
    plt.title(f"{args.metric} by perturbation")
    plt.tight_layout()

    if args.output:
        plt.savefig(args.output, dpi=300, bbox_inches="tight")
        print(f"Saved plot to {args.output}")
    else:
        plt.show()


if __name__ == "__main__":
    main()

