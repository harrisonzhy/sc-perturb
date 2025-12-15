import argparse
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import random
import numpy as np

def load_baseline_dict(path, metric_col):
    """Load baseline CSV as a minimal dict: feature -> metric."""
    usecols = ["feature", metric_col]
    df = pd.read_csv(path, usecols=usecols)
    # Drop NaNs in the metric
    df = df.dropna(subset=[metric_col])
    # Map feature -> metric
    baseline_map = dict(zip(df["feature"], df[metric_col]))
    return baseline_map


def classify_direction(old_val, new_val, metric_name):
    """Classify direction consistency for coloring."""
    if metric_name == "fold_change":
        old_up = old_val > 1
        new_up = new_val > 1
    else:  # percent_change
        old_up = old_val > 0
        new_up = new_val > 0

    if old_up and new_up:
        return "green"   # both up
    elif (not old_up) and (not new_up):
        return "red"     # both down
    else:
        return "orange"  # direction changed


def main():
    parser = argparse.ArgumentParser(
        description="Memory-efficient scatter plot comparison of effect sizes between two datasets."
    )
    parser.add_argument("-p", "--path", required=True,
                        help="Path to NEW dataset CSV")
    parser.add_argument("-b", "--baseline", required=True,
                        help="Path to BASELINE dataset CSV")
    parser.add_argument("-t", "--title", required=True,
                        help="Title for the plot")
    parser.add_argument("-m", "--metric",
                        choices=["fold_change", "percent_change"],
                        default="fold_change",
                        help="Effect size metric to compare (default: fold_change)")
    parser.add_argument("-o", "--output",
                        default="scatter_plot.png",
                        help="Output image file (default: scatter_plot.png)")
    parser.add_argument("--max-points", type=int,
                        default=200_000,
                        help="Maximum number of points to keep for plotting (default: 200000)")
    parser.add_argument("--chunksize", type=int,
                        default=100_000,
                        help="CSV chunk size for streaming (default: 100000)")

    args = parser.parse_args()
    metric_col = args.metric

    # 1) Load baseline into a minimal dict
    print(f"Loading baseline from {args.baseline} ...")
    baseline_map = load_baseline_dict(args.baseline, metric_col)
    if not baseline_map:
        raise ValueError("Baseline file produced an empty mapping. Check columns and data.")

    print(f"Baseline features loaded: {len(baseline_map)}")

    # 2) Stream the new dataset in chunks and do a streaming join
    x_vals = []
    y_vals = []
    colors = []

    n_seen = 0  # total matching points seen before sampling
    max_points = args.max_points

    print(f"Streaming new data from {args.path} in chunks of {args.chunksize} ...")

    usecols = ["feature", metric_col]
    for chunk in pd.read_csv(args.path, usecols=usecols, chunksize=args.chunksize):
        # Drop rows with NaN metric
        chunk = chunk.dropna(subset=[metric_col])

        # Keep only those features present in baseline
        # Faster to use .map for lookup
        chunk["old_metric"] = chunk["feature"].map(baseline_map)

        # Drop rows where baseline metric is missing
        chunk = chunk.dropna(subset=["old_metric"])
        if chunk.empty:
            continue

        for _, row in chunk.iterrows():
            old_val = row["old_metric"]
            new_val = row[metric_col]

            # Skip if either is NaN (safety)
            if pd.isna(old_val) or pd.isna(new_val):
                continue

            visibility_thres = 15
            if max(abs(new_val), abs(old_val)) > visibility_thres:
                continue

            n_seen += 1

            # Decide color based on direction
            c = classify_direction(old_val, new_val, args.metric)

            # Reservoir sampling: keep at most max_points uniformly
            if len(x_vals) < max_points:
                x_vals.append(old_val)
                y_vals.append(new_val)
                colors.append(c)
            else:
                j = random.randint(0, n_seen - 1)
                if j < max_points:
                    x_vals[j] = old_val
                    y_vals[j] = new_val
                    colors[j] = c

    if not x_vals:
        raise ValueError("No overlapping (non-NaN) feature metrics found between the two datasets.")

    print(f"Total matching points seen: {n_seen}")
    print(f"Points kept for plotting: {len(x_vals)}")

    # 3) Make the scatter plot
    plt.figure(figsize=(10, 8))

    x_arr = np.asarray(x_vals)
    y_arr = np.asarray(y_vals)

    jitter = 0.05
    x_plot = x_arr + np.random.normal(0, jitter, size=len(x_arr))
    y_plot = y_arr + np.random.normal(0, jitter, size=len(y_arr))

    plt.scatter(
        x_arr,
        y_arr,
        c=colors,
        alpha=0.7,
        linewidth=0,
        s=4,
        edgecolor="none",
    )


    # Identity line (y = x)
    min_val = min(min(x_vals), min(y_vals))
    max_val = max(max(x_vals), max(y_vals))
    padding = (max_val - min_val) * 0.05 if max_val > min_val else 1.0
    lo = min_val - padding
    hi = max_val + padding
    plt.plot([lo, hi], [lo, hi], "k--", linewidth=1)

    # Axis labels
    if args.metric == "fold_change":
        xlabel = "Fold Change (Baseline)"
        ylabel = "Fold Change (Ours)"
    else:
        xlabel = "Percent Change (Baseline)"
        ylabel = "Percent Change (Ours)"

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(args.title)

    # Legend
    legend_patches = [
        Patch(color="green", label="Both Up"),
        Patch(color="red", label="Both Down"),
        Patch(color="orange", label="Direction Changed"),
    ]
    plt.legend(handles=legend_patches)

    plt.tight_layout()
    plt.savefig(args.output, dpi=300)
    print(f"Scatter plot saved to {args.output}")


if __name__ == "__main__":
    main()

