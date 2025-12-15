import argparse
import pandas as pd
import matplotlib.pyplot as plt

def load_data(path):
    """Load CSV and ensure required columns exist."""
    df = pd.read_csv(path)
    required_cols = {"feature", "fold_change"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"CSV {path} must contain columns: {required_cols}")
    return df[["feature", "fold_change"]]


def main():
    parser = argparse.ArgumentParser(description="Scatter plot comparison of fold changes.")
    parser.add_argument("-p", "--path", required=True, help="Path to NEW dataset CSV")
    parser.add_argument("-b", "--baseline", required=True, help="Path to BASELINE dataset CSV")
    parser.add_argument("-t", "--title", required=True, help="Title for the plot")
    parser.add_argument("-o", "--output", default="scatter_plot.png", help="Output image file")

    args = parser.parse_args()

    # Load datasets
    df_new = load_data(args.path)
    df_old = load_data(args.baseline)

    # Merge on feature
    merged = pd.merge(df_old, df_new, on="feature", suffixes=("_old", "_new"))

    if merged.empty:
        raise ValueError("No overlapping features between the two datasets!")

    x = merged["fold_change_old"]
    y = merged["fold_change_new"]

    # Determine colors based on direction change
    # (Optional – points turning from down->up or up->down become orange)
    def classify(row):
        if row["fold_change_old"] < 1 and row["fold_change_new"] < 1:
            return "red"     # both downregulated
        elif row["fold_change_old"] > 1 and row["fold_change_new"] > 1:
            return "green"   # both upregulated
        else:
            return "orange"  # direction changed between datasets

    colors = merged.apply(classify, axis=1)

    # Create scatter plot
    plt.figure(figsize=(10, 8))
    plt.scatter(x, y, c=colors, alpha=0.7, edgecolor="black")

    # Identity line for reference
    min_val = min(x.min(), y.min())
    max_val = max(x.max(), y.max())
    plt.plot([min_val, max_val], [min_val, max_val], "k--", linewidth=1)

    plt.xlabel("Fold Change (Baseline Dataset)")
    plt.ylabel("Fold Change (New Dataset)")
    plt.title(args.title)

    # Legend
    from matplotlib.patches import Patch
    legend_patches = [
        Patch(color="red", label="Both Downregulated"),
        Patch(color="green", label="Both Upregulated"),
        Patch(color="orange", label="Direction Changed"),
    ]
    plt.legend(handles=legend_patches)

    plt.tight_layout()
    plt.savefig(args.output, dpi=300)
    print(f"Scatter plot saved to {args.output}")


if __name__ == "__main__":
    main()

