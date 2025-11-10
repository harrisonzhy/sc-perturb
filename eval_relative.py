import argparse
import pandas as pd
from cell_eval import score_agg_metrics

def main():
    parser = argparse.ArgumentParser(description="Compute relative aggregate metric scores.")
    parser.add_argument(
        "-u", "--user_input", required=True, help="Path to user agg_results.csv file"
    )
    parser.add_argument(
        "-b", "--base_input", required=True, help="Path to base agg_results.csv file"
    )
    parser.add_argument(
        "-o", "--output", default="./score.csv", help="Path to output CSV file (default: ./score.csv)"
    )
    args = parser.parse_args()

    results_user = pd.read_csv(args.user_input)
    results_base = pd.read_csv(args.base_input)

    score_agg_metrics(
        results_user=results_user,
        results_base=results_base,
        output=args.output,
    )

if __name__ == "__main__":
    main()

