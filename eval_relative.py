import argparse
import pandas as pd

import logging

import numpy as np
import polars as pl
from numpy.typing import NDArray

from cell_eval._types import MetricBestValue
from cell_eval.metrics import metrics_registry

logger = logging.getLogger(__name__)

def patched_score_agg_metrics(
    results_user: pl.DataFrame | str,
    results_base: pl.DataFrame | str,
    output: str | None = None,
    comparison_statistic: str = "mean",
) -> pl.DataFrame:
    """Scores the aggregated results between a user's model and a base model.

    Files are expected to be in CSV format.

    """

    if isinstance(results_user, str):
        logger.info(f"Reading user results from {results_user}")
        results_user = pl.read_csv(results_user)
    if isinstance(results_base, str):
        logger.info(f"Reading base results from {results_base}")
        results_base = pl.read_csv(results_base)

    if not results_user.columns.equals(results_base.columns):
        raise ValueError(
            f"Columns do not match: {results_user.columns} != {results_base.columns}"
        )
    if "statistic" not in results_user.columns:
        raise ValueError(
            "Missing 'statistic' column in agg results (likely wrong file input)"
        )

    # Determine the statistic to be compared
    comp_stats = results_user["statistic"].unique()
    if comparison_statistic not in comp_stats:
        raise ValueError(
            f"Comparison statistic '{comparison_statistic}' not found in results: {comp_stats}"
        )

    # Determine the necessary normalization strategy for each metric
    idx_norm_by_one = []
    idx_norm_by_zero = []
    metric_names = np.array(results_user.columns[1:])
    for idx, m in enumerate(metric_names):
        try:
            info = metrics_registry.get_metric(m)
            norm_type = info.best_value
            if norm_type == MetricBestValue.ZERO:
                idx_norm_by_zero.append(idx)
            elif norm_type == MetricBestValue.ONE:
                idx_norm_by_one.append(idx)
        except KeyError:
            logger.warning(f"Metric '{m}' not found in registry (skipping index {idx})")
    idx_norm_by_zero = np.array(idx_norm_by_zero)
    idx_norm_by_one = np.array(idx_norm_by_one)

    # Determine the row index for the comparison statistic
    u_row_idx = np.flatnonzero(results_user["statistic"] == comparison_statistic)[0]
    b_row_idx = np.flatnonzero(results_base["statistic"] == comparison_statistic)[0]
    
    def strip_meta(df):
        df = df.copy()
        # Drop 'statistic' if it's a column
        if 'statistic' in df.columns:
            df = df.drop(columns=['statistic'])
        # If 'statistic' was used as the index name, reset it away
        if df.index.name == 'statistic':
            df = df.reset_index(drop=True)
        return df

    scores_user = strip_meta(results_user).to_numpy()[u_row_idx]
    scores_base = strip_meta(results_base).to_numpy()[b_row_idx]
    
    logger.info(f"user score: {scores_user}")
    logger.info(f"base score: {scores_base}")
    
    metrics_by_zero = metric_names[idx_norm_by_zero]
    logger.info(
        f"Calculating norm by zero for {len(metrics_by_zero)} metrics: {', '.join(metrics_by_zero)}"
    )
    norm_by_zero = _calc_norm_by_zero(
        scores_user[idx_norm_by_zero],
        scores_base[idx_norm_by_zero],
    )

    metrics_by_one = metric_names[idx_norm_by_one]
    logger.info(
        f"Calculating norm by one for {len(metrics_by_one)} metrics: {', '.join(metrics_by_one)}"
    )
    norm_by_one = _calc_norm_by_one(
        scores_user[idx_norm_by_one],
        scores_base[idx_norm_by_one],
    )

    # Concatenate the two norm modes, replace nan, and clip lower bound to zero
    all_scores = np.concatenate([norm_by_zero, norm_by_one])
    all_scores[np.isnan(all_scores)] = 0.0
    all_scores = np.clip(all_scores, 0, None)

    # Build output dataframe
    results = pl.DataFrame(
        {
            "metric": np.concatenate(
                [metrics_by_zero, metrics_by_one, np.array(["avg_score"])]
            ),
            "from_baseline": np.concatenate([all_scores, [all_scores.mean()]]),
        }
    )

    if output is not None:
        logger.info(f"Writing results to {output}")
        results.write_csv(output)

    return results


def _calc_norm_by_zero(
    user: NDArray[np.float64],
    base: NDArray[np.float64],
) -> NDArray[np.float64]:
    return 1.0 - (user / base)


def _calc_norm_by_one(
    user: NDArray[np.float64],
    base: NDArray[np.float64],
) -> NDArray[np.float64]:
    return (user - base) / (1 - base)

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

    # Load CSVs
    results_user = pd.read_csv(args.user_input)
    results_base = pd.read_csv(args.base_input)

    # Check for column differences
    if not results_user.columns.equals(results_base.columns):
        common_cols = results_user.columns.intersection(results_base.columns)
        print("Common columns: ", common_cols)

        dropped_user = set(results_user.columns) - set(common_cols)
        dropped_base = set(results_base.columns) - set(common_cols)

        print("[WARNING] Column mismatch detected.")
        if dropped_user:
            print(f"  Dropping from user_input: {sorted(dropped_user)}")
        if dropped_base:
            print(f"  Dropping from base_input: {sorted(dropped_base)}")

        # Restrict to intersection
        results_user = results_user[common_cols]
        results_base = results_base[common_cols]
    else:
        common_cols = results_user.columns  # identical columns

    # Compute relative scores
    patched_score_agg_metrics(
        results_user=results_user,
        results_base=results_base,
        output=args.output,
    )

if __name__ == "__main__":
    main()

