"""Measure example-level correlations between methods."""

from __future__ import annotations

import sys
from itertools import combinations
from pathlib import Path

import fire
import pandas as pd
from loguru import logger
from scipy.stats import spearmanr

logger.remove()
logger.add(
    sys.stdout,
    colorize=True,
    format="{message}",
)


def read_df(file_path: Path) -> pd.DataFrame:
    """Read scores from a file."""
    with file_path.open() as rf:
        lines = rf.readlines()
    header = lines[0].split()
    data = [[float(item) for item in line.split()] for line in lines[1:]]
    return pd.DataFrame(data, columns=["id", *header])


def compute_correlations(
    methods: list[str],
    scores: dict[str, pd.DataFrame],
    metrics: list[str],
) -> None:
    """Compute correlations between methods."""
    for metric in metrics:
        correlations = {}
        for method in methods:
            correlations[method] = {method: 1.0 for method in methods}
        for method1, method2 in combinations(methods, 2):
            statistic, p_value = spearmanr(
                scores[method1][metric], scores[method2][metric]
            )
            correlations[method1][method2] = statistic
            correlations[method2][method1] = statistic
        correlations = pd.DataFrame(correlations)
        # print as a string
        logger.info("Metric: {}", metric)
        logger.info(correlations.to_string(float_format="%.2f"))


def best_worst(
    methods: list[str],
    scores: dict[str, pd.DataFrame],
    metrics: list[str],
) -> None:
    """
    Best-worst votes among methods.

    Identify best and worst methods for each example, and store counts.
    """
    num_examples = len(scores[methods[0]])
    for metric in metrics:
        counter = {"best": {}, "worst": {}}
        example_ids = {"best": {}, "worst": {}}
        for method in methods:
            counter["best"][method] = 0
            counter["worst"][method] = 0
            example_ids["best"][method] = []
            example_ids["worst"][method] = []
        for ex_idx in range(num_examples):
            ex_scores = {method: scores[method].iloc[ex_idx] for method in methods}
            # get all the best methods
            best_score = max([ex_scores[method][metric] for method in methods])
            worst_score = min([ex_scores[method][metric] for method in methods])
            if best_score == worst_score:
                continue
            for method in methods:
                if ex_scores[method][metric] == best_score:
                    counter["best"][method] += 1
                    example_ids["best"][method] += [ex_idx]
                if ex_scores[method][metric] == worst_score:
                    counter["worst"][method] += 1
                    example_ids["worst"][method] += [ex_idx]
        counter = pd.DataFrame(counter)
        # print as a string
        logger.info("Metric: {}", metric)
        logger.info(counter.to_string())
        logger.info("Best example ids:")
        logger.info(example_ids["best"])
        logger.info("Worst example ids:")
        logger.info(example_ids["worst"])


def stat(
    model_name: str,
    dataset_config_name: str,
    split: str,
    results_dir: str,
) -> None:
    """Compute stats between methods."""
    results_dir = Path(results_dir)
    logger.info("-" * 80)
    logger.info("Model: {}", model_name)
    logger.info("Dataset: {}, Split: {}", dataset_config_name, split)
    method2suffix = {
        "full-context": "",
        "hierarchical": "_Hierarchical",
        "incremental": "_Incremental",
        "rag_sfr": "_RAG_SFR",
    }
    filepaths = {
        method: results_dir
        / f"{dataset_config_name}_{model_name}{suffix}_{split}_per_ex_scores.txt"
        for method, suffix in method2suffix.items()
    }
    # a mapping between method and its per-example scores for each metric
    scores = {method: read_df(filepath) for method, filepath in filepaths.items()}
    compute_correlations(
        methods=list(scores.keys()),
        scores=scores,
        metrics=["rouge/rougeLsum", "a3cu/f1"],
    )
    best_worst(
        methods=list(scores.keys()),
        scores=scores,
        metrics=["a3cu/f1"],
    )


if __name__ == "__main__":
    fire.Fire(stat)
