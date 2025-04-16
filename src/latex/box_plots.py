"""Generate box plots to show example-level variance in a dataset."""

from __future__ import annotations

import random
import sys
from collections import defaultdict
from pathlib import Path

import fire
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from loguru import logger

logger.remove()
logger.add(
    sys.stdout,
    colorize=True,
    format="{message}",
)

transformer2prefix = {
    "Llama-3.1-8B": "Llama31_8B",
    "Llama-3.1-70B": "Llama31_70B_FP8",
    "Command-R": "CommandR",
    "Jamba-1.5-Mini": "Jamba15Mini",
}
method2suffix = {
    "Full-Context": "",
    "Hierarchical": "_Hierarchical",
    "Incremental": "_Incremental",
    "Retrieval": "_RAG_SFR",
}


def read_example_scores(file_path: Path) -> pd.DataFrame:
    """Read scores from a file."""
    with file_path.open() as rf:
        lines = rf.readlines()
    header = lines[0].split()
    data = [[float(item) for item in line.split()] for line in lines[1:]]
    return pd.DataFrame(data, columns=["id", *header])


def prepare_scores(
    systems: list[tuple[str, str]],
    dataset_config_name: str,
    split: str,
    results_dir: Path,
) -> pd.DataFrame:
    """Load scores and create a dataframe."""
    system2df = {}
    for t, m in systems:
        filepath = results_dir / "{}_{}{}_{}_per_ex_scores.txt".format(
            dataset_config_name,
            transformer2prefix[t],
            method2suffix[m],
            split,
        )
        if not filepath.exists():
            logger.warning(f"File {filepath} does not exist.")
            continue
        system2df[(t, m)] = read_example_scores(filepath)
    scores = defaultdict(list)
    for ex_idx in range(len(system2df[systems[0]])):
        for system, df in system2df.items():
            system_scores = df.iloc[ex_idx]
            scores["ex_id"] += [system_scores["id"]]
            scores["transformer"] += [system[0]]
            scores["method"] += [system[1]]
            scores["rouge/rougeLsum"] += [system_scores["rouge/rougeLsum"]]
            scores["a3cu/f1"] += [system_scores["a3cu/f1"]]
    return pd.DataFrame(scores)


def tabulate(  # noqa: PLR0913
    dataset_config_name: str,
    split: str,
    results_dir: Path,
    output_dir: Path,
    lineplots: bool = False,
    sample: bool = False,
) -> None:
    """Generate a table with metrics for all systems."""
    results_dir = Path(results_dir)
    output_dir = Path(output_dir)  # path to write box plots
    metrics = ["a3cu/f1"]
    systems = [(t, m) for t in transformer2prefix for m in method2suffix]
    scores = prepare_scores(
        systems=systems,
        dataset_config_name=dataset_config_name,
        split=split,
        results_dir=results_dir,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    for metric in metrics:
        plt.figure(figsize=(10, 8))
        # limit box plot to Full-Context and RAG
        sns.boxplot(data=scores, x="transformer", y=metric, hue="method")
        sns.despine(offset=10, trim=True)
        # save to output path
        out_path = "{}_{}_{}_boxplot.png"
        plt_title = "{} | {} | {}"
        out_path = output_dir / out_path.format(
            dataset_config_name,
            split,
            metric.split("/")[0],
        )
        plt.title(plt_title.format(dataset_config_name, split, metric), fontsize=16)
        plt.tick_params(axis="both", which="major", labelsize=16)
        plt.legend(loc="lower center", bbox_to_anchor=(0.5, 1.05), ncol=4, fontsize=16)
        plt.xlabel("Transformer", fontsize=16)
        plt.ylabel("A3CU/F1", fontsize=16)
        plt.tight_layout()
        plt.savefig(out_path)

    if not lineplots:
        return

    # sample 100 examples for easy visualization of line plots
    if sample:
        ex_ids = random.sample(scores["ex_id"].unique().tolist(), 100)
        scores = scores[scores["ex_id"].isin(ex_ids)]

    for metric in metrics:
        for transformer in transformer2prefix:
            # compares Full-Context against other methods
            # include both full-context and the other method
            for method in method2suffix:
                if method == "Full-Context":
                    continue
                plt.figure(figsize=(12, 6))
                sns.lineplot(
                    data=scores[
                        (scores["transformer"] == transformer)
                        & (scores["method"].isin(["Full-Context", method]))
                    ],
                    x="ex_id",
                    y=metric,
                    hue="method",
                )
                # save to output path
                if sample:
                    out_path = "{}_{}_{}_{}_{}_lineplot_sample.png"
                    plt_title = "{} | {} | {} | {} | {} | sample"
                else:
                    out_path = "{}_{}_{}_{}_{}_lineplot.png"
                    plt_title = "{} | {} | {} | {} | {}"
                out_path = output_dir / out_path.format(
                    dataset_config_name,
                    split,
                    metric.split("/")[0],
                    transformer,
                    method,
                )
                plt.title(
                    plt_title.format(
                        dataset_config_name,
                        split,
                        metric,
                        transformer,
                        method,
                    )
                )
                plt.savefig(out_path, dpi=500)

        for method in method2suffix:
            plt.figure(figsize=(12, 6))
            sns.lineplot(
                data=scores[scores["method"] == method],
                x="ex_id",
                y=metric,
                hue="transformer",
            )
            # save to output path
            if sample:
                out_path = "{}_{}_{}_{}_lineplot_sample.png"
                plt_title = "{} | {} | {} | {} | sample"
            else:
                out_path = "{}_{}_{}_{}_lineplot.png"
                plt_title = "{} | {} | {} | {}"
            out_path = output_dir / out_path.format(
                dataset_config_name,
                split,
                metric.split("/")[0],
                method,
            )
            plt.title(plt_title.format(dataset_config_name, split, metric, method))
            plt.savefig(out_path, dpi=500)


if __name__ == "__main__":
    fire.Fire(tabulate)
