"""
Track information loss in input compression-based systems.

Use a3cu/recall as the metric.
"""

from __future__ import annotations

import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import fire
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from loguru import logger

sns.set_style("darkgrid")

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


def read_best_recall(file_path: Path) -> list[float]:
    """Read best a3cu/recall score."""
    output = []
    with file_path.open() as rf:
        for ex_idx, line in enumerate(rf):
            data = json.loads(line)
            scores = data["intermediate_scores"]
            if isinstance(scores, dict):
                # flatten the values into a single list
                scores = [
                    item
                    for sublist in dict(
                        sorted(scores.items(), key=lambda x: int(x[0]))
                    ).values()
                    for item in sublist
                ]
            # skip last score, which is the final score for incremental and hierarchical
            if len(scores) == 1:
                logger.warning(
                    "only one score found in row {} of {}".format(ex_idx, file_path)
                )
                best = scores[0]  # just one document in the input
            else:
                best = max(scores[:-1])
            output += [best]
    return output


def prepare_scores(
    dataset_config_name: str,
    split: str,
    results_dir: Path,
) -> pd.DataFrame:
    """Load scores and create a dataframe."""
    systems = [(t, m) for t in transformer2prefix for m in method2suffix]
    system2best = {}
    system2final = {}
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
        final_scores = read_example_scores(filepath)["a3cu/recall"]
        system2final[(t, m)] = final_scores
        if m != "Full-Context":
            filepath = results_dir / "{}_{}{}_{}_inf_loss.jsonl".format(
                dataset_config_name,
                transformer2prefix[t],
                method2suffix[m],
                split,
            )
            if not filepath.exists():
                logger.warning(f"File {filepath} does not exist.")
                continue
            system2best[(t, m)] = read_best_recall(filepath)
    scores = defaultdict(list)
    for ex_idx in range(len(system2final[systems[0]])):
        for system in system2final:
            for _type in ["best", "final"]:
                scores["ex_id"] += [ex_idx]
                scores["Transformer"] += [system[0]]
                scores["Method"] += [system[1]]
                scores["Type"] += [_type]
                if _type == "final" or system[1] == "Full-Context":
                    scores["Recall"] += [system2final[system][ex_idx]]
                elif _type == "best":
                    scores["Recall"] += [system2best[system][ex_idx]]
    return pd.DataFrame(scores)


def aggregate_scores(scores: pd.DataFrame) -> str:
    """Write markdown table."""
    # create a markdown table for Full-Context, method, best and worst scores
    scores = scores.groupby(["Transformer", "Method", "Type"]).agg(
        {"Recall": ["mean", "std"]}
    )
    scores = scores.reset_index()
    scores.columns = [
        "Transformer",
        "Method",
        "Type",
        "mean",
        "std",
    ]
    scores = scores.pivot_table(
        index=["Transformer", "Method"],
        columns="Type",
        values=["mean", "std"],
    )
    # use type as the first level of the column index, and mean/std as the second level
    scores.columns = scores.columns.swaplevel(0, 1)
    scores = scores.sort_index(axis=1, level=0)
    scores = scores.reset_index()
    # for a given transformer, method and type, merge mean and std into a single value
    scores = scores.set_index(["Transformer", "Method"])
    scores.columns = [
        f"{col[0]}_{col[1]}" if col[1] else col[0] for col in scores.columns
    ]
    scores = scores.reindex(
        transformer2prefix.keys(),
        level=0,
    )
    scores = scores.reset_index()
    scores = scores.round(1)
    logger.info(scores)
    scores["Best"] = scores.apply(
        lambda row: "{}\\textsubscript{{$\\pm$ {}}}".format(
            row["best_mean"],
            row["best_std"],
        ),
        axis=1,
    )
    scores["Final"] = scores.apply(
        lambda row: "{}\\textsubscript{{$\\pm$ {}}}".format(
            row["final_mean"],
            row["final_std"],
        ),
        axis=1,
    )
    scores = scores.drop(columns=["best_mean", "best_std", "final_mean", "final_std"])
    scores = scores.set_index(["Transformer", "Method"])
    scores = scores.reindex(
        transformer2prefix.keys(),
        level=0,
    )
    return scores.reset_index()


def compute_loss(  # noqa: PLR0913
    dataset_config_name: str,
    split: str,
    results_dir: Path,
    output_dir: Path,
    plot: bool = False,
    sample: bool = False,
) -> None:
    """Get best and final information loss scores."""
    results_dir = Path(results_dir)
    output_dir = Path(output_dir)
    scores = prepare_scores(
        dataset_config_name=dataset_config_name,
        split=split,
        results_dir=results_dir,
    )
    agg_scores = aggregate_scores(scores)
    out_path = "{}_inf_loss_summary.tex"
    out_path = output_dir / out_path.format(dataset_config_name)
    with out_path.open("w") as wf:
        # replace ± with $\pm$ in agg_scores
        agg_scores = agg_scores.replace("±", r"$\\pm$", regex=True)
        agg_scores = agg_scores.replace("_", "-", regex=True)
        wf.write(agg_scores.to_latex(index=False))

    if not plot:
        # don't generate plots if output_dir is not provided
        return

    # sample 100 examples for easy visualization of line plots
    if sample:
        ex_ids = random.sample(scores["ex_id"].unique().tolist(), 100)
        scores = scores[scores["ex_id"].isin(ex_ids)]

    # print all rows, without hiding
    output_dir.mkdir(parents=True, exist_ok=True)
    for transformer in transformer2prefix:
        # compare full-context against other methods
        # includes both full-context and the other method
        for method in method2suffix:
            if method == "Full-Context":
                continue
            # create a line plot for Full-Context and method
            plt.figure(figsize=(12, 6))
            sns.lineplot(
                data=scores[
                    (scores["Transformer"] == transformer)
                    & (scores["Method"].isin(["Full-Context", method]))
                ],
                x="ex_id",
                y="Recall",
                hue="Method",
                style="Type",
            )
            if sample:
                out_path = "{}_{}_{}_recall_lineplot_sample.png"
                plt_title = "{} | {} | {} | sample"
            else:
                out_path = "{}_{}_{}_recall_lineplot.png"
                plt_title = "{} | {} | {}"
            out_path = output_dir / out_path.format(
                dataset_config_name,
                transformer,
                method,
            )
            plt.title(plt_title.format(dataset_config_name, transformer, method))
            plt.savefig(out_path, dpi=500)
    for method in ["RAG"]:
        if method == "Full-Context":
            continue
        plt.figure(figsize=(12, 6))
        sns.lineplot(
            data=scores[scores["Method"] == method],
            x="ex_id",
            y="Recall",
            hue="Transformer",
            style="Type",
        )
        if sample:
            out_path = "{}_{}_recall_lineplot_sample.png"
            plt_title = "{} | {} | sample"
        else:
            out_path = "{}_{}_recall_lineplot.png"
            plt_title = "{} | {}"
        out_path = output_dir / out_path.format(
            dataset_config_name,
            method,
        )
        plt.title(plt_title.format(dataset_config_name, method))
        plt.savefig(out_path, dpi=500)


if __name__ == "__main__":
    fire.Fire(compute_loss)
