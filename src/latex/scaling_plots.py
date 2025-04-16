"""Generate box plots to show example-level variance in a dataset."""

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


def read_example_scores(file_path: Path) -> pd.DataFrame:
    """Read scores from a file."""
    with file_path.open() as rf:
        lines = rf.readlines()
    header = lines[0].split()
    data = [[float(item) for item in line.split()] for line in lines[1:]]
    return pd.DataFrame(data, columns=["id", *header])


def tabulate(
    dataset_config_name: str, split: str, results_dir: Path, plots_dir: Path
) -> None:
    """Generate a table with metrics for all systems."""
    results_dir = Path(results_dir)
    plots_dir = Path(plots_dir)  # path to write box plots
    transformers = ["Llama31_8B", "Llama31_70B_FP8"]
    method2suffix = {
        "Full-Context": "",
        "Hierarchical": "_Hierarchical",
        "Incremental": "_Incremental",
        "RAG": "_RAG_SFR",
    }

    systems = [(t, m) for t in transformers for m in method2suffix]

    # write all scores as a latex table
    filepaths = {
        f"{t}{method2suffix[m]}": results_dir
        / "{}_{}{}_{}_scores.txt".format(
            dataset_config_name, t, method2suffix[m], split
        )
        for t, m in systems
    }
    # load dataframe for each system and create a single dataframe
    all_scores = []
    for system, filepath in filepaths.items():
        if not filepath.exists():
            logger.warning(f"File {filepath} does not exist.")
            continue
        scores = pd.read_csv(filepath, sep=r"\s+")
        scores["system"] = system
        all_scores += [scores]
    all_scores = pd.concat(all_scores, axis=0, ignore_index=True)
    selected_cols = ["system", "rouge/rougeLsum", "a3cu/f1"]
    all_scores = all_scores[selected_cols]

    # write dataframe as latex table
    logger.info(
        all_scores.to_latex(
            index=False,
            caption=f"{dataset_config_name} {split} scores",
            float_format="%.1f",
        )
    )

    # create box plots
    system2df = {}
    for t, m in systems:
        filepath = results_dir / "{}_{}{}_{}_per_ex_scores.txt".format(
            dataset_config_name, t, method2suffix[m], split
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
    scores = pd.DataFrame(scores)

    plots_dir.mkdir(parents=True, exist_ok=True)
    for metric in ["rouge/rougeLsum", "a3cu/f1"]:
        plt.figure(figsize=(8, 6))
        sns.boxplot(data=scores, x="transformer", y=metric, hue="method")
        sns.despine(offset=10, trim=True)
        # save to output path
        out_path = plots_dir / "scaling_{}_{}_{}_boxplot.png".format(
            dataset_config_name, split, metric.split("/")[0]
        )
        plt.title(f"{dataset_config_name} | {split} | {metric}")
        plt.legend(loc="lower center", bbox_to_anchor=(0.5, 1.05), ncol=4)
        plt.savefig(out_path, dpi=500)


if __name__ == "__main__":
    fire.Fire(tabulate)
