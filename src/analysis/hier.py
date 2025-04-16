"""
Compare best intermediate summary against final summary.

Hierarchical and Incremental
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import fire
import nltk
import numpy as np
import pandas as pd
from loguru import logger

nltk.download("punkt_tab", quiet=True)

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
    "Gemini-1.5-Flash": "Gemini15Flash",
    "Gemini-1.5-Pro": "Gemini15Pro",
}
method2suffix = {
    "Full-Context": "",
    "Hierarchical": "_Hierarchical",
    "Incremental": "_Incremental",
    "Retrieval": "_RAG_SFR",
}


def load_data(file_path: Path) -> list[dict]:
    """Load inf_loss.jsonl."""
    output = []
    with file_path.open() as rf:
        for line in rf:
            data = json.loads(line)
            best_score, best_pred = 0.0, ""
            if isinstance(data["intermediate_scores"], dict):
                # hierarchical
                for k, v in data["intermediate_scores"].items():
                    for idx, score in enumerate(v):
                        if score > best_score:
                            best_score = score
                            best_pred = data["intermediate_pred"][k][idx]
                scores = [
                    item
                    for sublist in dict(
                        sorted(
                            data["intermediate_scores"].items(), key=lambda x: int(x[0])
                        )
                    ).values()
                    for item in sublist
                ]
            else:
                # incremental
                scores = data["intermediate_scores"]
                for idx, score in enumerate(scores):
                    if score > best_score:
                        best_score = score
                        best_pred = data["intermediate_pred"][idx]
            output += [
                {
                    "best_pred": best_pred,
                    "final_pred": data["pred"],
                    "gold": data["gold"],
                    "best_score": best_score,
                    "final_score": scores[-1],
                }
            ]
    return output


def load_pred(filepath: Path) -> list[dict]:
    """Load predictions."""
    gold, pred = [], []
    with filepath.open() as rf:
        for line in rf:
            data = json.loads(line)
            if isinstance(data["gold"], list):
                values = [len(nltk.word_tokenize(g)) for g in data["gold"]]
                gold += [np.mean(values)]
            else:
                gold += [len(nltk.word_tokenize(data["gold"]))]
            pred += [len(nltk.word_tokenize(data["pred"]))]
    return {"gold": np.mean(gold), "pred": np.mean(pred)}


def get_length_stats(data: list[dict]) -> dict:
    """Get length statistics for best and final summaries."""
    best, final, gold = [], [], []
    for item in data:
        best += [len(nltk.word_tokenize(item["best_pred"]))]
        final += [len(nltk.word_tokenize(item["final_pred"]))]
        if isinstance(item["gold"], list):
            gold += [np.mean([len(nltk.word_tokenize(g)) for g in item["gold"]])]
        else:
            gold += [len(nltk.word_tokenize(item["gold"]))]
    return {
        "best": np.mean(best),
        "final": np.mean(final),
        "gold": np.mean(gold),
    }


def main(
    dataset_config_name: str,
    split: str,
    results_dir: str,
    output_dir: str,
) -> None:
    """Write best and final summaries."""
    results_dir = Path(results_dir)
    output_dir = Path(output_dir)

    stats = defaultdict(list)
    for transformer in transformer2prefix:
        for method in method2suffix:
            if method not in ["Hierarchical", "Incremental"]:
                continue
            filepath = results_dir / "{}_{}{}_{}_inf_loss.jsonl".format(
                dataset_config_name,
                transformer2prefix[transformer],
                method2suffix[method],
                split,
            )
            if not filepath.exists():
                logger.warning(f"File {filepath} does not exist.")
                continue
            scores = load_data(filepath)
            out_filepath = output_dir / "{}_{}{}_{}_best_final.jsonl".format(
                dataset_config_name,
                transformer2prefix[transformer],
                method2suffix[method],
                split,
            )
            with out_filepath.open("w") as wf:
                for item in scores:
                    wf.write(json.dumps(item) + "\n")
            mean_len = get_length_stats(scores)
            stats["Transformer"] += [transformer]
            stats["Method"] += [method]
            stats["Best"] += [mean_len["best"]]
            stats["Final"] += [mean_len["final"]]
            stats["Gold"] += [mean_len["gold"]]
    stats = pd.DataFrame(stats)
    stats = stats.pivot_table(
        index="Transformer", columns="Method", values=["Best", "Final"]
    )
    stats = stats.swaplevel(axis=1)
    stats = stats.sort_index(axis=1)
    # sort rows by transformer
    stats = stats.reindex(transformer2prefix.keys(), level=0)
    stats = stats.round(0)
    logger.info(stats.to_latex(float_format="%.0f"))

    stats = defaultdict(list)
    for transformer in transformer2prefix:
        for method in method2suffix:
            filepath = results_dir / "{}_{}{}_{}.jsonl".format(
                dataset_config_name,
                transformer2prefix[transformer],
                method2suffix[method],
                split,
            )
            if not filepath.exists():
                logger.warning(f"File {filepath} does not exist.")
                continue
            data = load_pred(filepath)
            stats["Transformer"] += [transformer]
            stats["Method"] += [method]
            stats["Pred"] += [data["pred"]]
            stats["Gold"] += [data["gold"]]
    stats = pd.DataFrame(stats)
    stats = stats.pivot_table(
        index="Transformer", columns="Method", values=["Pred", "Gold"]
    )
    stats = stats.swaplevel(axis=1)
    stats = stats.sort_index(axis=1)
    # sort rows by transformer
    stats = stats.reindex(transformer2prefix.keys(), level=0)
    stats = stats.round(0)
    logger.info(stats)
    logger.info(stats.to_latex(float_format="%.0f"))


if __name__ == "__main__":
    fire.Fire(main)
