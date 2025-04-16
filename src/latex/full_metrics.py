"""Generate a table with metrics for all systems."""

import sys
from collections import defaultdict
from pathlib import Path

import fire
import pandas as pd
from loguru import logger

logger.remove()
logger.add(
    sys.stdout,
    colorize=True,
    format="{message}",
)

transformer2prefix = {
    "Llama-3.1-8B": "Llama31_8B",
    "Llama-3-8B": "Llama3_8B",
    "Llama-3.1-70B": "Llama31_70B_FP8",
    "Llama-3-70B": "Llama3_70B",
    "Command-R": "CommandR",
    "Jamba-1.5-Mini": "Jamba15Mini",
    "Jamba-1.5-Mini-Grounded": "Jamba15MiniGrounded",
    "Gemini-1.5-Flash": "Gemini15Flash",
    "Gemini-1.5-Pro": "Gemini15Pro",
}
method2suffix = {
    "Full-Context": "",
    "Hierarchical": "_Hierarchical",
    "Hierarchical-8K": "_Hierarchical_8K",
    "Hierarchical-16K": "_Hierarchical_16K",
    "Hierarchical-32K": "_Hierarchical_32K",
    "Incremental": "_Incremental",
    "Retrieval": "_RAG_SFR",
    "Retrieval_E5": "_RAG_E5",
}


def tabulate(
    dataset_config_name: str,
    split: str,
    results_dir: Path,
    output_dir: Path,
) -> None:
    """Generate a table with metrics for all systems."""
    results_dir = Path(results_dir)
    output_dir = Path(output_dir)
    scores = defaultdict(list)
    for transformer in transformer2prefix:
        for method in method2suffix:
            file_path = results_dir / "{}_{}_{}_scores.txt".format(
                dataset_config_name,
                transformer2prefix[transformer] + method2suffix[method],
                split,
            )
            if not file_path.exists():
                logger.warning("File does not exist: {}", file_path)
                continue
            system_scores = pd.read_csv(file_path, sep=r"\s+")
            scores["Transformer"] += [transformer]
            scores["Method"] += [method]
            scores["ROUGE-1"] += [system_scores["rouge/rouge1"].to_numpy()[0]]
            scores["ROUGE-2"] += [system_scores["rouge/rouge2"].to_numpy()[0]]
            scores["ROUGE-L"] += [system_scores["rouge/rougeL"].to_numpy()[0]]
            scores["ROUGE-Lsum"] += [system_scores["rouge/rougeLsum"].to_numpy()[0]]
            scores["A3CU/Recall"] += [system_scores["a3cu/recall"].to_numpy()[0]]
            scores["A3CU/Precision"] += [system_scores["a3cu/precision"].to_numpy()[0]]
            scores["A3CU/F1"] += [system_scores["a3cu/f1"].to_numpy()[0]]
    scores = pd.DataFrame(scores)
    scores = scores.replace("_", "-", regex=True)
    logger.info(scores)
    # write dataframe as latex table
    out_path = output_dir / "{}_{}_scores.tex".format(dataset_config_name, split)
    with out_path.open("w") as wf:
        wf.write(scores.to_latex(index=False, float_format="%.1f"))


if __name__ == "__main__":
    fire.Fire(tabulate)
