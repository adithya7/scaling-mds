"""
Score system summaries.

Suports ROUGE, AutoACU (A2CU, A3CU)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import fire
import pandas as pd
from loguru import logger

from metrics import compute_summ_metrics

logger.remove()
logger.add(
    sys.stdout,
    colorize=True,
    format="<m>{time:YYYY-MM-DD at HH:mm:ss}</m> | {level} | {message}",
)


def score(
    model_config_name: str,
    dataset_config_name: str,
    split: str,
    artifacts_dir: str,
    output_dir: str,
) -> None:
    """Score system summaries."""
    artifacts_dir = Path(artifacts_dir)
    output_dir = Path(output_dir)

    pred_path = output_dir / f"{dataset_config_name}_{model_config_name}_{split}.jsonl"
    pred, tgt = [], []
    with pred_path.open() as rf:
        for line in rf:
            data = json.loads(line)
            pred += [data["pred"]]
            tgt += [data["gold"]]
    logger.info("pred: {}, tgt: {}", len(pred), len(tgt))
    scores, per_ex_scores = compute_summ_metrics(
        pred=pred, tgt=tgt, artifacts_dir=artifacts_dir
    )
    # write corpus-level scores
    scores_path = (
        output_dir / f"{dataset_config_name}_{model_config_name}_{split}_scores.txt"
    )
    with scores_path.open("w") as wf:
        wf.write(
            pd.DataFrame.from_dict(scores, orient="index", columns=[model_config_name])
            .transpose()
            .to_string(),
        )
        wf.write("\n")
    # write per example scores
    per_ex_score_path = (
        output_dir
        / f"{dataset_config_name}_{model_config_name}_{split}_per_ex_scores.txt"
    )
    with per_ex_score_path.open("w") as wf:
        wf.write(
            pd.DataFrame.from_dict(per_ex_scores, orient="index")
            .transpose()
            .to_string(),
        )
        wf.write("\n")


if __name__ == "__main__":
    fire.Fire(score)
