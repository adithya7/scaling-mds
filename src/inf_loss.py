"""
Measure information loss at various stages of compression-based methods.

Use a3cu/recall.
"""

from __future__ import annotations

import json
import sys
from copy import deepcopy
from pathlib import Path

import fire
import numpy as np
from autoacu import A3CU
from loguru import logger
from tqdm import tqdm

logger.remove()
logger.add(
    sys.stdout,
    colorize=True,
    format="<m>{time:YYYY-MM-DD at HH:mm:ss}</m> | {level} | {message}",
)


def score(pred: list[str], tgt: list[str] | list[list[str]], a3cu: A3CU) -> list[float]:
    """Compute A3CU recall scores for intermediate outputs."""
    if isinstance(tgt[0], list):
        # first, check if all examples have the same number of references
        num_tgt = [len(item) for item in tgt]
        if np.mean(num_tgt) != np.max(num_tgt):
            logger.warning("found varying number of references, skipping A3CU")
            return None
        # for each reference, compute scores and average
        recall_scores, f1_scores = [], []
        for i in range(num_tgt[0]):
            ref = [item[i] for item in tgt]
            recall, _, f1 = a3cu.score(
                references=ref,
                candidates=pred,
                batch_size=32,
                verbose=False,
            )
            recall_scores += [recall]
            f1_scores += [f1]
        f1_scores = np.array(f1_scores)
        best_ref_indices = np.argmax(f1_scores, axis=0)
        recall_scores = [
            recall_scores[best_ref_idx][ex_idx]
            for ex_idx, best_ref_idx in enumerate(best_ref_indices)
        ]
    else:
        recall_scores, _, _ = a3cu.score(
            references=tgt,
            candidates=pred,
            batch_size=32,
            verbose=False,
        )
    return [round(item * 100, 1) for item in recall_scores]


def get_total(file_path: Path) -> int:
    """Get the total number of intermediate outputs."""
    count = 0
    with file_path.open() as rf:
        for line in rf:
            example = json.loads(line)
            if isinstance(example["intermediate_pred"], list):
                count += len(example["intermediate_pred"])
            elif isinstance(example["intermediate_pred"], dict):
                for v in example["intermediate_pred"].values():
                    count += len(v)
    return count


def inf_loss(
    model_config_name: str,
    dataset_config_name: str,
    split: str,
    artifacts_dir: str,
    output_dir: str,
) -> None:
    """Measure information loss in intermediate outputs."""
    artifacts_dir = Path(artifacts_dir)
    output_dir = Path(output_dir)

    model_pt = artifacts_dir / "huggingface/model/Yale-LILY/a3cu"
    a3cu = A3CU(model_pt=model_pt)

    pred_path = output_dir / f"{dataset_config_name}_{model_config_name}_{split}.jsonl"
    total_count = get_total(pred_path)
    pbar = tqdm(total=total_count, desc="computing information loss scores")
    output = []
    with pred_path.open() as rf:
        for line in rf:
            example = json.loads(line)
            out_example = deepcopy(example)
            if isinstance(example["intermediate_pred"], list):
                scores = score(
                    pred=example["intermediate_pred"],
                    tgt=[example["gold"]] * len(example["intermediate_pred"]),
                    a3cu=a3cu,
                )
                pbar.update(len(example["intermediate_pred"]))
                out_example["intermediate_scores"] = scores
            elif isinstance(example["intermediate_pred"], dict):
                out_example["intermediate_scores"] = {}
                for k, v in example["intermediate_pred"].items():
                    scores = score(
                        pred=v,
                        tgt=[example["gold"]] * len(v),
                        a3cu=a3cu,
                    )
                    pbar.update(len(v))
                    out_example["intermediate_scores"][k] = scores
            output += [out_example]
    pbar.close()
    output_path = (
        output_dir / f"{dataset_config_name}_{model_config_name}_{split}_inf_loss.jsonl"
    )
    with output_path.open("w") as wf:
        for item in output:
            wf.write(json.dumps(item) + "\n")


if __name__ == "__main__":
    fire.Fire(inf_loss)
