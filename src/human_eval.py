"""
Prepare data for human evaluation.

We rank the four system summaries on a best to worst scale.
To rank, we match them against the reference summary.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import fire


def load_jsonl(file_path: Path) -> list[dict]:
    """Load a JSON file."""
    data = []
    with file_path.open("r") as rf:
        for idx, line in enumerate(rf):
            row = json.loads(line)
            data += [
                {
                    "idx": idx,
                    "question": row["question"],
                    "pred": row["pred"].replace("\n", " "),
                    "gold": row["gold"].replace("\n", " "),
                }
            ]
    return data


def main(dir_path: str, output_path: str) -> None:
    """Write data files for human evaluation."""
    systems = [
        "Llama31_8B",
        "Llama31_8B_Hierarchical",
        "Llama31_8B_Incremental",
        "Llama31_8B_RAG_SFR",
    ]
    sys2data = {}
    for sys_name in systems:
        file_path = Path(dir_path) / f"SummHay_{sys_name}_test.jsonl"
        sys2data[sys_name] = load_jsonl(file_path)
    # collate all system predictions into one dictionary object
    data = []
    num_examples = len(sys2data[systems[0]])
    for idx in range(num_examples):
        row = {
            "gold": sys2data[systems[0]][idx]["gold"],
            "question": sys2data[systems[0]][idx]["question"],
            "idx": idx,
        }
        for sys_name in systems:
            row[sys_name] = sys2data[sys_name][idx]["pred"]
        data += [row]

    random.seed(31)
    random.shuffle(data)

    # write data to one file
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    data_path_wf = (output_path / "human_eval_data.tsv").open("w")
    id_path_wf = (output_path / "human_eval_ids.tsv").open("w")

    # write header
    id_path_wf.write(
        "\t".join(["idx"] + ["System " + str(i) for i in range(1, len(systems) + 1)])
        + "\n"
    )
    data_path_wf.write(
        "\t".join(
            ["idx", "question", "gold"]
            + ["System " + str(i) for i in range(1, len(systems) + 1)]
        )
        + "\n"
    )
    for row in data:
        # shuffle system names in each row, create new list
        shuffled_systems = systems.copy()
        random.shuffle(shuffled_systems)
        id_path_wf.write(
            "\t".join(
                [
                    str(row["idx"]),
                    *shuffled_systems,
                ],
            )
        )
        data_path_wf.write(
            "\t".join(
                [
                    str(row["idx"]),
                    row["question"],
                    row["gold"],
                ]
                + [row[sys_name] for sys_name in shuffled_systems],
            )
        )
        id_path_wf.write("\n")
        data_path_wf.write("\n")
    id_path_wf.close()
    data_path_wf.close()


if __name__ == "__main__":
    fire.Fire(main)
