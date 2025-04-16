"""Collate results from best-worst ratings from human evaluation."""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import fire
from loguru import logger

logger.remove()
logger.add(
    sys.stdout,
    colorize=True,
    format="<m>{time:YYYY-MM-DD at HH:mm:ss}</m> | {level} | {message}",
)


def load_tsv(file_path: Path) -> list[dict]:
    """Load a TSV file."""
    data = []
    with file_path.open("r") as rf:
        for idx, line in enumerate(rf):
            if idx == 0:
                header = line.strip("\n").split("\t")
            else:
                row = line.strip("\n").split("\t")
                if len(header) != len(row):
                    logger.warning("stopping at row {} due to mismatched columns", idx)
                    break
                data += [dict(zip(header, row))]
    return data


def main(id_path: str, ratings_path: str) -> None:
    """Collate results from best-worst ratings from human evaluation."""
    id_data = load_tsv(Path(id_path))
    ratings_data = load_tsv(Path(ratings_path))

    best = defaultdict(int)
    worst = defaultdict(int)
    for idx, row in enumerate(ratings_data):
        id_row = id_data[idx]
        logger.info(id_row)
        if row["Best"] != "":
            for system in row["Best"].split(", "):
                best[id_row[system]] += 1
        if row["Worst"] != "":
            for system in row["Worst"].split(", "):
                worst[id_row[system]] += 1

    # plot a table with system names as rows, and best and worst counts as columns
    logger.info("System\tBest\tWorst")
    for system in best:
        logger.info(f"{system}\t{best[system]}\t{worst[system]}")


if __name__ == "__main__":
    fire.Fire(main)
