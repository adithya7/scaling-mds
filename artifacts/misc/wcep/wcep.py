"""HF dataset loading script for WCEP dataset."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import ClassVar, Iterator

import datasets

DATA = Path(os.getenv("HF_DATA_PATH", None)) / "misc/wcep"

_VERSION = "1.0.1"
_DESCRIPTION = "WCEP dataset"
_DATA = DATA / "ghalandari-etal-acl-2020"
_HOMEPAGE = "https://github.com/complementizer/wcep-mds-dataset"
_TASK_NAME = "wcep"
_TASK_DESCRIPTION = "WCEP dataset"


class WCEPConfig(datasets.BuilderConfig):
    """BuilderConfig for WCEP."""

    def __init__(self: WCEPConfig, features: list, **kwargs: dict) -> None:
        """Init config."""
        super().__init__(version=datasets.Version(_VERSION), **kwargs)
        self.features = features


class WCEP(datasets.GeneratorBasedBuilder):
    """WCEP dataset."""

    VERSION = datasets.Version(_VERSION)
    BUILDER_CONFIGS: ClassVar[list] = [
        WCEPConfig(
            name=_TASK_NAME,
            description=_TASK_DESCRIPTION,
            features=["document", "summary", "id"],
        ),
    ]

    def _info(self: WCEP) -> datasets.DatasetInfo:
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "document": datasets.Sequence(datasets.Value("string")),
                    "summary": datasets.Value("string"),
                    "id": datasets.Value("int32"),
                },
            ),
        )

    def _split_generators(
        self: WCEP,
        dl_manager: datasets.DownloadManager,  # noqa: ARG002
    ) -> list:
        filepath = {
            "train": _DATA / "train.jsonl",
            "val": _DATA / "val.jsonl",
            "test": _DATA / "test.jsonl",
        }
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"filepath": filepath["train"]},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={"filepath": filepath["val"]},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={"filepath": filepath["test"]},
            ),
        ]

    @staticmethod
    def unicode_to_ascii(txt: str) -> str:
        """Convert special unicode characters to ascii."""
        return (
            txt.replace("”", '"')
            .replace("“", '"')
            .replace("’", "'")  # noqa: RUF001
            .replace("‘", "'")  # noqa: RUF001
        )

    def _generate_examples(self: WCEP, filepath: Path) -> Iterator[tuple[int, dict]]:
        with filepath.open() as rf:
            for idx, line in enumerate(rf):
                data = json.loads(line)
                documents = [
                    self.unicode_to_ascii(article["text"])
                    for article in data["articles"]
                ]
                summary = self.unicode_to_ascii(data["summary"])
                yield (
                    idx,
                    {
                        "document": documents,
                        "summary": summary,
                        "id": data["id"],
                    },
                )
