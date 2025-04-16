"""
HF dataset script for background summarization from news articles.

Background summaries from Pratapa et al., 2023: https://huggingface.co/datasets/adithya7/background-summaries
Documents from Timeline17, Crisis, and SocialTimeline datasets.

Summary: background summary on a given timestamp. (multiple references)
Documents: news articles published on or before the timestamp.
Query: update summary on the timestamp. (multiple, as summarized by the annotators)
"""

from __future__ import annotations

import json
import os
import re
from collections import defaultdict
from datetime import date
from pathlib import Path
from typing import ClassVar, Iterator

import datasets
import pandas as pd

DATA = Path(os.getenv("HF_DATA_PATH", None)) / "misc/timeline_summ"

_VERSION = "1.0.0"
_DESCRIPTION = "Background summarization from news articles"
_BACKGROUND_SUMM_URL = DATA / "pratapa-etal-emnlp-2023/background-summaries"
_TIMELINE17_URL = DATA / "tran-etal-www-2013/Timeline17/Data"
_CRISIS_URL = DATA / "tran-etal-ecir-2015"
_SOCIALTIMELINE_URL = DATA / "wang-etal-naacl-2015/socialtimeline/events"
# list of document paths for each event
EVENT2DOCS = {
    "swine_flu": [
        _TIMELINE17_URL / "H1N1_bbc/InputDocs",
        _TIMELINE17_URL / "H1N1_guardian/InputDocs",
        _TIMELINE17_URL / "H1N1_reuters/InputDocs",
    ],
    "financial_crisis": [
        _TIMELINE17_URL / "Finan_washingtonpost/InputDocs",
    ],
    "iraq_war": [
        _TIMELINE17_URL / "IraqWar_guardian/InputDocs",
    ],
    "haitian_earthquake": [
        _TIMELINE17_URL / "haiti_bbc/InputDocs",
    ],
    "mj_death": [
        _TIMELINE17_URL / "MJ_bbc/InputDocs",
    ],
    "bp_oil_spill": [
        _TIMELINE17_URL / "bpoil_bbc/InputDocs",
        _TIMELINE17_URL / "bpoil_foxnews/InputDocs",
        _TIMELINE17_URL / "bpoil_guardian/InputDocs",
        _TIMELINE17_URL / "bpoil_reuters/InputDocs",
        _TIMELINE17_URL / "bpoil_washingtonpost/InputDocs",
    ],
    "nsa_leak": [
        _SOCIALTIMELINE_URL / "nsa/cnn/articles",
        _SOCIALTIMELINE_URL / "nsa/nyt/articles",
    ],
    "gaza_conflict": [
        _SOCIALTIMELINE_URL / "gaza/bbc/articles",
        _SOCIALTIMELINE_URL / "gaza/cnn/articles",
        _SOCIALTIMELINE_URL / "gaza/nyt/articles",
    ],
    "mh370_disappearance": [
        _SOCIALTIMELINE_URL / "mh370/bbc/articles",
        _SOCIALTIMELINE_URL / "mh370/cnn/articles",
        _SOCIALTIMELINE_URL / "mh370/nyt/articles",
    ],
    "yemen_crisis": [
        _CRISIS_URL / "yemen/public/content",
    ],
    "ukraine_conflict": [
        _SOCIALTIMELINE_URL / "ukraine/bbc/articles",
        _SOCIALTIMELINE_URL / "ukraine/cnn/articles",
        _SOCIALTIMELINE_URL / "ukraine/nyt/articles",
    ],
    "libyan_war": [
        _TIMELINE17_URL / "LibyaWar_cnn/InputDocs",
        _TIMELINE17_URL / "LibyaWar_reuters/InputDocs",
        _CRISIS_URL / "libya/public/content",
    ],
    "egyptian_crisis": [
        _TIMELINE17_URL / "EgyptianProtest_cnn/InputDocs",
        _CRISIS_URL / "egypt/public/content",
    ],
    "syrian_crisis": [
        _TIMELINE17_URL / "SyrianCrisis_bbc/InputDocs",
        _TIMELINE17_URL / "SyrianCrisis_reuters/InputDocs",
        _CRISIS_URL / "syria/public/content",
    ],
}
DOC_LIMIT = 100000


class TimelineSummConfig(datasets.BuilderConfig):
    """BuilderConfig for timeline summarization of events from news articles."""

    def __init__(self: TimelineSummConfig, features: list, **kwargs: dict) -> None:
        """Init config."""
        super().__init__(version=datasets.Version(_VERSION), **kwargs)
        self.features = features


class TimelineSumm(datasets.GeneratorBasedBuilder):
    """News event timeline summarization dataset."""

    VERSION = datasets.Version(_VERSION)
    BUILDER_CONFIGS: ClassVar[list] = [
        TimelineSummConfig(
            name="update",
            description="Summarize today's articles into an update summary.",
            features=["articles", "summary", "event", "timestamp"],
        ),
        TimelineSummConfig(
            name="background",
            description="Summarize past articles into a background summary.",
            features=["past_articles", "summary", "event", "timestamp", "query"],
        ),
        TimelineSummConfig(
            name="background_short",
            description="Summarize past updates into a background summary.",
            features=["past_updates", "summary", "event", "timestamp", "query"],
        ),
    ]

    def _info(self: TimelineSumm) -> datasets.DatasetInfo:
        features = {}
        for feature in self.config.features:
            if feature in ["articles", "summary"]:
                features[feature] = datasets.Sequence(datasets.Value("string"))
            elif feature in ["past_articles", "past_updates"]:
                features[feature] = datasets.Sequence(
                    datasets.Sequence(datasets.Value("string"))
                )
            else:
                features[feature] = datasets.Value("string")
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(features),
        )

    def _split_generators(
        self: TimelineSumm,
        dl_manager: datasets.DownloadManager,  # noqa: ARG002
    ) -> list:
        """Generate splits."""
        filepaths = {
            "train": _BACKGROUND_SUMM_URL / "splits" / "train.txt",
            "dev": _BACKGROUND_SUMM_URL / "splits" / "dev.txt",
            "test": _BACKGROUND_SUMM_URL / "splits" / "test.txt",
            "events": _BACKGROUND_SUMM_URL / "events",
        }
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "splits_path": filepaths["train"],
                    "background_summ_path": filepaths["events"],
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "splits_path": filepaths["dev"],
                    "background_summ_path": filepaths["events"],
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "splits_path": filepaths["test"],
                    "background_summ_path": filepaths["events"],
                },
            ),
        ]

    @staticmethod
    def _format_document(timestamp: str, doc: str) -> str:
        """Format document to include timestamp."""
        return "Date: {}, {}".format(
            date.fromisoformat(timestamp).strftime("%B %d, %Y"),
            re.sub(r"[\s+|\n+]", " ", doc).strip(),
        )

    def _generate_examples(  # noqa: PLR0912
        self: TimelineSumm,
        splits_path: Path,
        background_summ_path: Path,
    ) -> Iterator[tuple[int, dict]]:
        """Generate examples."""
        # load documents for each date and event
        with splits_path.open() as rf:
            event_names = [line.strip() for line in rf]

        data_idx = 0
        annotators = ["annotator1", "annotator2", "annotator3"]
        for event in event_names:
            # load update and background summaries from the three annotators
            ts2update, ts2background = defaultdict(list), defaultdict(list)
            for ann in annotators:
                # load tsv path
                tsv_path = background_summ_path / event / f"{ann}.tsv"
                background_df = pd.read_csv(tsv_path, sep="\t")
                background_df = background_df.fillna("")
                for row in background_df.itertuples():
                    ts = row.Date.strip("[]")
                    update = row.Update.replace("\\n", " ")
                    update = re.sub(r"[ ]+", r" ", update).strip()
                    background = row.Background.replace("\\n", " ")
                    background = re.sub(r"[ ]+", r" ", background).strip()
                    ts2update[ts] += [update]
                    ts2background[ts] += [background]

            """
            for each timestamp with an update/background summary, compile articles.
            """
            past_articles, past_updates = [], []
            past_text_updates = []  # just the update text, no timestamps
            for ts in sorted(ts2update.keys()):
                # iterate through event timestamps in chronological order.
                # load articles from the current timestamp
                ts_articles = []
                file_paths = EVENT2DOCS[event]
                for src_path in file_paths:
                    if "tran-etal-www-2013" in str(src_path):
                        for file_path in (src_path / ts).glob("*.txt"):
                            with file_path.open(errors="ignore") as rf:
                                ts_articles += [self._format_document(ts, rf.read())]
                    elif "tran-etal-ecir-2015" in str(src_path):
                        for file_path in (src_path / ts).glob("*.cont"):
                            with file_path.open(errors="ignore") as rf:
                                ts_articles += [self._format_document(ts, rf.read())]
                    elif "wang-etal-naacl-2015" in str(src_path):
                        for file_path in src_path.iterdir():
                            with file_path.open() as rf:
                                file_data = json.load(rf)
                                file_ts = date.fromisoformat(
                                    file_data["timestamp"].split(" ")[0]
                                ).isoformat()
                                if file_ts == ts:
                                    ts_articles += [
                                        self._format_document(ts, file_data["body"])
                                    ]

                # pick the longest update as the query
                query = ts2update[ts][0]
                for update in ts2update[ts][1:]:
                    if len(update) > len(query):
                        query = update
                if self.config.name == "update" and len(ts_articles) != 0:
                    yield (
                        data_idx,
                        {
                            "event": event,
                            "timestamp": ts,
                            "articles": ts_articles,
                            "summary": ts2update[ts],
                        },
                    )
                    data_idx += 1
                elif self.config.name == "background" and len(past_articles) != 0:
                    yield (
                        data_idx,
                        {
                            "event": event,
                            "timestamp": ts,
                            "query": query,
                            "past_articles": past_articles,
                            "summary": ts2background[ts],
                        },
                    )
                    data_idx += 1
                elif self.config.name == "background_short" and len(past_updates) != 0:
                    yield (
                        data_idx,
                        {
                            "event": event,
                            "timestamp": ts,
                            "query": query,
                            "past_updates": past_updates,
                            "summary": ts2background[ts],
                        },
                    )
                    data_idx += 1

                # update past articles
                if len(ts_articles) > 0:
                    # limit to maximum of 5 articles per timestamp
                    past_articles += [ts_articles[:5]]
                # include all past updates, even in the case of no articles.
                # useful for background summ short version.
                past_updates += [
                    [self._format_document(ts, _update) for _update in ts2update[ts]]
                ]
                past_text_updates += [ts2update[ts]]
