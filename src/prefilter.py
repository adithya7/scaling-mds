"""
Prefilter dataset to include only relevant documents.

Tested for BackgroundSumm.
"""

from __future__ import annotations

import sys
from pathlib import Path
from pprint import pformat
from typing import TYPE_CHECKING

import fire
import nltk
from datasets import DatasetDict, load_dataset
from loguru import logger
from tqdm import tqdm

from config_utils import get_dataset_config, get_retriever_config
from retriever_utils import HFRetriever

if TYPE_CHECKING:
    from datasets import Dataset

    from configs.datasets import SummDataset
    from configs.retriever import BaseRetrieverConfig

logger.remove()
logger.add(
    sys.stdout,
    colorize=True,
    format="<m>{time:YYYY-MM-DD at HH:mm:ss}</m> | {level} | {message}",
)

nltk.download("punkt_tab", quiet=True)


def preprocess_rag(
    dataset: Dataset,
    dataset_config: SummDataset,
    retriever_config: BaseRetrieverConfig,
    max_segment_tokens: int,
    max_input_tokens: int,
) -> list[list[str]]:
    """Preprocess documents for RAG."""
    retriever = HFRetriever(retriever_config)
    # truncate documents to max_segment_tokens
    truncated_docs = []
    for idx in tqdm(range(len(dataset)), colour="blue", desc="truncation"):
        truncated_docs += [
            [
                retriever.truncate_doc(doc, max_segment_tokens)
                for doc in dataset[idx][dataset_config.doc_key]
            ]
        ]
    # prepare inputs
    output = []
    for idx in tqdm(range(len(dataset)), colour="red", desc="retrieval"):
        query = (
            dataset[idx][dataset_config.query_key]
            if dataset_config.query_key
            else dataset_config.default_query
        )
        # truncate each document to max_segment_tokens
        selected_segments = retriever.retrieve_docs(
            truncated_docs[idx],
            query,
            max_input_tokens=max_input_tokens,
            max_segment_tokens=max_segment_tokens,
        )
        output += [selected_segments]

    return output


def flatten_docs(example: dict, doc_key: str) -> dict:
    """Flatten docs in each example."""
    if isinstance(example[doc_key][0], list):
        example[doc_key] = [doc for docs in example[doc_key] for doc in docs]
    return example


def prefilter(  # noqa: PLR0913
    retriever_config_name: str,
    dataset_config_name: str,
    split: str,
    max_segment_tokens: int,
    max_input_tokens: int,
    artifacts_dir: str,
) -> None:
    """Get summary predictions."""
    artifacts_dir = Path(artifacts_dir)

    # setup configs
    if retriever_config_name not in ["SFREmbedding", "E5RoPE"]:
        msg = f"retriever {retriever_config_name} not implemented"
        raise NotImplementedError(msg)
    retriever_config = get_retriever_config(retriever_config_name)
    dataset_config = get_dataset_config(dataset_config_name)
    dataset_config.path = str(artifacts_dir / dataset_config.path)
    retriever_config.model_name_or_path = (
        artifacts_dir / "huggingface/model" / retriever_config.model_name_or_path
    )
    logger.info(f"\nmodel config: {pformat(retriever_config)}")
    logger.info(f"\ndataset config: {pformat(dataset_config)}")
    if not hasattr(dataset_config, "query_key"):
        msg = "query_key not found in dataset config"
        raise ValueError(msg)

    # load dataset
    logger.info("{} | {} | {}", dataset_config.path, dataset_config.name, split)
    dataset = load_dataset(
        path=dataset_config.path,
        name=dataset_config.name,
        split=split,
        trust_remote_code=True,
    )

    # flatten docs in each example of the dataset
    dataset = dataset.map(
        flatten_docs,
        batched=False,
        desc="flattening docs",
        load_from_cache_file=False,
        fn_kwargs={"doc_key": dataset_config.doc_key},
    )
    # setup retriever
    # for each example, a list of segments selected by the retriever
    intermediate_outputs = preprocess_rag(
        dataset=dataset,
        dataset_config=dataset_config,
        retriever_config=retriever_config,
        max_segment_tokens=max_segment_tokens,
        max_input_tokens=max_input_tokens,
    )
    # update doc_key for each example in the dataset with intermediate_outputs
    dataset = dataset.map(
        lambda example, idx: {dataset_config.doc_key: intermediate_outputs[idx]},  # noqa: ARG005
        with_indices=True,
    )
    # remap dataset to a DatasetDict
    dataset = DatasetDict({split: dataset})
    # save dataset (to artifacts directory)
    out_path = (
        artifacts_dir
        / "misc"
        / "{}_prefiltered_{}".format(dataset_config.name, retriever_config_name)
    )
    dataset.save_to_disk(out_path)


if __name__ == "__main__":
    fire.Fire(prefilter)
