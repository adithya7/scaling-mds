"""Data related utils."""

from __future__ import annotations

from typing import TYPE_CHECKING

import nltk
from datasets import Dataset, load_dataset, load_from_disk
from loguru import logger

if TYPE_CHECKING:
    from transformers import AutoTokenizer

    from configs.datasets import SummDataset
    from configs.models import BaseModel


def truncate_helper(
    docs: list[str],
    max_inp_tokens: int,
    tokenizer: AutoTokenizer,
    min_doc_tokens: int | None = None,
) -> list[str]:
    """
    Truncate documents within a collection to `max_inp_tokens`.

    Longest documents are truncated first.
    All documents in the collection are related to each other.

    If min_doc_tokens is provided, documents aren't truncated below that value.
    Instead, remove the last documents to fit the budget.
    """
    toks = tokenizer(docs, add_special_tokens=False).input_ids
    doc_lengths = [len(x) for x in toks]

    if min_doc_tokens:
        # remove documents such that the rest fit into the budget
        # this is to avoid very short documents.
        max_num_docs = max_inp_tokens // min_doc_tokens
        if len(docs) > max_num_docs:
            toks = toks[:max_num_docs]
            doc_lengths = doc_lengths[:max_num_docs]

    indexed_lst = sorted([(val, i) for i, val in enumerate(doc_lengths)])
    result = []
    remaining_sum = max_inp_tokens
    for i in range(len(indexed_lst)):
        avg = remaining_sum // (len(indexed_lst) - i)
        if indexed_lst[i][0] <= avg:
            result.append((indexed_lst[i][0], indexed_lst[i][1]))
            remaining_sum -= indexed_lst[i][0]
        else:
            result.append((avg, indexed_lst[i][1]))
            remaining_sum -= avg
    result.sort(key=lambda x: x[1])
    truncated_doc_lengths = [val for val, _ in result]
    # limits tokens in each document to the above values,
    # but only truncate at sentence boundaries
    truncated_docs = []
    for doc, doc_tok_limit in zip(docs, truncated_doc_lengths):
        sents = nltk.sent_tokenize(doc)
        truncated_doc = ""
        for sent in sents:
            if (
                len(
                    tokenizer.encode(
                        truncated_doc + " " + sent, add_special_tokens=False
                    )
                )
                > doc_tok_limit
            ):
                break
            truncated_doc += " " + sent
        truncated_docs += [truncated_doc]
    return truncated_docs


def preprocess_example(  # noqa: PLR0913
    example: dict,
    tokenizer: AutoTokenizer,
    max_inp_tokens: int,
    max_summary_tokens: int,
    doc_key: str,
    min_doc_tokens: int | None = None,
) -> dict:
    r"""
    Truncate documents to max_length, and return a list of documents.

    - If a list of list of documents is provided, budget equally to each list of docs.
        This is useful for timeline summarization datasets.
    - Within a list of topically related docs, truncate the longest documents first.
    """
    docs = example[doc_key]
    # 100 tokens is a buffer for instructions etc.,
    adjusted_max_inp_tokens = max_inp_tokens - max_summary_tokens - 100
    # check for minimum document length
    if isinstance(docs[0], list):
        if min_doc_tokens and (adjusted_max_inp_tokens // len(docs)) < min_doc_tokens:
            # we have skip some timestamps to allow at least one document in each topic
            # keep most recent timestamps
            docs = docs[-1 * (adjusted_max_inp_tokens // min_doc_tokens) :]
        truncated_docs = []
        for i in range(len(docs)):
            truncated_docs += truncate_helper(
                docs[i],
                max_inp_tokens=adjusted_max_inp_tokens // len(docs),
                tokenizer=tokenizer,
                min_doc_tokens=min_doc_tokens,
            )
    elif isinstance(docs, list):
        truncated_docs = truncate_helper(
            docs,
            max_inp_tokens=adjusted_max_inp_tokens,
            tokenizer=tokenizer,
            min_doc_tokens=min_doc_tokens,
        )
    example[doc_key] = truncated_docs
    return example


def load_and_truncate(
    dataset_config: SummDataset,
    model_config: BaseModel,
    split: str,
    tokenizer: AutoTokenizer,
) -> Dataset:
    """Load dataset (HF format) and truncate."""
    if getattr(dataset_config, "load_from_disk", False):
        # for prefilterd datasets
        logger.info("{} | {}", dataset_config.path, split)
        dataset = load_from_disk(dataset_config.path)
        dataset = dataset[split]
    else:
        logger.info("{} | {} | {}", dataset_config.path, dataset_config.name, split)
        dataset = load_dataset(
            path=dataset_config.path,
            name=dataset_config.name,
            split=split,
            trust_remote_code=True,
        )
    max_length = model_config.max_length
    if hasattr(model_config, "max_inp_tokens"):
        max_length = model_config.max_inp_tokens
    logger.info(
        "max allowed tokens — doc: {}, summary: {}",
        max_length,
        dataset_config.max_summary_tokens,
    )
    return dataset.map(
        preprocess_example,
        batched=False,
        desc="truncating documents",
        load_from_cache_file=False,
        fn_kwargs={
            "tokenizer": tokenizer,
            "max_inp_tokens": max_length,
            "max_summary_tokens": dataset_config.max_summary_tokens,
            "doc_key": dataset_config.doc_key,
            "min_doc_tokens": getattr(dataset_config, "min_doc_tokens", None),
        },
    )


def prepare_queries(dataset: Dataset, config: SummDataset) -> list[str]:
    """Prepare queries for a given dataset."""
    if getattr(config, "query_key", None) is None:
        return [config.default_query] * len(dataset)
    if getattr(config, "query_prompt", None) is None:
        return [ex[config.query_key] for ex in dataset]
    return [config.query_prompt.format(query=ex[config.query_key]) for ex in dataset]
