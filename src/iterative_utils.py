"""Compression based summarization."""

from __future__ import annotations

from collections import defaultdict, deque
from itertools import groupby
from typing import TYPE_CHECKING

import nltk
from loguru import logger

from client_api import APIClient
from data_utils import prepare_queries
from llm_api_utils import predict_api
from vllm_utils import predict_vllm

if TYPE_CHECKING:
    from datasets import Dataset
    from transformers import AutoTokenizer
    from vllm import LLM

    from configs.datasets import SummDataset
    from configs.models import BaseModel


def count_tokens(text: str, tokenizer: AutoTokenizer) -> int:
    """Count tokens using HF tokenizer."""
    return len(tokenizer.encode(text, add_special_tokens=False))


def prepare_chunks(
    dataset: Dataset,
    tokenizer: AutoTokenizer,
    max_chunk_tokens: int,
    dataset_config: SummDataset,
) -> list[tuple[int, str]]:
    """
    Truncate each document to max chunk tokens.

    Only truncate at sentence boundaries.
    """
    indexed_docs = []
    for idx, docs in enumerate(dataset[dataset_config.doc_key]):
        truncated_docs = []
        for doc in docs:
            sents = nltk.sent_tokenize(doc)
            truncated_doc = ""
            for sent in sents:
                if (
                    count_tokens(truncated_doc + " " + sent, tokenizer)
                    > max_chunk_tokens
                ):
                    break
                truncated_doc += " " + sent
            truncated_docs += [truncated_doc]
        indexed_docs += [(idx, truncated_doc) for truncated_doc in truncated_docs]
    return indexed_docs


def pack_inputs(
    docs: list[str],
    tokenizer: AutoTokenizer,
    max_tokens: int,
) -> list[list[str]]:
    """Pack existing summaries into the model context limit, for next merging step."""
    doc_toks = [tokenizer.encode(doc, add_special_tokens=False) for doc in docs]
    output = []
    start_idx, end_idx = 0, 0
    while start_idx < len(doc_toks):
        if sum([len(item) for item in doc_toks[start_idx:]]) <= max_tokens:
            end_idx = len(doc_toks)
        else:
            input_len = 0
            for doc_idx, doc in enumerate(doc_toks[start_idx:], start=start_idx):
                if input_len + len(doc) > max_tokens:
                    end_idx = doc_idx
                    break
                input_len += len(doc)
        output += [tokenizer.batch_decode(doc_toks[start_idx:end_idx])]
        start_idx = end_idx
    return output


def hierarchical_merge(  # noqa: PLR0913
    docs: list[tuple[int, str]],
    ex2query: dict[int, str],
    model: LLM | APIClient,
    tokenizer: AutoTokenizer,
    model_config: BaseModel,
    dataset_config: SummDataset,
    max_doc_tokens: int,
) -> tuple[dict[int, str], dict[int, dict[int, list[str]]]]:
    """
    Hierarchical merging of summaries.

    Returns final summaries and intermediate summaries at each level.
    """
    # get level-0 summaries
    level = 0
    logger.info("starting level {}...", level)
    if isinstance(model, APIClient):
        summaries = predict_api(
            client=model,
            docs=[[item[1]] for item in docs],
            queries=[ex2query[item[0]] for item in docs],
            model_config=model_config,
            dataset_config=dataset_config,
        )
    else:
        summaries = predict_vllm(
            model=model,
            docs=[[item[1]] for item in docs],
            queries=[ex2query[item[0]] for item in docs],
            tokenizer=tokenizer,
            model_config=model_config,
            dataset_config=dataset_config,
        )
    # example indexed summaries
    summaries = [(doc[0], summary) for doc, summary in zip(docs, summaries)]
    ex2summary = {}  # final summary

    ex2intermediate = {}  # intermediate summaries
    for ex_idx, summary in summaries:
        # cache level-0 summaries
        if ex_idx not in ex2intermediate:
            ex2intermediate[ex_idx] = defaultdict(list)
        ex2intermediate[ex_idx][level] += [summary]  # level-0

    num_examples = max([item[0] for item in docs]) + 1
    level += 1
    while len(ex2summary) < num_examples:
        # prepare inputs for next merging step
        level_inputs = []
        for k, v in groupby(summaries, key=lambda x: x[0]):
            ex_summaries = [item[1] for item in v]
            if len(ex_summaries) == 1:
                # example is fully processed, obtained a single summary
                ex2summary[k] = ex_summaries[0]
            else:
                ex_inputs = pack_inputs(
                    docs=ex_summaries,
                    tokenizer=tokenizer,
                    max_tokens=max_doc_tokens,
                )
                level_inputs += [(k, item) for item in ex_inputs]
        logger.info(
            "starting level {}, already finished {}/{} examples...",
            level,
            len(ex2summary),
            num_examples,
        )
        if len(level_inputs) == 0:
            break
        # get level-N summaries
        # each doc is a list of summaries from previous level;
        #   packed into max_doc_tokens
        if isinstance(model, APIClient):
            summaries = predict_api(
                client=model,
                docs=[item[1] for item in level_inputs],
                queries=[ex2query[item[0]] for item in level_inputs],
                model_config=model_config,
                dataset_config=dataset_config,
            )
        else:
            summaries = predict_vllm(
                model=model,
                docs=[item[1] for item in level_inputs],
                queries=[ex2query[item[0]] for item in level_inputs],
                tokenizer=tokenizer,
                model_config=model_config,
                dataset_config=dataset_config,
            )
        # example indexed summaries
        summaries = [(inp[0], summary) for inp, summary in zip(level_inputs, summaries)]
        for ex_idx, summary in summaries:
            ex2intermediate[ex_idx][level] += [summary]
        level += 1

    return ex2summary, ex2intermediate


def incremental_merge(  # noqa: PLR0913
    docs: list[tuple[int, str]],
    ex2query: dict[int, str],
    model: LLM | APIClient,
    tokenizer: AutoTokenizer,
    model_config: BaseModel,
    dataset_config: SummDataset,
) -> tuple[dict[int, str], dict[int, list[str]]]:
    """
    Incremental merging.

    Returns final summaries and intermediate summaries at each step.
    """
    # we pass past summary and the current chunk as input to the model
    # group chunks by example
    ex2chunks = defaultdict(deque)
    for ex_idx, chunk in docs:
        ex2chunks[ex_idx] += [chunk]
    # max number of chunks
    max_chunks = max([len(chunks) for chunks in ex2chunks.values()])
    # populate summaries
    ex2summary = defaultdict(str)
    # intermediate summaries
    ex2intermediate = defaultdict(list)
    # iterative over chunks, left to right
    for idx in range(max_chunks):
        logger.info("progress: {}/{} chunks", idx, max_chunks)
        # prepare inputs
        inputs = []
        for ex_idx, chunks in ex2chunks.items():
            if len(chunks) == 0:
                continue
            # pack existing summary with next chunk
            inputs += [(ex_idx, [ex2summary[ex_idx], chunks.popleft()])]
        # get summaries
        # each input contains two items, past summary and current chunk
        # we assume the concatenation is smaller than model_config.chunk_size
        if isinstance(model, APIClient):
            summaries = predict_api(
                client=model,
                docs=[item[1] for item in inputs],
                queries=[ex2query[item[0]] for item in inputs],
                model_config=model_config,
                dataset_config=dataset_config,
            )
        else:
            summaries = predict_vllm(
                model=model,
                docs=[item[1] for item in inputs],
                queries=[ex2query[item[0]] for item in inputs],
                tokenizer=tokenizer,
                model_config=model_config,
                dataset_config=dataset_config,
            )
        # update summaries
        for ex_idx, summary in zip([item[0] for item in inputs], summaries):
            # update summaries for relevant examples
            ex2summary[ex_idx] = summary
            ex2intermediate[ex_idx] += [summary]

    return ex2summary, ex2intermediate


def predict_iterative(
    model: LLM | APIClient,
    dataset: Dataset,
    tokenizer: AutoTokenizer,
    model_config: BaseModel,
    dataset_config: SummDataset,
) -> tuple[list[str], list[list[str]] | list[dict[int, list[str]]]]:
    """Get summary predictions."""
    # prepare inputs for iterative summarization
    indexed_docs = prepare_chunks(
        dataset=dataset,
        tokenizer=tokenizer,
        max_chunk_tokens=model_config.chunk_size,
        dataset_config=dataset_config,
    )
    # prepare queries
    queries = prepare_queries(dataset, dataset_config)
    ex2query = dict(enumerate(queries))
    # predict
    if model_config.iterative_method == "hierarchical":
        ex2summary, ex2intermediate = hierarchical_merge(
            docs=indexed_docs,
            ex2query=ex2query,
            model=model,
            tokenizer=tokenizer,
            model_config=model_config,
            dataset_config=dataset_config,
            max_doc_tokens=model_config.chunk_size,
        )
    elif model_config.iterative_method == "incremental":
        ex2summary, ex2intermediate = incremental_merge(
            docs=indexed_docs,
            ex2query=ex2query,
            model=model,
            tokenizer=tokenizer,
            model_config=model_config,
            dataset_config=dataset_config,
        )
    return (
        [ex2summary[idx] for idx in range(len(dataset))],
        [ex2intermediate[idx] for idx in range(len(dataset))],
    )
