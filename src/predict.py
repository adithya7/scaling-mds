"""Prediction script."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import fire
import nltk
from loguru import logger
from transformers import AutoTokenizer

from config_utils import get_retriever_config, init_eval_config
from data_utils import load_and_truncate, prepare_queries
from iterative_utils import predict_iterative
from llm_api_utils import init_api, predict_api
from retriever_utils import preprocess_rag
from vllm_utils import init_vllm, predict_vllm

logger.remove()
logger.add(
    sys.stdout,
    colorize=True,
    format="<m>{time:YYYY-MM-DD at HH:mm:ss}</m> | {level} | {message}",
)

nltk.download("punkt_tab", quiet=True)


def predict(  # noqa: PLR0913
    model_config_name: str,
    dataset_config_name: str,
    split: str,
    artifacts_dir: str,
    output_dir: str,
    num_gpus: int | None = None,
) -> None:
    """Get summary predictions."""
    artifacts_dir = Path(artifacts_dir)
    output_dir = Path(output_dir)
    model_config, dataset_config = init_eval_config(
        model_config_name=model_config_name,
        dataset_config_name=dataset_config_name,
        split=split,
        artifacts_dir=artifacts_dir,
        output_dir=output_dir,
    )
    # load dataset and truncate documents if needed
    # this truncation is standard across all methods (full context, iterative or RAG)
    # ensures each method gets the same input
    tokenizer = AutoTokenizer.from_pretrained(model_config.tokenizer_name_or_path)
    dataset = load_and_truncate(dataset_config, model_config, split, tokenizer)
    # predict summaries (full context, iterative or RAG)
    # for iterative and RAG methods, helper functions will preprocess the documents.
    intermediate_outputs = None
    if hasattr(model_config, "retriever"):
        logger.info("using RAG w/ retriever: {}", model_config.retriever)
        retriever_config = get_retriever_config(model_config.retriever)
        retriever_config.model_name_or_path = (
            artifacts_dir / "huggingface/model" / retriever_config.model_name_or_path
        )
        # for each example, a list of segments selected by the retriever
        intermediate_outputs = preprocess_rag(
            dataset=dataset,
            model_config=model_config,
            dataset_config=dataset_config,
            retriever_config=retriever_config,
        )
        # for each example, concatenate selected segments
        queries = prepare_queries(dataset, dataset_config)
        if hasattr(model_config, "api"):
            # API-based model inference
            logger.info("using API-based inference...")
            client = init_api(model_config)
            outputs = predict_api(
                client=client,
                docs=intermediate_outputs,
                queries=queries,
                model_config=model_config,
                dataset_config=dataset_config,
            )
        else:
            # initialize model (and tokenizer)
            # to avoid OOM, do this after RAG
            logger.info("using vLLM for inference...")
            model = init_vllm(model_config, num_gpus)
            outputs = predict_vllm(
                model=model,
                docs=intermediate_outputs,
                queries=queries,
                tokenizer=tokenizer,
                model_config=model_config,
                dataset_config=dataset_config,
            )
    elif hasattr(model_config, "iterative_method"):
        # iterative
        logger.info("using iterative method {}", model_config.iterative_method)
        # initialize model
        if hasattr(model_config, "api"):
            logger.info("using API-based inference...")
            model = init_api(model_config)
        else:
            logger.info("using vLLM for inference...")
            model = init_vllm(model_config, num_gpus)
        # preprocessing and summary prediction is handled by the helper function
        outputs, intermediate_outputs = predict_iterative(
            model=model,
            dataset=dataset,
            tokenizer=tokenizer,
            model_config=model_config,
            dataset_config=dataset_config,
        )
    else:
        # full context
        logger.info("using full context")
        queries = prepare_queries(dataset, dataset_config)
        if hasattr(model_config, "api"):
            # API-based model inference
            logger.info("using API-based inference...")
            client = init_api(model_config)
            outputs = predict_api(
                client=client,
                docs=dataset[dataset_config.doc_key],
                queries=queries,
                model_config=model_config,
                dataset_config=dataset_config,
            )
        else:
            # local inference using vLLM
            # initialize model (and tokenizer)
            logger.info("using vLLM for inference...")
            model = init_vllm(model_config, num_gpus)
            outputs = predict_vllm(
                model=model,
                docs=dataset[dataset_config.doc_key],
                queries=queries,
                tokenizer=tokenizer,
                model_config=model_config,
                dataset_config=dataset_config,
            )

    # save predictions
    logger.info("saving predictions to {}", model_config.pred_path)
    with model_config.pred_path.open("w") as wf:
        for idx in range(len(outputs)):
            # for retrieval and iterative methods, save intermediate predictions
            # retrieved segments (truncated documents) or
            # summaries at each level (hierarchical) or step (incremental)
            intermediate_pred = (
                intermediate_outputs[idx] if intermediate_outputs else None
            )
            question = (
                dataset[idx][dataset_config.query_key]
                if dataset_config.query_key
                else dataset_config.default_query
            )
            wf.write(
                json.dumps(
                    {
                        "src": dataset[idx][dataset_config.doc_key],
                        "intermediate_pred": intermediate_pred,
                        "pred": outputs[idx],
                        "gold": dataset[idx][dataset_config.summary_key],
                        "question": question,
                    }
                )
                + "\n"
            )


if __name__ == "__main__":
    fire.Fire(predict)
