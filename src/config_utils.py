"""Utils."""

from __future__ import annotations

import inspect
from pprint import pformat
from typing import TYPE_CHECKING

from loguru import logger

import configs.datasets as custom_datasets
import configs.models as custom_models
import configs.retriever as custom_retrievers

if TYPE_CHECKING:
    from pathlib import Path


def get_model_config(model_name: str) -> custom_models.BaseModel:
    """Get model config."""
    dataset_configs = dict(inspect.getmembers(custom_models))
    return dataset_configs[model_name]()


def get_dataset_config(dataset_name: str) -> custom_datasets.SummDataset:
    """Get dataset config."""
    dataset_configs = dict(inspect.getmembers(custom_datasets))
    return dataset_configs[dataset_name]()


def get_retriever_config(retriever_name: str) -> custom_retrievers.BaseRetrieverConfig:
    """Get retriever config."""
    dataset_configs = dict(inspect.getmembers(custom_retrievers))
    return dataset_configs[retriever_name]()


def init_eval_config(
    model_config_name: str,
    dataset_config_name: str,
    split: str,
    artifacts_dir: Path,
    output_dir: Path,
) -> tuple[custom_models.BaseModel, custom_datasets.SummDataset]:
    """Load and init evaluation config objects."""
    # load model and dataset configs
    model_config = get_model_config(model_config_name)
    dataset_config = get_dataset_config(dataset_config_name)
    # convert max_summary_length from words to (LLM) tokens
    dataset_config.max_summary_tokens = round(
        dataset_config.max_summary_words * model_config.word2token_ratio
    )
    # setup output path
    model_config.output_dir = output_dir
    model_config.output_dir.mkdir(parents=True, exist_ok=True)
    model_config.pred_path = (
        model_config.output_dir
        / f"{dataset_config_name}_{model_config_name}_{split}.jsonl"
    )
    # setup model_name_or_path
    if not hasattr(model_config, "api"):
        model_config.model_name_or_path = (
            artifacts_dir / "huggingface/model" / model_config.model_name_or_path
        )
    if model_config.tokenizer_name_or_path is None:
        model_config.tokenizer_name_or_path = model_config.model_name_or_path
    else:
        model_config.tokenizer_name_or_path = (
            artifacts_dir / "huggingface/model" / model_config.tokenizer_name_or_path
        )
    # setup dataset path
    dataset_config.path = str(artifacts_dir / dataset_config.path)
    # setup log path
    model_config.log_path = (
        output_dir / "logs" / f"{model_config_name}_{dataset_config_name}_{split}"
    )
    model_config.log_path.mkdir(parents=True, exist_ok=True)
    logger.add(
        str(model_config.log_path) + "/" + "{time}.log",
        format="<m>{time:YYYY-MM-DD at HH:mm:ss}</m> | {level} | {message}",
    )
    # log configs
    logger.info(f"\nmodel config: {pformat(model_config)}")
    logger.info(f"\ndataset config: {pformat(dataset_config)}")

    return model_config, dataset_config
