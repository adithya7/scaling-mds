"""Retriever systems."""

from dataclasses import dataclass
from pathlib import Path


@dataclass
class BaseRetrieverConfig:
    """Base class for retrievers."""


@dataclass
class SFREmbedding(BaseRetrieverConfig):
    """
    SFR Embedding (Meng et al., 2024).

    https://huggingface.co/Salesforce/SFR-Embedding-2_R
    """

    model_name_or_path: Path = "Salesforce/SFR-Embedding-2_R"
    query_prompt: str = (
        "Instruct: "
        "Given a document, retrieve relevant segments from the document."
        "\n"
        "Query: {query}"
    )
    segment_prompt: str = "{segment}"
    pooling_strategy: str = "last_token"


@dataclass
class E5RoPE(BaseRetrieverConfig):
    """
    E5RoPE (Zhu et al., 2024).

    https://github.com/dwzhu-pku/LongEmbed?tab=readme-ov-file
    """

    model_name_or_path: Path = "dwzhu/e5rope-base"
    query_prompt: str = "query: {query}"
    segment_prompt: str = "passage: {segment}"
    pooling_strategy: str = "average"
