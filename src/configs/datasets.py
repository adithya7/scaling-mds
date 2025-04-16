"""Config file for datasets."""

from dataclasses import dataclass


@dataclass
class SummDataset:
    """Default config for summarization datasets."""

    # default keys
    doc_key: str = "document"
    summary_key: str = "summary"
    query_key: str = "query"

    # default query for datasets that don't have one
    default_query: str = "Generate a summary of the document."

    min_doc_length: int = 128


@dataclass
class WCEP(SummDataset):
    """
    WCEP dataset (Ghalandari et al., ACL 2020).

    Full version from original work.
    """

    path: str = "misc/wcep/wcep.py"
    name: str = "wcep"
    query_key: str = None

    max_summary_words: int = 46  # 80th percentile on nltk tokens (validation set)


@dataclass
class BackgroundSumm(SummDataset):
    """
    Background summarization of events from Pratapa et al., 2023.

    https://arxiv.org/abs/2310.16197
    """

    path: str = "misc/timeline_summ/timeline_summ.py"
    name: str = "background"
    doc_key: str = "past_articles"
    query_key: str = None  # experimenting with no query.
    max_summary_words: int = 238  # 80th percentile nltk tokens (validation set)


@dataclass
class BackgroundQFSumm(BackgroundSumm):
    """
    Query-focused variant of background summarization.

    We use the update summary as the query.
    """

    query_key: str = "query"
    query_prompt: str = (
        "You are provided a news update about the event."
        "\n"
        "News update: {query}"
        "\n"
        "For a user reading the above news update,"
        " "
        "generate a background summary of the event using the provided document."
    )


@dataclass
class BackgroundFiltered(SummDataset):
    """
    Background summarization, but filtered using query.

    Same format at BackgroundSumm
    """

    path: str = "misc/background_prefiltered_E5RoPE"
    doc_key: str = "past_articles"
    query_key: str = None  # experimenting with no query.
    max_summary_words: int = 238  # 80th percentile nltk tokens (validation set)
    load_from_disk: bool = True


@dataclass
class SummHay(SummDataset):
    """
    Summary of a haystack dataset from Laban et al., 2024.

    https://arxiv.org/abs/2407.01370
    """

    path: str = "misc/summhay/summhay.py"
    name: str = "summhay"
    max_summary_words: int = 245  # 80th percentile nltk tokens (test set)


@dataclass
class SummHayOracle(SummHay):
    """SummHay datasets with oracle documents."""

    name: str = "summhay_oracle"
