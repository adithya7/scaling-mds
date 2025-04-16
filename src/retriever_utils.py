"""Utils for retriever systems."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import cohere
import nltk
import numpy as np
import tiktoken
import torch
import torch.nn.functional as F  # noqa: N812
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

if TYPE_CHECKING:
    from datasets import Dataset

    from configs.datasets import SummDataset
    from configs.models import BaseModel
    from configs.retriever import BaseRetrieverConfig

encoding = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    """Count number of tokens, using OpenAI's tokenizer."""
    return len(encoding.encode(text))


class Retriever:
    """Retriever class."""

    def __init__(self, config: BaseRetrieverConfig) -> None:
        """Initialize retriever."""
        self.config = config
        self.init_model()

    def init_model(self) -> None:
        """Initialize model."""
        raise NotImplementedError

    def retrieve_docs(self, docs: list[str], query: str) -> list[str]:
        """Retrieve relevant documents."""
        raise NotImplementedError

    def truncate_doc(
        self,
        doc: str,
        max_segment_tokens: int,
    ) -> list[str]:
        """
        Truncate document to max_segment_tokens.

        Only truncate at sentence boundaries.
        """
        sents = nltk.sent_tokenize(doc)
        truncated_doc = ""
        for sent in sents:
            if count_tokens(truncated_doc + " " + sent) > max_segment_tokens:
                break
            truncated_doc += " " + sent
        return truncated_doc


class HFRetriever(Retriever):
    """Retriever models from huggingface."""

    def __init__(self, config: BaseRetrieverConfig) -> None:
        """Initialize retriever."""
        super().__init__(config)

    def init_model(self) -> None:
        """Initialize model."""
        if "e5rope" in str(self.config.model_name_or_path):
            self.model = AutoModel.from_pretrained(
                self.config.model_name_or_path,
                trust_remote_code=True,
            ).cuda()
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.model_name_or_path,
                trust_remote_code=True,
            )
        else:
            self.model = AutoModel.from_pretrained(
                self.config.model_name_or_path,
                device_map="sequential",  # use as many GPUs as needed
            )
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.model_name_or_path
            )

    @staticmethod
    def _last_token_pool(
        last_hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """From https://hf.co/Salesforce/SFR-Embedding-Mistral."""
        left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]
        if left_padding:
            return last_hidden_states[:, -1]
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[
            torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths
        ]

    @staticmethod
    def _average_pool(
        last_hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """From https://huggingface.co/dwzhu/e5rope-base."""
        last_hidden = last_hidden_states.masked_fill(
            ~attention_mask[..., None].bool(), 0.0
        )
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

    def _get_embeddings(
        self,
        docs: list[str],
        max_segment_tokens: int,
        batch_size: int = 32,
    ) -> np.ndarray:
        """
        Compute embeddings for summary and source sentences.

        adapted from the code at https://hf.co/Salesforce/SFR-Embedding-Mistral
        """
        embeddings = []
        for i in range(0, len(docs), batch_size):
            batch = docs[i : i + batch_size]
            inputs = self.tokenizer(
                batch,
                padding=True,
                return_tensors="pt",
                truncation=True,
                max_length=max_segment_tokens,
                pad_to_multiple_of=8,
            ).to(self.model.device)
            with torch.no_grad():
                outputs = self.model(**inputs)
                if self.config.pooling_strategy == "average":
                    batch_embeddings = self._average_pool(
                        outputs.last_hidden_state, inputs["attention_mask"]
                    )
                elif self.config.pooling_strategy == "last_token":
                    batch_embeddings = self._last_token_pool(
                        outputs.last_hidden_state, inputs["attention_mask"]
                    )
                batch_embeddings = F.normalize(batch_embeddings, p=2, dim=1)
                embeddings += [batch_embeddings]
        return torch.cat(embeddings).cpu().numpy()

    def retrieve_docs(
        self,
        docs: list[str],
        query: str,
        max_segment_tokens: int,
        max_input_tokens: int,
    ) -> list[str]:
        """
        Retrieve relevant documents. Select documents based on relevance to query.

        Expects segments as input, and returns a list of scores.
        """
        # use Huggingface models
        query_prompt = self.config.query_prompt.format(query=query)
        doc_prompts = [self.config.segment_prompt.format(segment=doc) for doc in docs]
        input_prompts = [query_prompt, *doc_prompts]
        self.model.eval()
        embeddings = self._get_embeddings(
            docs=input_prompts, max_segment_tokens=max_segment_tokens
        )
        scores = embeddings[:1] @ embeddings[1:].T
        scores = scores.tolist()[0]

        # pick segments with highest scores
        # we use OpenAI's tokenizer to count tokens
        # this ensures same number of words for any summarizer
        curr_token_count = 0
        selected_docs = [False] * len(docs)
        for idx in np.argsort(scores)[::-1]:
            doc_token_count = count_tokens(docs[idx])
            if (curr_token_count + doc_token_count) > max_input_tokens:
                break
            selected_docs[idx] = True
            curr_token_count += doc_token_count
        return [docs[idx] for idx in range(len(docs)) if selected_docs[idx]]


class CohereRetriever(Retriever):
    """Cohere's Rerank-3."""

    def __init__(self, config: BaseRetrieverConfig) -> None:
        """Initialize retriever."""
        super().__init__(config)

    def init_model(self) -> None:
        """Initialize API client."""
        with Path(f"api_keys/{self.config.api}.txt").open() as f:
            api_key = f.read().strip()
        self.model = cohere.Client(api_key)

    def retrieve_docs(
        self,
        docs: list[str],
        query: str,
        max_input_tokens: int,
    ) -> list[str]:
        """Retrieve relevant documents."""
        """
        Select documents based on relevance to query.

        Expects segments as input, and returns a list of scores.
        """
        # Cohere's Rerank-3
        with Path(f"api_keys/{self.config.api}.txt").open() as f:
            api_key = f.read().strip()
        api_client = cohere.Client(api_key)
        response = api_client.rerank(
            model=self.config.model,
            query=query,
            documents=docs,
        )
        scores = [0.0] * len(docs)
        for result in response["results"]:
            scores[result["index"]] = result["relevance_score"]

        # pick segments with highest scores
        # we use OpenAI's tokenizer to count tokens
        # this ensures same number of words for any summarizer
        curr_token_count = 0
        selected_docs = [False] * len(docs)
        for idx in np.argsort(scores)[::-1]:
            doc_token_count = count_tokens(docs[idx])
            if (curr_token_count + doc_token_count) > max_input_tokens:
                break
            selected_docs[idx] = True
            curr_token_count += doc_token_count
        return [docs[idx] for idx in range(len(docs)) if selected_docs[idx]]


def preprocess_rag(
    dataset: Dataset,
    model_config: BaseModel,
    dataset_config: SummDataset,
    retriever_config: BaseRetrieverConfig,
) -> list[list[str]]:
    """Preprocess documents for RAG."""
    # setup retriever
    if model_config.retriever in ("SFREmbedding", "E5RoPE"):
        retriever = HFRetriever(retriever_config)
    else:
        msg = f"retriever {model_config.retriever} not implemented"
        raise NotImplementedError(msg)

    # truncate documents to max_segment_tokens
    truncated_docs = []
    for idx in tqdm(range(len(dataset)), colour="blue", desc="truncation"):
        truncated_docs += [
            [
                retriever.truncate_doc(doc, model_config.max_segment_tokens)
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
            max_input_tokens=model_config.max_input_tokens,
            max_segment_tokens=model_config.max_segment_tokens,
        )
        output += [selected_segments]

    return output
