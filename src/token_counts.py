"""Get word2token ratio for LLMs."""

import sys
from pathlib import Path

import fire
import google.generativeai as genai
import nltk
import numpy as np
import pandas as pd
import tiktoken
from loguru import logger
from tokenizers import Tokenizer

nltk.download("punkt_tab")

logger.remove()
logger.add(
    sys.stdout,
    colorize=True,
    format="{time:YYYY-MM-DD at HH:mm:ss} | {level} | {message}",
)


def get_hf_word2token_ratio(tokenizer: Tokenizer, filepath: str) -> float:
    """Get word2token ratio for a specific file."""
    with Path(filepath).open() as f:
        text = f.read().strip()
    num_words = len(nltk.word_tokenize(text))
    num_tokens = len(tokenizer.encode(text, add_special_tokens=False).ids)
    return num_tokens / num_words


def get_gemini_word2token_ratio(model: str, filepath: str) -> float:
    """Get word2token ratio for Gemini."""
    with Path("api_keys/gemini.txt").open() as f:
        api_key = f.read().strip()
    genai.configure(api_key=api_key)
    gemini = genai.GenerativeModel(model)
    with Path(filepath).open() as f:
        text = f.read().strip()
    num_words = len(nltk.word_tokenize(text))
    num_tokens = gemini.count_tokens(text).total_tokens
    return num_tokens / num_words


def get_gpt_word2token_ratio(model: str, filepath: str) -> float:
    """Get word2token ratio for OpenAI models."""
    enc = tiktoken.encoding_for_model(model)
    with Path(filepath).open() as f:
        text = f.read().strip()
    num_words = len(nltk.word_tokenize(text))
    num_tokens = len(enc.encode(text))
    return num_tokens / num_words


def token_counts(dir_path: str) -> None:
    """Get word2token ratio for LLMs."""
    filepaths = list(Path(dir_path).glob("*.txt"))
    tokenizer2ratio = {}
    # llama-3.1 and Cohere's command-r-08-2024
    for tokenizer_name in [
        "meta-llama/Meta-Llama-3-8B-Instruct",
        "meta-llama/Meta-Llama-3-70B-Instruct",
        "meta-llama/Meta-Llama-3.1-8B-Instruct",
        "meta-llama/Meta-Llama-3.1-70B-Instruct",
        "CohereForAI/c4ai-command-r-08-2024",
        "CohereForAI/c4ai-command-r-plus-08-2024",
        "ai21labs/AI21-Jamba-1.5-Mini",
    ]:
        tokenizer = Tokenizer.from_pretrained(tokenizer_name)
        word2token_ratio = [
            get_hf_word2token_ratio(tokenizer, filepath) for filepath in filepaths
        ]
        tokenizer2ratio[tokenizer_name] = (
            f"{np.mean(word2token_ratio):.3f} ± {np.std(word2token_ratio):.3f}"
        )
        logger.info(
            "tokenizer: {}, word2token ratio: {:.3f} ± {:.3f}",
            tokenizer_name,
            np.mean(word2token_ratio),
            np.std(word2token_ratio),
        )
    # gemini-1.5
    for model in ["gemini-1.5-flash", "gemini-1.5-pro"]:
        word2token_ratio = [
            get_gemini_word2token_ratio(model, filepath) for filepath in filepaths
        ]
        tokenizer2ratio[model] = (
            f"{np.mean(word2token_ratio):.3f} ± {np.std(word2token_ratio):.3f}"
        )
        logger.info(
            "tokenizer: {}, word2token ratio: {:.3f} ± {:.3f}",
            model,
            np.mean(word2token_ratio),
            np.std(word2token_ratio),
        )
    # gpt-4o-mini, gpt-4o
    for model in ["gpt-4o-mini", "gpt-4o"]:
        word2token_ratio = [
            get_gpt_word2token_ratio(model, filepath) for filepath in filepaths
        ]
        tokenizer2ratio[model] = (
            f"{np.mean(word2token_ratio):.3f} ± {np.std(word2token_ratio):.3f}"
        )
        logger.info(
            "tokenizer: {}, word2token ratio: {:.3f} ± {:.3f}",
            model,
            np.mean(word2token_ratio),
            np.std(word2token_ratio),
        )

    # print tokenizer2ratio as a markdown table, using pandas
    word2token_ratio_df = pd.DataFrame(
        tokenizer2ratio.items(), columns=["Tokenizer", "Word2Token Ratio"]
    )
    logger.info("\n{}", word2token_ratio_df.to_markdown(index=False))


if __name__ == "__main__":
    fire.Fire(token_counts)
