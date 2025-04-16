"""Model config at inference."""

from dataclasses import dataclass
from pathlib import Path


@dataclass
class BaseModel:
    """Default model config."""

    output_dir: Path = None
    pred_path: Path = None

    # prompt
    prompt: str = (
        "{document}"
        "\n\n"
        "Question: {question}"
        "\n\n"
        "Answer the question based on the provided document. "
        "Be concise and directly address only the specific question asked. "
        "Limit your response to a maximum of {num_words} words."
        "\n\n"
    )

    # default tokenizer path
    tokenizer_name_or_path: str = None


"""
Llama 3
"""


@dataclass
class Llama3(BaseModel):
    """Default inference config for Llama-3-based models."""

    max_inp_tokens: int = 128 * 1024  # for llama3, this is just the input length limit
    max_length: int = 8192
    word2token_ratio: float = 1.145


@dataclass
class Llama3_8B(Llama3):
    """Llama-3 8B Instruct."""

    model_name_or_path: Path = "meta-llama/Meta-Llama-3-8B-Instruct"


@dataclass
class Llama3_8B_Hierarchical(Llama3_8B):
    """Hierarchical generation with Llama-3 8B Instruct."""

    chunk_size: int = 8192  # same as model max length
    iterative_method: str = "hierarchical"


@dataclass
class Llama3_8B_Incremental(Llama3_8B):
    """Incremental generation with Llama-3 8B Instruct."""

    chunk_size: int = 8192  # same as model max length
    iterative_method: str = "incremental"


@dataclass
class Llama3_70B(Llama3):
    """
    Llama-3 70B Instruct.

    bf16 precision.
    """

    model_name_or_path: Path = "meta-llama/Meta-Llama-3-70B-Instruct"
    max_length: int = 8192


@dataclass
class Llama3_70B_Hierarchical(Llama3_70B):
    """Hierarchical generation with Llama-3 70B Instruct."""

    chunk_size: int = 8192  # same as model max length
    iterative_method: str = "hierarchical"


@dataclass
class Llama3_70B_Incremental(Llama3_70B):
    """Incremental generation with Llama-3 70B Instruct."""

    chunk_size: int = 8192  # same as model max length
    iterative_method: str = "incremental"


"""
Llama 3.1
"""


@dataclass
class Llama31(BaseModel):
    """Default inference config for Llama-3.1-based models."""

    max_length: int = 128 * 1024
    word2token_ratio: float = 1.145


@dataclass
class Llama31_8B(Llama31):
    """Llama-3.1 8B Instruct."""

    model_name_or_path: Path = "meta-llama/Meta-Llama-3.1-8B-Instruct"


@dataclass
class Llama31_8B_Hierarchical(Llama31_8B):
    """Hierarchical generation with Llama-3.1 8B Instruct."""

    chunk_size: int = 4096
    iterative_method: str = "hierarchical"


@dataclass
class Llama31_8B_Hierarchical_8K(Llama31_8B_Hierarchical):
    """Hierarchical generation with Llama-3.1 8B Instruct, 8k chunks."""

    chunk_size: int = 8192


@dataclass
class Llama31_8B_Hierarchical_16K(Llama31_8B_Hierarchical):
    """Hierarchical generation with Llama-3.1 8B Instruct, 16k chunks."""

    chunk_size: int = 16384


@dataclass
class Llama31_8B_Hierarchical_32K(Llama31_8B_Hierarchical):
    """Hierarchical generation with Llama-3.1 8B Instruct, 32k chunks."""

    chunk_size: int = 32768


@dataclass
class Llama31_8B_Incremental(Llama31_8B):
    """Incremental generation with Llama-3.1 8B Instruct."""

    chunk_size: int = 4096
    iterative_method: str = "incremental"


@dataclass
class Llama31_8B_RAG_SFR(Llama31_8B):
    """Retrieval-based generation with Llama-3.1 8B Instruct."""

    retriever: str = "SFREmbedding"
    # OpenAI tokens
    max_segment_tokens: int = 1024
    max_input_tokens: int = 32768


@dataclass
class Llama31_8B_RAG_E5(Llama31_8B):
    """Retrieval-based generation with Llama-3.1 8B Instruct."""

    retriever: str = "E5RoPE"
    # OpenAI tokens
    max_segment_tokens: int = 1024
    max_input_tokens: int = 32768


@dataclass
class Llama31_70B(Llama31):
    """
    Llama-3.1 70B Instruct.

    bf16 precision, but limited context window.
    """

    model_name_or_path: Path = "meta-llama/Meta-Llama-3.1-70B-Instruct"
    max_length: int = 100 * 1024  # 100k


@dataclass
class Llama31_70B_FP8(Llama31):
    """
    Llama-3.1 70B Instruct.

    fp8 quantization.
    supports full context window.
    """

    model_name_or_path: Path = "meta-llama/Meta-Llama-3.1-70B-Instruct"
    quantization: str = "fp8"


@dataclass
class Llama31_70B_FP8_Hierarchical(Llama31_70B_FP8):
    """Hierarchical generation with Llama-3.1 70B Instruct."""

    chunk_size: int = 4096
    iterative_method: str = "hierarchical"


@dataclass
class Llama31_70B_FP8_Incremental(Llama31_70B_FP8):
    """Incremental generation with Llama-3.1 70B Instruct."""

    chunk_size: int = 4096
    iterative_method: str = "incremental"


@dataclass
class Llama31_70B_FP8_RAG_SFR(Llama31_70B_FP8):
    """Retrieval-based generation with Llama-3.1 70B Instruct."""

    retriever: str = "SFREmbedding"
    # OpenAI tokens
    max_segment_tokens: int = 1024
    max_input_tokens: int = 32768


@dataclass
class Llama31_70B_FP8_RAG_E5(Llama31_70B_FP8):
    """Retrieval-based generation with Llama-3.1 70B Instruct."""

    retriever: str = "E5RoPE"
    # OpenAI tokens
    max_segment_tokens: int = 1024
    max_input_tokens: int = 32768


"""
Cohere's Command-R
"""


@dataclass
class CommandR(BaseModel):
    """Default inference config for Cohere's Command-R."""

    max_length: int = 128 * 1024
    word2token_ratio: float = 1.167

    model_name_or_path: Path = "CohereForAI/c4ai-command-r-08-2024"


@dataclass
class CommandR_Hierarchical(CommandR):
    """Hierarchical generation with Command-R."""

    chunk_size: int = 4096
    iterative_method: str = "hierarchical"


@dataclass
class CommandR_Incremental(CommandR):
    """Incremental generation with Command-R."""

    chunk_size: int = 4096
    iterative_method: str = "incremental"


@dataclass
class CommandR_RAG_SFR(CommandR):
    """Retrieval-based generation with Command-R."""

    retriever: str = "SFREmbedding"  # config from retriever.py
    # OpenAI tokens
    max_segment_tokens: int = 1024  # max tokens in a segment
    max_input_tokens: int = 32768  # max tokens in the input to the summarizer


@dataclass
class CommandR_RAG_E5(CommandR):
    """Retrieval-based generation with Command-R."""

    retriever: str = "E5RoPE"
    # OpenAI tokens
    max_segment_tokens: int = 1024
    max_input_tokens: int = 32768


@dataclass
class CommandRGrounded(CommandR):
    """
    Grounded generation with Command-R.

    https://huggingface.co/CohereForAI/c4ai-command-r-08-2024#grounded-generation-and-rag-capabilities
    """

    grounded_generation: bool = True
    grounded_template: str = "command-r"
    # this prompt doesn't include the documents
    # documents are passed separately tokenizer's prompt template
    prompt: str = (
        "Question: {question}"
        "\n\n"
        "Answer the question based on the provided documents. "
        "Be concise and directly address only the specific question asked. "
        "Limit your response to a maximum of {num_words} words."
        "\n\n"
    )


"""
Jamba 1.5
"""


@dataclass
class Jamba15(BaseModel):
    """Default inference config for Jamba-1.5-based models."""

    max_length: int = 128 * 1024
    word2token_ratio: float = 1.219


@dataclass
class Jamba15Mini(Jamba15):
    """Jamba-1.5 Mini."""

    model_name_or_path: Path = "ai21labs/AI21-Jamba-1.5-Mini"


@dataclass
class Jamba15Mini_Hierarchical(Jamba15Mini):
    """Hierarchical generation with Jamba-1.5 Mini."""

    chunk_size: int = 4096
    iterative_method: str = "hierarchical"


@dataclass
class Jamba15Mini_Hierarchical_8K(Jamba15Mini_Hierarchical):
    """Hierarchical generation with Jamba-1.5 Mini, 8k chunks."""

    chunk_size: int = 8192


@dataclass
class Jamba15Mini_Hierarchical_16K(Jamba15Mini_Hierarchical):
    """Hierarchical generation with Jamba-1.5 Mini, 16k chunks."""

    chunk_size: int = 16384


@dataclass
class Jamba15Mini_Hierarchical_32K(Jamba15Mini_Hierarchical):
    """Hierarchical generation with Jamba-1.5 Mini, 32k chunks."""

    chunk_size: int = 32768


@dataclass
class Jamba15Mini_Incremental(Jamba15Mini):
    """Incremental generation with Jamba-1.5 Mini."""

    chunk_size: int = 4096
    iterative_method: str = "incremental"


@dataclass
class Jamba15Mini_RAG_SFR(Jamba15Mini):
    """Retrieval-based generation with Jamba-1.5 Mini."""

    retriever: str = "SFREmbedding"
    # OpenAI tokens
    max_segment_tokens: int = 1024  # max tokens in a segment
    max_input_tokens: int = 32768  # max tokens in the input to the summarizer


@dataclass
class Jamba15Mini_RAG_E5(Jamba15Mini):
    """Retrieval-based generation with Jamba-1.5 Mini."""

    retriever: str = "E5RoPE"
    # OpenAI tokens
    max_segment_tokens: int = 1024
    max_input_tokens: int = 32768


@dataclass
class Jamba15MiniGrounded(Jamba15Mini):
    """
    Grounded generation with Jamba-1.5 Mini.

    https://huggingface.co/ai21labs/AI21-Jamba-1.5-Mini#grounded-generation-with-jamba
    """

    grounded_generation: bool = True
    grounded_template: str = "jamba-1.5"
    # this prompt doesn't include the documents
    # documents are passed separately tokenizer's prompt template
    prompt: str = (
        "Question: {question}"
        "\n\n"
        "Answer the question based on the provided documents. "
        "Be concise and directly address only the specific question asked. "
        "Limit your response to a maximum of {num_words} words."
        "\n\n"
    )


@dataclass
class Jamba15MiniGrounded_Hierarchical(Jamba15MiniGrounded):
    """Hierarchical generation with Jamba-1.5 Mini."""

    chunk_size: int = 4096
    iterative_method: str = "hierarchical"


@dataclass
class Jamba15MiniGrounded_Incremental(Jamba15MiniGrounded):
    """Incremental generation with Jamba-1.5 Mini."""

    chunk_size: int = 4096
    iterative_method: str = "incremental"


@dataclass
class Jamba15MiniGrounded_RAG_SFR(Jamba15MiniGrounded):
    """Retrieval-based generation with Jamba-1.5 Mini."""

    retriever: str = "SFREmbedding"
    # OpenAI tokens
    max_segment_tokens: int = 1024
    max_input_tokens: int = 32768


@dataclass
class Gemini15(BaseModel):
    """Default inference config for Gemini-1.5-based models."""

    max_length: int = 128 * 1024
    word2token_ratio: float = 1.149
    api: str = "gemini"
    key_path: str = "api_keys/gemini.txt"
    # this looks strange, but we want to pretokenize and truncate input
    # this keeps the input consistent across full-context and compression-based methods
    # Gemini's tokenizer is not publicly available, so we pretokenize using Llama-3
    tokenizer_name_or_path: str = "meta-llama/Meta-Llama-3.1-8B-Instruct"


@dataclass
class Gemini15Flash(Gemini15):
    """Gemini-1.5 Flash."""

    model_name_or_path: str = "gemini-1.5-flash"


@dataclass
class Gemini15Flash_Hierarchical(Gemini15Flash):
    """Hierarchical generation with Gemini-1.5 Flash."""

    chunk_size: int = 4096
    iterative_method: str = "hierarchical"


@dataclass
class Gemini15Flash_Incremental(Gemini15Flash):
    """Incremental generation with Gemini-1.5 Flash."""

    chunk_size: int = 4096
    iterative_method: str = "incremental"


@dataclass
class Gemini15Flash_RAG_SFR(Gemini15Flash):
    """Retrieval-based generation with Gemini-1.5 Flash."""

    retriever: str = "SFREmbedding"
    # OpenAI tokens
    max_segment_tokens: int = 1024
    max_input_tokens: int = 32768


@dataclass
class Gemini15Flash_RAG_E5(Gemini15Flash):
    """Retrieval-based generation with Gemini-1.5 Flash."""

    retriever: str = "E5RoPE"
    # OpenAI tokens
    max_segment_tokens: int = 1024
    max_input_tokens: int = 32768


@dataclass
class Gemini15Pro(Gemini15):
    """Gemini-1.5 Pro."""

    model_name_or_path: str = "gemini-1.5-pro"


@dataclass
class Gemini15Pro_Hierarchical(Gemini15Pro):
    """Hierarchical generation with Gemini-1.5 Pro."""

    chunk_size: int = 4096
    iterative_method: str = "hierarchical"


@dataclass
class Gemini15Pro_Incremental(Gemini15Pro):
    """Incremental generation with Gemini-1.5 Pro."""

    chunk_size: int = 4096
    iterative_method: str = "incremental"


@dataclass
class Gemini15Pro_RAG_SFR(Gemini15Pro):
    """Retrieval-based generation with Gemini-1.5 Pro."""

    retriever: str = "SFREmbedding"
    # OpenAI tokens
    max_segment_tokens: int = 1024
    max_input_tokens: int = 32768


@dataclass
class Gemini15Pro_RAG_E5(Gemini15Pro):
    """Retrieval-based generation with Gemini-1.5 Pro."""

    retriever: str = "E5RoPE"
    # OpenAI tokens
    max_segment_tokens: int = 1024
    max_input_tokens: int = 32768
