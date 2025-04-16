"""vLLM utils."""

from __future__ import annotations

from typing import TYPE_CHECKING

from vllm import LLM, SamplingParams

if TYPE_CHECKING:
    from transformers import AutoTokenizer

    from configs.datasets import SummDataset
    from configs.models import BaseModel


def init_vllm(config: BaseModel, num_gpus: int) -> LLM:
    """Initialize vLLM model."""
    return LLM(
        model=config.model_name_or_path,
        dtype="bfloat16",
        tensor_parallel_size=num_gpus,
        trust_remote_code=True,
        max_model_len=config.max_length,
        quantization=getattr(config, "quantization", None),
        load_format=getattr(config, "load_format", "auto"),
    )


def check_input_type(docs: list[list[str]]) -> bool:
    """Check if the input is a list of list of strings."""
    return (
        isinstance(docs, list)
        and all(isinstance(sublist, list) for sublist in docs)
        and all(isinstance(item, str) for sublist in docs for item in sublist)
    )


def get_prompt_token_ids(
    docs: list[list[str]],
    queries: list[str],
    tokenizer: AutoTokenizer,
    model_config: BaseModel,
    dataset_config: SummDataset,
) -> list[list[int]]:
    """Prepare input for each example, and tokenize."""
    if not check_input_type(docs):
        msg = "docs should be a list of list of strings"
        raise TypeError(msg)
    # prepare input for each example
    # concatenate documents within each example
    if getattr(model_config, "grounded_generation", False):
        # use grounded generation prompt
        # pass documents separately from the message
        # used for CommandR and Jamba-1.5
        prompts = [
            model_config.prompt.format(
                num_words=dataset_config.max_summary_words,
                question=query,
            )
            for query in queries
        ]
        if model_config.grounded_template == "command-r":
            # use CommandR's grounded generation template
            prompt_token_ids = [
                tokenizer.apply_grounded_generation_template(
                    [{"role": "user", "content": prompt}],
                    tokenize=True,
                    documents=[{"text": doc_text} for doc_text in ex_docs],
                    add_generation_prompt=True,
                    citation_mode="fast",
                )
                for prompt, ex_docs in zip(prompts, docs)
            ]
        elif model_config.grounded_template == "jamba-1.5":
            # default grounded generation (Jamba-1.5)
            prompt_token_ids = [
                tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    tokenize=True,
                    documents=[{"text": doc_text} for doc_text in ex_docs],
                    add_generation_prompt=True,
                )
                for prompt, ex_docs in zip(prompts, docs)
            ]
        else:
            msg = (
                f"grounded template '{model_config.grounded_template}' not implemented"
            )
            raise NotImplementedError(msg)
    else:
        # use default chat template
        # include documents in the message
        prompts = [
            model_config.prompt.format(
                document="\n".join(ex_docs),
                num_words=dataset_config.max_summary_words,
                question=query,
            )
            for ex_docs, query in zip(docs, queries)
        ]
        # add model specific chat template
        prompt_token_ids = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=True,
                add_generation_prompt=True,
            )
            for prompt in prompts
        ]
    return prompt_token_ids


def predict_vllm(  # noqa: PLR0913
    model: LLM,
    docs: list[list[str]],
    queries: list[str],
    tokenizer: AutoTokenizer,
    model_config: BaseModel,
    dataset_config: SummDataset,
) -> list[str]:
    """Predict using vLLM."""
    sampling_params = SamplingParams(
        temperature=0.5,
        top_p=1.0,
        best_of=None,
        seed=None,
        max_tokens=dataset_config.max_summary_tokens,
    )
    prompt_token_ids = get_prompt_token_ids(
        docs=docs,
        queries=queries,
        tokenizer=tokenizer,
        model_config=model_config,
        dataset_config=dataset_config,
    )
    outputs = model.generate(
        prompt_token_ids=prompt_token_ids,
        sampling_params=sampling_params,
        use_tqdm=True,
    )
    return [output.outputs[0].text for output in outputs]
