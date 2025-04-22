# Scaling Multi-document Summarization

Code for our NAACL 2025 paper, [Scaling Multi-Document Event Summarization: Evaluating Compression vs. Full-Text Approaches](https://arxiv.org/abs/2502.06617).

## Setup

See [requirements.txt](requirements.txt).

## Data

We provide the HuggingFace dataset loading scripts for the three datasets in [data/](data/). Before loading the dataset, use the original releases to download the source data.

## Experiments

We provide slurm scripts for downloading models, running prediction and scoring system generated summaries.

### Downloading models

```bash
sbatch bash_scripts/download.slurm
```

### Generating summaries

```bash
# for Llama-3.1-8B-Instruct on SummHay
# full context
bash bash_scripts/pred.sh Llama31_8B SummHay test
# hierarchical
bash bash_scripts/pred.sh Llama31_8B_Hierarchical SummHay test
# incremental
bash bash_scripts/pred.sh Llama31_8B_Incremental SummHay test
# retrieval-augmented using SFR-Embedding_2
bash bash_scripts/pred.sh Llama31_8B_RAG_SFR SummHay test
```

See [src/configs/](src/configs/) for the full list of retrievers, summarizers and datasets used in our experiments.

### Scoring system generated summaries

We compute ROUGE and A3CU metrics.

```bash
# for Llama-3.1-8B-Instruct on SummHay
# full context
bash bash_scripts/score.sh Llama31_8B SummHay test
```

### System Predictions and Human Eval

We share our system predictions and human evaluation data in this [Google Drive folder](https://drive.google.com/drive/folders/19E-lOSHNNq_fdcxXl0IbPx2xhIJkSUNA?usp=drive_link).

## License

This project is licensed under the MIT License. See the LICENSE file.

## Reference

If you find this work useful, please consider citing our NAACL paper:

```bibtex
@misc{pratapa-mitamura-2025-scaling,
    title={Scaling Multi-Document Event Summarization: Evaluating Compression vs. Full-Text Approaches},
    author={Adithya Pratapa and Teruko Mitamura},
    year={2025},
    eprint={2502.06617},
    archivePrefix={arXiv},
    primaryClass={cs.CL},
    url={https://arxiv.org/abs/2502.06617},
}
```
