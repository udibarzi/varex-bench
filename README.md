# VAREX-Bench

**A benchmark for multi-modal structured extraction from documents**

VAREX-Bench evaluates how well language models extract structured JSON from document images, spatial text, and plain text. It contains 1,777 U.S. government forms with 1,771 unique per-document schemas and 21,084 scoreable fields, evaluated across 20 models and 4 input modalities.

**[Project Page](https://udibarzi.github.io/varex-bench/)** · **[Dataset](https://huggingface.co/datasets/ibm-research/VAREX)** · **[Paper (ArXiv)](#)**

## Quick Start

```bash
# Install dependencies
pip install datasets rapidfuzz numpy scipy

# Download the dataset
from datasets import load_dataset
ds = load_dataset("ibm-research/VAREX", split="benchmark")

# Score predictions
python evaluation/score.py results/ \
    --data-dir data/ \
    --manifest evaluation/manifest.json \
    --field-exclusions evaluation/field_exclusions.json
```

## Key Results (Vision, Image Only)

| # | Model | Size | EM % |
|---|-------|------|------|
| 1 | Gemini 2.5 Pro | API | 98.0 |
| 2 | Gemini 2.5 Flash | API | 97.3 |
| 3 | Qwen3-VL | 8B | 96.6 |
| 4 | Llama 4 Maverick | 17B×128E | 95.6 |
| 5 | GPT-4o | API | 94.8 |
| 6 | Ministral | 14B | 94.8 |
| 7 | NuExtract 2.0 | 2B | 90.8 |

See [full results on the project page](https://udibarzi.github.io/varex-bench/).

## Citation

```bibtex
@inproceedings{varex2026,
  title  = {VAREX: A Benchmark for Multi-Modal
            Structured Extraction from Documents},
  author = {Barzelay, Udi and Szpektor, Idan},
  year   = {2026}
}
```

## License

Apache 2.0. See [LICENSE](LICENSE).
