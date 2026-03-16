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

See [full results and model comparison on the project page](https://udibarzi.github.io/varex-bench/).

## Citation

```bibtex
@article{varex2026,
  title   = {VAREX: A Benchmark for Multi-Modal
             Structured Extraction from Documents},
  author  = {Barzelay, Udi and Azulai, Ophir and Shapira, Inbar and
             Friedman, Idan and Abo Dahood, Foad and Lee, Madison and
             Daniels, Abraham},
  year    = {2026},
  note    = {arXiv preprint forthcoming}
}
```

## License

Apache 2.0. See [LICENSE](LICENSE).
