# VAREX-Bench

**A benchmark for multi-modal structured extraction from documents**

VAREX-Bench evaluates how well language models extract structured JSON from document images, spatial text, and plain text. It contains 1,777 U.S. government forms with 1,771 unique per-document schemas and 21,084 scoreable fields, evaluated across 20 models and 4 input modalities.

**[Project Page](https://udibarzi.github.io/varex-bench/)** · **[Dataset](https://huggingface.co/datasets/ibm-research/VAREX)** · **[Paper (ArXiv)](https://arxiv.org/abs/2603.15118)**

## Quick Start

### Install

```bash
pip install datasets openai Pillow rapidfuzz numpy scipy
```

### Load the dataset

```python
from datasets import load_dataset
import json

ds = load_dataset("ibm-research/VAREX", split="benchmark")
doc = ds[0]

schema = json.loads(doc["schema"])     # extraction target schema
gt = json.loads(doc["ground_truth"])   # ground truth values
image = doc["image"]                   # PIL Image, 200 DPI
text = doc["text_layout"]             # spatial text with layout
```

### Run extraction

The benchmark supports four input modalities. Here's the primary one (Image):

```python
from openai import OpenAI
import base64, io, json

client = OpenAI()  # or OpenAI(base_url="http://localhost:8000/v1") for vLLM

schema = json.loads(doc["schema"])
buf = io.BytesIO()
doc["image"].save(buf, format="PNG")
b64 = base64.b64encode(buf.getvalue()).decode()

response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": f"Extract structured data from this document.\n"
                                       f"Return a JSON object matching this schema:\n\n"
                                       f"{json.dumps(schema, indent=2)}\n\n"
                                       f"Return null for fields you cannot find.\n"
                                       f"Return ONLY valid JSON.\n"
                                       f"Return an instance of the JSON with extracted values, not the schema itself."},
        {"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}", "detail": "high"}}
        ]},
    ],
    temperature=0,
    response_format={"type": "json_object"},
)
prediction = json.loads(response.choices[0].message.content)
```

See [`example_inference.py`](example_inference.py) for all four modalities (Image, Spatial Text, Plain Text, Spatial+Image), batch processing, and output saving.

> **Important:** All reported results use `temperature=0` and `response_format={"type": "json_object"}` where supported. The system prompt must include the instruction *"Return an instance of the JSON with extracted values, not the schema itself"* — without it, small models may echo the schema structure instead of extracting values.

### Batch inference

```bash
# Run 10 documents on image modality with GPT-4o
python example_inference.py --mode image --model gpt-4o --n 10 --output results/gpt4o/image_only

# Run on spatial text with a local vLLM server
python example_inference.py --mode spatial --model Qwen/Qwen2.5-VL-7B-Instruct \
    --base-url http://localhost:8000/v1 --api-key dummy --output results/qwen7b/text_layout
```

### Score predictions

```bash
# Score against HuggingFace dataset (no local data needed)
python evaluation/score.py results/gpt4o/ \
    --dataset ibm-research/VAREX \
    --field-exclusions evaluation/field_exclusions.json

# Or score against local data directory
python evaluation/score.py results/gpt4o/ \
    --data-dir data/ \
    --field-exclusions evaluation/field_exclusions.json
```

Predictions must be saved as `{output_dir}/{mode}/{doc_id}.pred.json` — the `example_inference.py` script handles this automatically.

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
  eprint  = {2603.15118},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CV}
}
```

## License

Apache 2.0. See [LICENSE](LICENSE).
