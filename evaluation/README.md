# VAREX-Bench Evaluation

Scoring code for reproducing the results in **Table 2** of the VAREX paper.

## Setup

```bash
pip install -r requirements.txt
```

## Quick Start

```bash
# Download the dataset
pip install datasets
python -c "from datasets import load_dataset; ds = load_dataset('ibm-research/VAREX', split='benchmark')"

# Score your predictions
python score.py results/ \
    --data-dir data/ \
    --manifest manifest.json \
    --field-exclusions field_exclusions.json
```

## Input Format

Predictions should be organized as:
```
results/{model_name}/{mode}/{doc_id}.pred.json
```

Where `mode` is one of: `image_only`, `text_flow`, `text_layout`, `layout_and_image`, `image_only_50dpi`.

Each `.pred.json` file should contain a JSON object matching the document's schema.

## Scoring Details

- **Primary metric**: Exact Match (EM) on normalized values
- **Normalization**: lowercase, whitespace collapse, trailing `.0` strip, trailing `,;` strip
- **Array matching**: Order-invariant via Hungarian algorithm (disable with `--no-order-invariant`)
- **Field exclusions**: 610 fields in 306 documents excluded due to known GT issues (schema-type mismatches, human review)
- **Splits**: Flat (299 docs), Nested (1,146 docs), Table (332 docs)

## Files

- `score.py` — Scoring script
- `manifest.json` — List of 1,777 active document IDs
- `field_exclusions.json` — Fields excluded from scoring (610 fields, 306 docs)
- `requirements.txt` — Python dependencies
