#!/usr/bin/env python3
"""VAREX-Bench — Example inference script.

Runs extraction on VAREX documents using any OpenAI-compatible API.
Saves predictions in the format expected by the scorer.

Usage:
    # Run 10 documents on image modality with GPT-4o
    python example_inference.py --mode image --model gpt-4o --n 10 --output results/gpt4o/image_only

    # Run on spatial text with a local vLLM server
    python example_inference.py --mode spatial --model Qwen/Qwen2.5-VL-7B-Instruct \
        --base-url http://localhost:8000/v1 --api-key dummy --output results/qwen7b/text_layout

    # Run all documents on all modalities
    python example_inference.py --mode all --model gpt-4o --output results/gpt4o

Inference settings (matching paper results):
    - temperature=0
    - response_format=json_object (where supported)
    - System prompt includes schema and explicit instruction to return extracted values
"""

import argparse
import base64
import io
import json
import sys
from pathlib import Path

from datasets import load_dataset
from openai import OpenAI


SYSTEM_PROMPT = """Extract structured data from this document.
Return a JSON object matching this schema:

{schema}

Return null for fields you cannot find.
Return ONLY valid JSON.
Return an instance of the JSON with extracted values, not the schema itself."""


def encode_image(pil_image):
    """Encode PIL Image to base64 data URL."""
    buf = io.BytesIO()
    pil_image.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/png;base64,{b64}"


def build_messages(doc, mode, schema_str):
    """Build chat messages for a given modality."""
    system = SYSTEM_PROMPT.format(schema=schema_str)

    if mode == "image":
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": encode_image(doc["image"]), "detail": "high"}}
            ]},
        ]
    elif mode == "spatial":
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": doc["text_layout"]},
        ]
    elif mode == "plain":
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": doc["text_flow"]},
        ]
    elif mode == "spatial+image":
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": [
                {"type": "text", "text": doc["text_layout"]},
                {"type": "image_url", "image_url": {"url": encode_image(doc["image"]), "detail": "high"}},
            ]},
        ]
    else:
        raise ValueError(f"Unknown mode: {mode}")


MODE_TO_DIR = {
    "image": "image_only",
    "spatial": "text_layout",
    "plain": "text_flow",
    "spatial+image": "layout_and_image",
}


def main():
    parser = argparse.ArgumentParser(description="VAREX-Bench inference example")
    parser.add_argument("--mode", choices=["image", "spatial", "plain", "spatial+image", "all"],
                        default="image", help="Input modality (default: image)")
    parser.add_argument("--model", required=True, help="Model name (e.g., gpt-4o, Qwen/Qwen2.5-VL-7B-Instruct)")
    parser.add_argument("--base-url", default=None, help="API base URL (default: OpenAI)")
    parser.add_argument("--api-key", default=None, help="API key (default: OPENAI_API_KEY env var)")
    parser.add_argument("--output", type=Path, required=True, help="Output directory for predictions")
    parser.add_argument("--n", type=int, default=None, help="Number of documents to process (default: all)")
    parser.add_argument("--dataset", default="ibm-research/VAREX", help="HuggingFace dataset ID")
    args = parser.parse_args()

    # Setup client
    client_kwargs = {}
    if args.base_url:
        client_kwargs["base_url"] = args.base_url
    if args.api_key:
        client_kwargs["api_key"] = args.api_key
    client = OpenAI(**client_kwargs)

    # Load dataset
    print(f"Loading dataset: {args.dataset}")
    ds = load_dataset(args.dataset, split="benchmark")
    if args.n:
        ds = ds.select(range(min(args.n, len(ds))))
    print(f"Processing {len(ds)} documents")

    # Determine modes to run
    modes = list(MODE_TO_DIR.keys()) if args.mode == "all" else [args.mode]

    for mode in modes:
        mode_dir = args.output / MODE_TO_DIR[mode] if args.mode == "all" else args.output
        mode_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n--- Mode: {mode} → {mode_dir} ---")

        for i, doc in enumerate(ds):
            doc_id = doc["doc_id"]
            out_path = mode_dir / f"{doc_id}.pred.json"

            if out_path.exists():
                print(f"  [{i+1}/{len(ds)}] {doc_id}: skipped (exists)")
                continue

            schema_str = doc["schema"]
            messages = build_messages(doc, mode, schema_str)

            try:
                response = client.chat.completions.create(
                    model=args.model,
                    messages=messages,
                    temperature=0,
                    response_format={"type": "json_object"},
                )
                content = response.choices[0].message.content
                prediction = json.loads(content)
            except json.JSONDecodeError:
                prediction = {"_error": "Invalid JSON", "_raw": content[:500]}
            except Exception as e:
                prediction = {"_error": str(e)}

            with open(out_path, "w") as f:
                json.dump(prediction, f, indent=2)

            print(f"  [{i+1}/{len(ds)}] {doc_id}: done")

    print(f"\nPredictions saved to {args.output}/")
    print(f"Score with: python evaluation/score.py {args.output} "
          f"--dataset {args.dataset} --field-exclusions evaluation/field_exclusions.json")


if __name__ == "__main__":
    main()
