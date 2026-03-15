#!/usr/bin/env python3
"""VAREX Benchmark — Score predictions against ground truth.


Computes 6 metrics per field, classifies documents into three splits
(Flat/Nested/Table), and reports per-split scores with perfect doc rates.

Uses rapidfuzz for fast Levenshtein distance and threads for parallel scoring.

Usage:
    python scripts/score.py results/ --data-dir data/ --manifest manifest.json

    # With field exclusions from human review:
    python scripts/score.py results/ --data-dir data/ --manifest manifest.json \
        --field-exclusions field_exclusions.json

Known scoring limitations (not normalized):
    - Phone/date formatting: (909) 555-0198 vs 909-555-0198
    - Name order: "Last, First" vs "First Last"
    These are documented as benchmark characteristics, not bugs.
"""

import argparse
import csv
import json
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from rapidfuzz.distance import Levenshtein
from scipy.optimize import linear_sum_assignment


def norm_ci(value):
    if value is None: return ""
    return str(value).lower()

def norm_full(value):
    """Normalize for exact_match_normalized: lowercase, collapse whitespace,
    strip trailing .0 on integers, strip trailing comma/semicolon."""
    if value is None: return ""
    s = " ".join(str(value).lower().split())
    if s.endswith(".0"):
        test_str = s[:-2].lstrip("-").replace(",", "")
        if test_str.isdigit():
            s = s[:-2]
    s = s.rstrip(",;")
    return s

def raw_str(value):
    if value is None: return ""
    return str(value)

def anls(gt, pred, threshold=0.5):
    if not gt and not pred: return 1.0
    if not gt or not pred: return 0.0
    mx = max(len(gt), len(pred))
    if mx == 0: return 1.0
    nl = Levenshtein.distance(gt, pred) / mx
    return 1.0 - nl if nl < threshold else 0.0

def flatten(d, prefix=""):
    items = {}
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict): items.update(flatten(v, key))
        elif isinstance(v, list) and v and isinstance(v[0], dict):
            for i, item in enumerate(v):
                if isinstance(item, dict):
                    items.update(flatten(item, f"{key}[{i}]"))
                else:
                    items[f"{key}[{i}]"] = item
        else: items[key] = v
    return items

def _flatten_leaves(obj, prefix=""):
    """Flatten a JSON object to leaf field paths (for similarity computation)."""
    items = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            key = f"{prefix}.{k}" if prefix else k
            items.update(_flatten_leaves(v, key))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            items.update(_flatten_leaves(v, f"{prefix}[{i}]"))
    else:
        items[prefix] = obj
    return items


def align_arrays(gt, pred):
    """Reorder prediction arrays to best match GT arrays using Hungarian algorithm.

    Recursively walks GT and pred JSON. For each array-of-dicts, computes a
    similarity matrix based on normalized leaf field matching and finds the
    optimal assignment. Returns a new pred with arrays reordered.
    """
    if not isinstance(gt, dict) or not isinstance(pred, dict):
        return pred

    result = {}
    for key, gt_val in gt.items():
        pred_val = pred.get(key)
        if pred_val is None:
            # Key missing in pred — leave it out (scorer handles missing)
            continue

        if (isinstance(gt_val, list) and gt_val and isinstance(gt_val[0], dict)
                and isinstance(pred_val, list) and pred_val and isinstance(pred_val[0], dict)):
            # Both are arrays of objects — align them
            result[key] = _align_array(gt_val, pred_val)
        elif isinstance(gt_val, dict) and isinstance(pred_val, dict):
            # Recurse into nested objects
            result[key] = align_arrays(gt_val, pred_val)
        else:
            result[key] = pred_val

    # Carry over any pred keys not in GT (scorer ignores them, but keep for completeness)
    for key, pred_val in pred.items():
        if key not in result:
            result[key] = pred_val

    return result


def _align_array(gt_arr, pred_arr):
    """Align a prediction array to a GT array using Hungarian matching."""
    n_gt = len(gt_arr)
    n_pred = len(pred_arr)

    # Flatten each element to normalized leaf values
    gt_leaves = [_flatten_leaves(elem) for elem in gt_arr]
    pred_leaves = [_flatten_leaves(elem) for elem in pred_arr]

    # Build similarity matrix: sim[i][j] = fraction of GT[i] leaves matched by pred[j]
    sim = np.zeros((n_gt, n_pred))
    for i, gl in enumerate(gt_leaves):
        if not gl:
            continue
        for j, pl in enumerate(pred_leaves):
            matches = sum(1 for k, v in gl.items()
                          if k in pl and norm_full(v) == norm_full(pl[k]))
            sim[i][j] = matches / len(gl)

    # Hungarian algorithm on cost matrix (1 - similarity)
    cost = 1.0 - sim
    row_ind, col_ind = linear_sum_assignment(cost)

    # Build mapping: gt_index -> pred_index
    used_pred = set()
    aligned = [None] * n_gt
    for gi, pi in zip(row_ind, col_ind):
        if sim[gi][pi] > 0:  # Only accept matches with at least some similarity
            aligned[gi] = pred_arr[pi]
            used_pred.add(pi)
        # else: leave as None, will be filled below

    # Fill unmatched GT positions with None (scorer will treat as missing)
    # and append unmatched pred extras at end
    reordered = []
    for i in range(n_gt):
        if aligned[i] is not None:
            # Recurse into matched pairs for nested arrays
            reordered.append(align_arrays(gt_arr[i], aligned[i]))
        else:
            reordered.append({})  # Empty dict — fields will score as missing

    # Append unmatched pred elements
    for j in range(n_pred):
        if j not in used_pred:
            reordered.append(pred_arr[j])

    return reordered


def score_field(gt_val, pred_val):
    gr, pr = raw_str(gt_val), raw_str(pred_val)
    gc, pc = norm_ci(gt_val), norm_ci(pred_val)
    gn, pn = norm_full(gt_val), norm_full(pred_val)
    return {
        "exact_match": int(gr == pr),
        "exact_match_ci": int(gc == pc),
        "exact_match_normalized": int(gn == pn),
        "anls": anls(gr, pr),
        "anls_ci": anls(gc, pc),
        "anls_normalized": anls(gn, pn),
    }

def classify_doc(fields):
    """Classify a document into Flat/Nested/Table based on its GT field paths."""
    has_array = any('[' in f for f in fields)
    has_nested = any('.' in f for f in fields)
    if has_array:
        return 'Table'
    elif has_nested:
        return 'Nested'
    else:
        return 'Flat'


def load_field_exclusions(path):
    """Load field exclusions from JSON file.

    Returns dict: doc_id -> set of field_paths to exclude from scoring.
    """
    if path is None:
        return {}
    data = json.loads(path.read_text())
    return {doc_id: set(fields) for doc_id, fields in data.get("exclusions", {}).items()}


def score_model_mode(model, mode, mode_dir, gt_by_id, doc_split, field_exclusions, output_dir, order_invariant=True):
    """Score a single model/mode combination. Thread-safe.

    Iterates over GT documents (not pred files) to avoid survivor bias —
    documents with missing predictions are scored as fully incorrect.
    """
    # Check if this mode has any predictions at all (avoid scoring empty modes)
    if not any(mode_dir.glob("*.pred.json")):
        return None

    rows = []
    for doc_id, gt in gt_by_id.items():
        pred_file = mode_dir / f"{doc_id}.pred.json"
        if pred_file.exists():
            try:
                pred = json.loads(pred_file.read_text())
            except (json.JSONDecodeError, ValueError):
                pred = {}
        else:
            pred = {}  # Missing prediction → scored as all-zero
        if order_invariant and "_error" not in pred:
            try:
                pred = align_arrays(gt, pred)
            except Exception:
                pass  # Fall back to unaligned if alignment fails
        fg = flatten(gt)
        try:
            fp = flatten(pred) if "_error" not in pred else {}
        except (AttributeError, TypeError):
            fp = {}

        excluded_fields = field_exclusions.get(doc_id, set())

        for key, gv in fg.items():
            if gv is None or gv == "" or gv == []: continue
            if key in excluded_fields: continue
            pv = fp.get(key)
            scores = score_field(gv, pv)
            rows.append({"doc_id": doc_id, "field": key,
                         "gt_value": gv, "pred_value": pv,
                         "gt_normalized": norm_full(gv),
                         "pred_normalized": norm_full(pv), **scores})

    if not rows:
        return None

    # Write per-field CSV
    csv_path = output_dir / f"{model}_{mode}.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)

    # Per-doc scores
    doc_fields = defaultdict(list)
    for r in rows:
        doc_fields[r["doc_id"]].append(int(r["exact_match_normalized"]))

    # Per-split aggregation
    split_field_correct = defaultdict(int)
    split_field_total = defaultdict(int)
    split_perfect = defaultdict(int)
    split_doc_total = defaultdict(int)

    for doc_id, field_scores in doc_fields.items():
        split = doc_split.get(doc_id, "Flat")
        correct = sum(field_scores)
        total = len(field_scores)
        split_field_correct[split] += correct
        split_field_total[split] += total
        split_doc_total[split] += 1
        if correct == total:
            split_perfect[split] += 1

    n = len(rows)
    all_correct = sum(split_field_correct.values())
    all_docs = sum(split_doc_total.values())
    all_perfect = sum(split_perfect.values())
    overall_em = all_correct / n if n else 0

    # Build output lines
    lines = []
    lines.append(f"  {model}/{mode}: {n} fields")
    lines.append(f"    VAREX-Bench:          {overall_em:.1%}  (perfect: {all_perfect/all_docs:.1%} {all_perfect}/{all_docs})")
    for split in ["Flat", "Nested", "Table"]:
        fc = split_field_correct[split]
        ft = split_field_total[split]
        pc = split_perfect[split]
        dt = split_doc_total[split]
        em = fc / ft if ft else 0
        perf_pct = pc / dt if dt else 0
        lines.append(f"      VAREX-Bench-{split:<7s} {em:.1%}  (perfect: {perf_pct:.1%} {pc}/{dt})")

    # Build summary row
    agg = {
        "model": model, "mode": mode,
        "docs": all_docs, "fields": n,
        "exact_match": sum(r["exact_match"] for r in rows) / n,
        "exact_match_ci": sum(r["exact_match_ci"] for r in rows) / n,
        "exact_match_normalized": overall_em,
        "anls": sum(r["anls"] for r in rows) / n,
        "anls_ci": sum(r["anls_ci"] for r in rows) / n,
        "anls_normalized": sum(r["anls_normalized"] for r in rows) / n,
        "perfect_docs": all_perfect,
        "perfect_docs_pct": all_perfect / all_docs if all_docs else 0,
    }
    for split in ["Flat", "Nested", "Table"]:
        ft = split_field_total[split]
        fc = split_field_correct[split]
        dt = split_doc_total[split]
        pc = split_perfect[split]
        sl = split.lower()
        agg[f"em_{sl}"] = fc / ft if ft else 0
        agg[f"docs_{sl}"] = dt
        agg[f"perfect_{sl}"] = pc
        agg[f"perfect_{sl}_pct"] = pc / dt if dt else 0

    return {"lines": lines, "agg": agg}


def main():
    parser = argparse.ArgumentParser(description="VAREX Benchmark scoring")
    parser.add_argument("results_dir", type=Path)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--field-exclusions", type=Path, required=True,
                        help="JSON file with fields to exclude from scoring (required — use evaluation/field_exclusions.json)")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--no-order-invariant", action="store_true",
                        help="Disable order-invariant array matching (strict index comparison)")
    args = parser.parse_args()
    args.order_invariant = not args.no_order_invariant

    output_dir = args.output_dir or args.results_dir / "scores"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load active doc IDs
    active_ids = None
    if args.manifest:
        manifest = json.loads(args.manifest.read_text())
        active_ids = set(manifest.get("active_doc_ids") or manifest.get("doc_ids", []))

    # Load field exclusions
    field_exclusions = load_field_exclusions(args.field_exclusions)
    if field_exclusions:
        total_excluded = sum(len(v) for v in field_exclusions.values())
        print(f"Field exclusions loaded: {total_excluded} fields in {len(field_exclusions)} docs")

    # Load GT and classify each doc
    gt_by_id = {}
    doc_split = {}
    for doc_dir in sorted(args.data_dir.iterdir()):
        if not doc_dir.is_dir(): continue
        doc_id = doc_dir.name
        if active_ids and doc_id not in active_ids: continue
        gt_path = doc_dir / "ground_truth.json"
        if not gt_path.exists(): continue
        gt = json.loads(gt_path.read_text())
        gt_by_id[doc_id] = gt
        fg = flatten(gt)
        non_null_keys = [k for k, v in fg.items() if v is not None and v != "" and v != []]
        doc_split[doc_id] = classify_doc(non_null_keys)

    # Print split distribution
    split_counts = defaultdict(int)
    for s in doc_split.values():
        split_counts[s] += 1
    print(f"VAREX-Bench: {len(doc_split)} documents")
    for split in ["Flat", "Nested", "Table"]:
        print(f"  VAREX-Bench-{split}: {split_counts[split]}")
    print()

    # Discover all model/mode combos
    tasks = []
    for model_dir in sorted(args.results_dir.iterdir()):
        if not model_dir.is_dir() or model_dir.name == "scores": continue
        for mode_dir in sorted(model_dir.iterdir()):
            if not mode_dir.is_dir(): continue
            tasks.append((model_dir.name, mode_dir.name, mode_dir))

    # Score all model/mode combos in parallel
    results = {}
    with ThreadPoolExecutor(max_workers=min(len(tasks), 16)) as pool:
        futures = {
            pool.submit(score_model_mode, model, mode, mode_dir, gt_by_id, doc_split, field_exclusions, output_dir, args.order_invariant): (model, mode)
            for model, mode, mode_dir in tasks
        }
        for fut in as_completed(futures):
            key = futures[fut]
            result = fut.result()
            if result:
                results[key] = result

    # Print results in sorted order
    summary_rows = []
    for model, mode, _ in tasks:
        key = (model, mode)
        if key not in results: continue
        for line in results[key]["lines"]:
            print(line)
        summary_rows.append(results[key]["agg"])

    if summary_rows:
        summary_path = output_dir / "summary.csv"
        with open(summary_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=summary_rows[0].keys())
            w.writeheader()
            w.writerows(summary_rows)
        print(f"\nSummary: {summary_path}")


if __name__ == "__main__":
    main()
