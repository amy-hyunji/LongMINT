#!/usr/bin/env python3

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from _common import em_f1, extract_boxed  # noqa: E402


def score_jsonl(path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            pred = extract_boxed(d.get("predicted", ""))
            gold = str(d.get("gold", ""))
            em, f1 = em_f1(pred, [gold])
            rows.append({
                "em":    em,
                "f1":    f1,
                "qtype": d.get("question_type", "unknown"),
                "pred":  pred,
                "gold":  gold,
            })
    return rows


def score_memalpha_json(path):
    rows = []
    with open(path) as f:
        data = json.load(f)
    for d in data:
        pred = extract_boxed(d.get("response", ""))
        gold = str(d.get("answer", ""))
        em, f1 = em_f1(pred, [gold])
        rows.append({
            "em":    em,
            "f1":    f1,
            "qtype": d.get("category") or d.get("question_type") or "unknown",
            "pred":  pred,
            "gold":  gold,
        })
    return rows


def score_hipporag_json(path):
    rows = []
    with open(path) as f:
        data = json.load(f)
    for inst in data:
        pd = inst.get("pred_dict") or {}
        for cat, blk in pd.items():
            for sol in blk.get("solutions", []):
                pred = extract_boxed(sol.get("answer", ""))
                golds = sol.get("gold_answers") or []
                if not golds:
                    continue
                em, f1 = em_f1(pred, golds)
                rows.append({
                    "em":    em,
                    "f1":    f1,
                    "qtype": cat,
                    "pred":  pred,
                    "gold":  golds[0],
                })
    return rows


SCORERS = {
    "jsonl":     score_jsonl,
    "mem_alpha": score_memalpha_json,
    "hipporag":  score_hipporag_json,
}


def autodetect_scorer(path: Path):
    """Pick a scorer based on file shape. JSONL by extension; for .json we
    peek at the top-level structure to disambiguate Mem-alpha (list of
    `{response, answer}`) from HippoRAG (list of `{pred_dict}`)."""
    if path.suffix == ".jsonl":
        return score_jsonl
    with open(path) as f:
        data = json.load(f)
    if isinstance(data, list) and data:
        first = data[0]
        if isinstance(first, dict):
            if "pred_dict" in first:
                return score_hipporag_json
            if "response" in first and "answer" in first:
                return score_memalpha_json
    raise SystemExit(
        f"Couldn't autodetect format for {path}. Pass --backend to override "
        f"(choices: {sorted(SCORERS)})."
    )


def collect_files(target: Path):
    """Return [(file_path, dataset_hint)] from a file or directory.

    Dataset hint comes from the filename stem when no override is given;
    used purely for the --by-dataset grouping in the summary.
    """
    if target.is_file():
        return [(target, target.stem)]
    files = []
    for ext in ("*.jsonl", "*.json"):
        files.extend(target.rglob(ext))
    files = sorted(files)
    return [(p, p.stem) for p in files]


def aggregate(rows):
    if not rows:
        return 0.0, 0.0, 0
    em = sum(r["em"] for r in rows) / len(rows)
    f1 = sum(r["f1"] for r in rows) / len(rows)
    return em, f1, len(rows)


def print_breakdown(rows, key, label):
    buckets = defaultdict(list)
    for r in rows:
        buckets[r[key]].append(r)
    for cat in sorted(buckets):
        em, f1, n = aggregate(buckets[cat])
        print(f"    {label}={cat:32s} n={n:5d}  EM={em:.4f}  F1={f1:.4f}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results", required=True,
                   help="Path to a result file (.json or .jsonl) or a directory "
                        "to walk recursively.")
    p.add_argument("--backend", choices=sorted(SCORERS), default=None,
                   help="Force a specific scorer; otherwise autodetected.")
    p.add_argument("--by-category", action="store_true",
                   help="Print per-question-type EM/F1.")
    p.add_argument("--by-dataset", action="store_true",
                   help="When walking a directory, print per-file (≈per-dataset) "
                        "EM/F1 alongside the overall.")
    p.add_argument("--show-errors", type=int, default=0,
                   help="Print up to N rows where EM == 0 (debugging).")
    args = p.parse_args()

    target = Path(args.results).expanduser().resolve()
    if not target.exists():
        raise SystemExit(f"Not found: {target}")

    files = collect_files(target)
    if not files:
        raise SystemExit(f"No .json / .jsonl files under {target}")

    all_rows = []
    per_file = []
    for path, stem in files:
        scorer = SCORERS[args.backend] if args.backend else autodetect_scorer(path)
        rows = scorer(path)
        for r in rows:
            r["_file"] = stem
        per_file.append((path, scorer.__name__, rows))
        all_rows.extend(rows)

    em, f1, n = aggregate(all_rows)
    print(f"target: {target}")
    print(f"files:  {len(files)}")
    print(f"rows:   {n}")
    print(f"EM:     {em:.4f}")
    # print(f"F1:     {f1:.4f}")

    if args.by_dataset and len(files) > 1:
        print("  by file:")
        for path, scorer_name, rows in per_file:
            em, f1, n = aggregate(rows)
            print(f"    [{scorer_name:>9s}] {path.name:50s} n={n:5d}  EM={em:.4f}  F1={f1:.4f}")

    if args.by_category:
        print("  by qtype:")
        print_breakdown(all_rows, "qtype", "qtype")

    if args.show_errors > 0:
        misses = [r for r in all_rows if r["em"] == 0.0]
        for r in misses[: args.show_errors]:
            print(f"  miss  qtype={r['qtype']}  gold={r['gold']!r}  pred={r['pred']!r}")


if __name__ == "__main__":
    main()
