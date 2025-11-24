import argparse
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.perplexity import LlamaPerplexityScorer  # noqa: E402


Variant = str
VARIANTS: Sequence[Variant] = ("good_typical", "bad_typical", "good_rare", "bad_rare")


def _mean(xs: List[float]) -> float:
    return float(sum(xs) / len(xs)) if xs else float("nan")


def _median(xs: List[float]) -> float:
    if not xs:
        return float("nan")
    sorted_xs = sorted(xs)
    mid = len(sorted_xs) // 2
    if len(sorted_xs) % 2:
        return float(sorted_xs[mid])
    return float((sorted_xs[mid - 1] + sorted_xs[mid]) / 2)


def load_records(path: str, limit: Optional[int] = None) -> List[dict]:
    records = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            rec["_row"] = i
            records.append(rec)
            if limit is not None and len(records) >= limit:
                break
    return records


def _build_items(records: List[dict]):
    items = []
    for pos, rec in enumerate(records):
        for variant in VARIANTS:
            text = rec.get(variant)
            if not text:
                continue
            items.append(
                {
                    "variant": variant,
                    "text": text,
                    "record_idx": rec.get("idx", pos),
                    "row": rec.get("_row", pos),
                    "group": rec.get("group"),
                    "phenomenon": rec.get("phenomenon"),
                    "subtask": rec.get("subtask"),
                }
            )
    return items


def _aggregate_by_variant(scored_items: List[dict]) -> Dict[Variant, Dict[str, float]]:
    stats: Dict[Variant, Dict[str, float]] = {}
    for variant in VARIANTS:
        subset = [it for it in scored_items if it["variant"] == variant]
        ppls = [it["ppl"] for it in subset]
        tokens = [it["token_count"] for it in subset]
        stats[variant] = {
            "count": len(subset),
            "mean_ppl": _mean(ppls),
            "median_ppl": _median(ppls),
            "mean_tokens": _mean(tokens),
        }
    return stats


def _pairwise_stats(per_record: Dict[int, Dict[Variant, dict]], typical: Variant, rare: Variant):
    deltas = []
    ratios = []
    rare_higher = 0
    for variants in per_record.values():
        if typical not in variants or rare not in variants:
            continue
        t = variants[typical]["ppl"]
        r = variants[rare]["ppl"]
        deltas.append(r - t)
        if t > 0:
            ratios.append(r / t)
        rare_higher += int(r > t)
    total = len(deltas)
    return {
        "pairs": total,
        "pct_rare_higher": (rare_higher / total * 100.0) if total else float("nan"),
        "mean_delta": _mean(deltas),
        "median_delta": _median(deltas),
        "mean_ratio": _mean(ratios),
    }


def _good_bad_stats(per_record: Dict[int, Dict[Variant, dict]], good: Variant, bad: Variant):
    deltas = []
    ratios = []
    bad_higher = 0
    for variants in per_record.values():
        if good not in variants or bad not in variants:
            continue
        g = variants[good]["ppl"]
        b = variants[bad]["ppl"]
        deltas.append(b - g)
        if g > 0:
            ratios.append(b / g)
        bad_higher += int(b > g)
    total = len(deltas)
    return {
        "pairs": total,
        "pct_bad_higher": (bad_higher / total * 100.0) if total else float("nan"),
        "mean_delta": _mean(deltas),
        "median_delta": _median(deltas),
        "mean_ratio": _mean(ratios),
    }


def _subtask_deltas(scored_items: List[dict], typical: Variant, rare: Variant, top_k: int = 8):
    grouped: Dict[str, Dict[Variant, List[float]]] = defaultdict(lambda: defaultdict(list))
    for item in scored_items:
        grouped[item.get("subtask") or "unknown"][item["variant"]].append(item["ppl"])
    rows = []
    for subtask, variants in grouped.items():
        if typical not in variants or rare not in variants:
            continue
        t_mean = _mean(variants[typical])
        r_mean = _mean(variants[rare])
        ratio = r_mean / t_mean if t_mean > 0 else float("nan")
        rows.append(
            {
                "subtask": subtask,
                "typical_mean": t_mean,
                "rare_mean": r_mean,
                "delta": r_mean - t_mean,
                "ratio": ratio,
            }
        )
    rows.sort(key=lambda r: (1, 0.0) if math.isnan(r["ratio"]) else (0, -r["ratio"]))
    return rows[:top_k]


def _save_details(path: Optional[str], scored_items: List[dict]):
    if not path:
        return
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for item in scored_items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/processed/pilot_tierA.jsonl")
    ap.add_argument("--model", default="meta-llama/Meta-Llama-3-8B")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-length", type=int, default=256)
    ap.add_argument("--limit", type=int, default=None, help="Optional limit on number of records to score.")
    ap.add_argument("--device", default=None, help="torch device (default: cuda if available).")
    ap.add_argument("--dtype", default="auto", choices=["auto", "bfloat16", "float16", "float32"])
    ap.add_argument(
        "--device-map",
        default=None,
        help="Pass through to transformers.from_pretrained device_map (e.g., 'auto' for multi-GPU).",
    )
    ap.add_argument("--compile", action="store_true", help="Try torch.compile for extra throughput on CUDA.")
    ap.add_argument("--details-out", default=None, help="Optional JSONL to write per-variant perplexities.")
    args = ap.parse_args()

    records = load_records(args.data, args.limit)
    print(f"Loaded {len(records)} records from {args.data}.")
    items = _build_items(records)
    print(f"Scoring {len(items)} variants across {len(VARIANTS)} buckets...")

    scorer = LlamaPerplexityScorer(
        model_name=args.model,
        device=args.device,
        dtype=args.dtype,
        device_map=args.device_map,
        compile_model=args.compile,
    )
    scores = scorer.score_texts(
        [item["text"] for item in items],
        batch_size=args.batch_size,
        max_length=args.max_length,
    )
    for item, score in zip(items, scores):
        item.update(
            {
                "ppl": score.ppl,
                "avg_nll": score.avg_nll,
                "total_nll": score.total_nll,
                "token_count": score.token_count,
            }
        )

    per_record: Dict[int, Dict[Variant, dict]] = defaultdict(dict)
    for item in items:
        per_record[item["row"]][item["variant"]] = item

    variant_stats = _aggregate_by_variant(items)
    rare_good = _pairwise_stats(per_record, "good_typical", "good_rare")
    rare_bad = _pairwise_stats(per_record, "bad_typical", "bad_rare")
    good_vs_bad_typical = _good_bad_stats(per_record, "good_typical", "bad_typical")
    good_vs_bad_rare = _good_bad_stats(per_record, "good_rare", "bad_rare")
    subtask_rows = _subtask_deltas(items, "good_typical", "good_rare")

    print("\nVariant perplexities:")
    for variant in VARIANTS:
        stats = variant_stats[variant]
        print(
            f"  {variant:12} count={stats['count']:5d} "
            f"mean_ppl={stats['mean_ppl']:.2f} median_ppl={stats['median_ppl']:.2f} "
            f"mean_tokens={stats['mean_tokens']:.1f}"
        )

    print("\nRare vs typical deltas:")
    print(
        f"  good: pairs={rare_good['pairs']:5d} pct_rare_higher={rare_good['pct_rare_higher']:.1f}% "
        f"mean_delta={rare_good['mean_delta']:.2f} median_delta={rare_good['median_delta']:.2f} "
        f"mean_ratio={rare_good['mean_ratio']:.3f}"
    )
    print(
        f"  bad : pairs={rare_bad['pairs']:5d} pct_rare_higher={rare_bad['pct_rare_higher']:.1f}% "
        f"mean_delta={rare_bad['mean_delta']:.2f} median_delta={rare_bad['median_delta']:.2f} "
        f"mean_ratio={rare_bad['mean_ratio']:.3f}"
    )

    print("\nGood vs bad checks:")
    print(
        f"  typical: pairs={good_vs_bad_typical['pairs']:5d} pct_bad_higher={good_vs_bad_typical['pct_bad_higher']:.1f}% "
        f"mean_delta={good_vs_bad_typical['mean_delta']:.2f} median_delta={good_vs_bad_typical['median_delta']:.2f} "
        f"mean_ratio={good_vs_bad_typical['mean_ratio']:.3f}"
    )
    print(
        f"  rare   : pairs={good_vs_bad_rare['pairs']:5d} pct_bad_higher={good_vs_bad_rare['pct_bad_higher']:.1f}% "
        f"mean_delta={good_vs_bad_rare['mean_delta']:.2f} median_delta={good_vs_bad_rare['median_delta']:.2f} "
        f"mean_ratio={good_vs_bad_rare['mean_ratio']:.3f}"
    )

    if subtask_rows:
        print("\nLargest rare/typical gaps by subtask (good sentences):")
        for row in subtask_rows:
            print(
                f"  {row['subtask']}: delta={row['delta']:.2f} ratio={row['ratio']:.3f} "
                f"(typical_mean={row['typical_mean']:.2f}, rare_mean={row['rare_mean']:.2f})"
            )

    _save_details(args.details_out, items)


if __name__ == "__main__":
    main()
