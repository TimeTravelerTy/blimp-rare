import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
try:
    from src.sentence_nll import LlamaNLLScorer  # noqa: E402
except ModuleNotFoundError as e:  # pragma: no cover
    LlamaNLLScorer = None  # type: ignore[assignment]
    _SENTENCE_NLL_IMPORT_ERROR = e
else:  # pragma: no cover
    _SENTENCE_NLL_IMPORT_ERROR = None

TIMESTAMP_RE = re.compile(r"^(\d{8}-\d{6})_")


def _iter_jsonl(path: str) -> Iterable[dict]:
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            rec["_row"] = i
            yield rec


def _load_jsonl(path: str, limit: Optional[int] = None) -> List[dict]:
    records: List[dict] = []
    for rec in _iter_jsonl(path):
        records.append(rec)
        if limit is not None and len(records) >= limit:
            break
    return records


def _pick_field(records: List[dict], explicit: Optional[str], candidates: List[str]) -> Optional[str]:
    if explicit:
        return explicit
    for c in candidates:
        for rec in records:
            if c in rec and rec[c] is not None:
                return c
    return None


def _select_fields(
    variant: str,
    records: List[dict],
    good_field: Optional[str],
    bad_field: Optional[str],
):
    if variant == "original":
        good_candidates = ["good_original", "sentence_good", "good", "grammatical"]
        bad_candidates = ["bad_original", "sentence_bad", "bad", "ungrammatical"]
    elif variant == "rare":
        good_candidates = ["good_rare", "sentence_good", "good", "grammatical"]
        bad_candidates = ["bad_rare", "sentence_bad", "bad", "ungrammatical"]
    else:
        good_candidates = [
            "good_rare",
            "good_original",
            "sentence_good",
            "good",
            "grammatical",
        ]
        bad_candidates = [
            "bad_rare",
            "bad_original",
            "sentence_bad",
            "bad",
            "ungrammatical",
        ]

    picked_good = _pick_field(records, good_field, good_candidates)
    picked_bad = _pick_field(records, bad_field, bad_candidates)
    if not picked_good or not picked_bad:
        raise ValueError(
            "Could not infer good/bad fields; set --good-field and --bad-field."
        )
    return picked_good, picked_bad


def _model_slug(model: str) -> str:
    return model.split("/")[-1].replace(".", "_")


def _data_slug(data: str) -> str:
    return Path(data).name.rsplit(".", 1)[0]


def _default_out_path(
    model: str,
    data: str,
    variant: str,
    run_ts: str,
    out_dir: Path,
) -> Path:
    model_slug = _model_slug(model)
    data_slug = _data_slug(data)
    name = f"{run_ts}_{model_slug}_{data_slug}_{variant}_pair-scores.jsonl"
    return out_dir / name


def _swap_counts(meta: dict) -> Dict[str, int]:
    def _len(key: str) -> int:
        val = meta.get(key)
        return len(val) if isinstance(val, list) else 0

    return {
        "g_swaps": _len("g_swaps"),
        "b_swaps": _len("b_swaps"),
        "g_verb_swaps": _len("g_verb_swaps"),
        "b_verb_swaps": _len("b_verb_swaps"),
        "g_adj_swaps": _len("g_adj_swaps"),
        "b_adj_swaps": _len("b_adj_swaps"),
    }


def _write_variant(
    scorer: LlamaNLLScorer,
    variant: str,
    records: List[dict],
    good_field: Optional[str],
    bad_field: Optional[str],
    data_path: Path,
    model: str,
    args,
    run_ts: str,
    out_dir: Path,
) -> Optional[Path]:
    picked_good, picked_bad = _select_fields(variant, records, good_field, bad_field)
    texts: List[str] = []
    items: List[dict] = []
    skipped = 0
    for rec in records:
        good = rec.get(picked_good)
        bad = rec.get(picked_bad)
        if not isinstance(good, str) or not isinstance(bad, str):
            skipped += 1
            continue
        texts.append(good)
        texts.append(bad)
        items.append(rec)

    if not texts:
        raise ValueError("No valid good/bad pairs found.")

    out_path = _default_out_path(model, str(data_path), variant, run_ts, out_dir)
    if out_path.exists() and not args.overwrite:
        print(f"[Skip] {out_path} already exists.")
        return None

    scores = scorer.score_texts(
        texts,
        batch_size=args.batch_size,
        max_length=args.max_length,
        show_progress=True,
    )
    if len(scores) != len(texts):
        raise RuntimeError("Scoring returned a different number of results.")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for i, rec in enumerate(items):
            meta = rec.get("meta") if isinstance(rec.get("meta"), dict) else {}
            good_score = scores[2 * i]
            bad_score = scores[2 * i + 1]
            row = {
                "idx": rec.get("idx"),
                "group": rec.get("group"),
                "phenomenon": rec.get("phenomenon"),
                "subtask": rec.get("subtask"),
                "variant": variant,
                "good_field": picked_good,
                "bad_field": picked_bad,
                "good_text": rec.get(picked_good),
                "bad_text": rec.get(picked_bad),
                "good_total_nll": good_score.total_nll,
                "bad_total_nll": bad_score.total_nll,
                "good_token_count": good_score.token_count,
                "bad_token_count": bad_score.token_count,
                "swap_counts": _swap_counts(meta),
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"[Saved] {variant} pair scores to {out_path}")
    if skipped:
        print(f"[Warn] Skipped {skipped} records with missing good/bad text for {variant}.")
    return out_path


def _collect_paths(data_dir: Path, timestamp: Optional[str], pattern: Optional[str]) -> List[Path]:
    if pattern:
        paths = [Path(p) for p in Path().glob(pattern)]
    else:
        if not timestamp:
            candidates = []
            for p in data_dir.glob("*.jsonl"):
                match = TIMESTAMP_RE.match(p.name)
                if match:
                    candidates.append(match.group(1))
            if not candidates:
                raise SystemExit(f"No timestamped datasets found in {data_dir}")
            timestamp = sorted(candidates)[-1]
            print(f"[Info] Using latest timestamp: {timestamp}")
        paths = list(data_dir.glob(f"{timestamp}_*.jsonl"))

    paths = [p for p in paths if p.is_file()]
    if not paths:
        raise SystemExit("No datasets found for the provided timestamp/pattern.")
    return sorted(paths)


def _parse_models(models: Sequence[str]) -> List[str]:
    if len(models) == 1 and "," in models[0]:
        return [m.strip() for m in models[0].split(",") if m.strip()]
    return list(models)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Batch BLiMP pair scoring for datasets sharing a timestamp."
    )
    ap.add_argument(
        "--timestamp",
        default=None,
        help="Dataset timestamp prefix (e.g., 20251224-185134). Defaults to latest.",
    )
    ap.add_argument(
        "--pattern",
        default=None,
        help="Optional glob pattern for datasets (overrides --timestamp).",
    )
    ap.add_argument(
        "--data-dir",
        default="data/processed",
        help="Directory containing processed datasets.",
    )
    ap.add_argument(
        "--models",
        nargs="+",
        default=[
            "meta-llama/Llama-3.2-1B",
            "meta-llama/Llama-3.2-3B",
            "meta-llama/Llama-3.1-8B",
            "meta-llama/Llama-3.1-8B-Instruct",
        ],
        help="Models to score (space-separated or comma-separated).",
    )
    ap.add_argument(
        "--variant",
        default="rare",
        choices=["rare", "original", "auto", "both"],
        help="Which sentence variant to score (default: rare).",
    )
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--max-length", type=int, default=256)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--dtype", default="auto", choices=["auto", "float16", "bfloat16", "float32"])
    ap.add_argument("--device-map", default=None, help="Optional HF device_map (e.g., auto).")
    ap.add_argument("--compile-model", action="store_true", help="Use torch.compile when available.")
    ap.add_argument("--good-field", default=None, help="Field for grammatical sentences.")
    ap.add_argument("--bad-field", default=None, help="Field for ungrammatical sentences.")
    ap.add_argument(
        "--out-dir",
        default="results/blimp_pair_scores",
        help="Output directory for pair score JSONL files.",
    )
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs.")
    args = ap.parse_args()

    if LlamaNLLScorer is None:
        raise ModuleNotFoundError(
            "sentence_nll could not be imported; check dependencies"
        ) from _SENTENCE_NLL_IMPORT_ERROR

    data_dir = Path(args.data_dir)
    paths = _collect_paths(data_dir, args.timestamp, args.pattern)
    models = _parse_models(args.models)
    out_dir = Path(args.out_dir)
    run_ts = time.strftime("%Y%m%d-%H%M%S")

    completed: List[Path] = []
    for model in models:
        print(f"\n[Model] Loading {model}")
        scorer = LlamaNLLScorer(
            model_name=model,
            device=args.device,
            dtype=args.dtype,
            device_map=args.device_map,
            compile_model=args.compile_model,
        )
        for data_path in paths:
            print(f"[Run] {data_path} ({args.variant})")
            records = _load_jsonl(str(data_path), limit=args.limit)
            if not records:
                raise RuntimeError(f"No records loaded from {data_path}")
            if args.variant == "both":
                out_path = _write_variant(
                    scorer,
                    "original",
                    records,
                    args.good_field,
                    args.bad_field,
                    data_path,
                    model,
                    args,
                    run_ts,
                    out_dir,
                )
                if out_path:
                    completed.append(out_path)
                out_path = _write_variant(
                    scorer,
                    "rare",
                    records,
                    args.good_field,
                    args.bad_field,
                    data_path,
                    model,
                    args,
                    run_ts,
                    out_dir,
                )
                if out_path:
                    completed.append(out_path)
            else:
                out_path = _write_variant(
                    scorer,
                    args.variant,
                    records,
                    args.good_field,
                    args.bad_field,
                    data_path,
                    model,
                    args,
                    run_ts,
                    out_dir,
                )
                if out_path:
                    completed.append(out_path)

    print("\nCompleted runs:")
    for p in completed:
        print(f"  {p}")


if __name__ == "__main__":
    main()
