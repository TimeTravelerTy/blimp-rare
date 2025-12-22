import argparse
import json
import statistics
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from wordfreq import zipf_frequency


def _zipf(word: str) -> float:
    # wordfreq handles casing + punctuation reasonably; keep this lightweight.
    return float(zipf_frequency(word, "en"))


def _values_to_aggs(values: List[float]) -> Dict[str, Optional[float]]:
    # Treat 0.0 as OOV (unknown to the Zipf lookup) and exclude from aggregates.
    oov_count = sum(1 for v in values if v <= 0.0)
    in_vocab = [v for v in values if v > 0.0]
    if not in_vocab:
        return {"n": 0, "oov_count": oov_count, "mean": None, "median": None, "min": None}
    return {
        "n": len(in_vocab),
        "oov_count": oov_count,
        "mean": sum(in_vocab) / len(in_vocab),
        "median": float(statistics.median(in_vocab)),
        "min": float(min(in_vocab)),
    }


def _delta(a: Optional[float], b: Optional[float]) -> Optional[float]:
    if a is None or b is None:
        return None
    return a - b


def _extract_words_from_swaps(
    swap_items: Iterable[Dict[str, Any]],
    *,
    which: str,
) -> List[str]:
    """
    Extract words used at swapped positions.

    `which` is either "old" (typical) or "new" (rare).
    Also includes verb-adjacent swaps if present (prep/particle).
    """
    words: List[str] = []
    for s in swap_items:
        w = s.get(which)
        if isinstance(w, str) and w:
            words.append(w)

        # Verb swaps may include additional swapped tokens.
        prep_key = f"prep_{which}"
        prep = s.get(prep_key)
        if isinstance(prep, str) and prep:
            words.append(prep)

        particle_key = f"particle_{which}"
        particle = s.get(particle_key)
        if isinstance(particle, str) and particle:
            words.append(particle)
    return words


def add_zipf_aggregates(record: Dict[str, Any]) -> Dict[str, Any]:
    meta = record.get("meta") or {}

    g_swaps = list(meta.get("g_swaps") or [])
    b_swaps = list(meta.get("b_swaps") or [])
    g_adj_swaps = list(meta.get("g_adj_swaps") or [])
    b_adj_swaps = list(meta.get("b_adj_swaps") or [])
    g_verb_swaps = list(meta.get("g_verb_swaps") or [])
    b_verb_swaps = list(meta.get("b_verb_swaps") or [])

    good_typical_words = (
        _extract_words_from_swaps(g_swaps, which="old")
        + _extract_words_from_swaps(g_adj_swaps, which="old")
        + _extract_words_from_swaps(g_verb_swaps, which="old")
    )
    bad_typical_words = (
        _extract_words_from_swaps(b_swaps, which="old")
        + _extract_words_from_swaps(b_adj_swaps, which="old")
        + _extract_words_from_swaps(b_verb_swaps, which="old")
    )
    good_rare_words = (
        _extract_words_from_swaps(g_swaps, which="new")
        + _extract_words_from_swaps(g_adj_swaps, which="new")
        + _extract_words_from_swaps(g_verb_swaps, which="new")
    )
    bad_rare_words = (
        _extract_words_from_swaps(b_swaps, which="new")
        + _extract_words_from_swaps(b_adj_swaps, which="new")
        + _extract_words_from_swaps(b_verb_swaps, which="new")
    )

    zipf_values = {
        "good_typical": [_zipf(w) for w in good_typical_words],
        "bad_typical": [_zipf(w) for w in bad_typical_words],
        "good_rare": [_zipf(w) for w in good_rare_words],
        "bad_rare": [_zipf(w) for w in bad_rare_words],
    }
    aggs = {k: _values_to_aggs(v) for k, v in zipf_values.items()}

    deltas = {
        "good_rare_minus_typical": {
            "mean": _delta(aggs["good_rare"]["mean"], aggs["good_typical"]["mean"]),
            "median": _delta(aggs["good_rare"]["median"], aggs["good_typical"]["median"]),
            "min": _delta(aggs["good_rare"]["min"], aggs["good_typical"]["min"]),
        },
        "bad_rare_minus_typical": {
            "mean": _delta(aggs["bad_rare"]["mean"], aggs["bad_typical"]["mean"]),
            "median": _delta(aggs["bad_rare"]["median"], aggs["bad_typical"]["median"]),
            "min": _delta(aggs["bad_rare"]["min"], aggs["bad_typical"]["min"]),
        },
    }

    meta_out = dict(meta)
    meta_out["zipf_swapped_position_aggregates"] = aggs
    meta_out["zipf_swapped_position_deltas"] = deltas

    out = dict(record)
    out["meta"] = meta_out
    return out


def _iter_paths(pattern: str) -> List[Path]:
    paths = sorted(Path(p) for p in Path().glob(pattern) if Path(p).is_file())
    if not paths:
        raise SystemExit(f"No files found for pattern: {pattern}")
    return paths


def _process_file(path: Path) -> Tuple[int, int, Path]:
    total = 0
    changed = 0
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with path.open("r", encoding="utf-8") as r, tmp_path.open("w", encoding="utf-8") as w:
        for line in r:
            line = line.strip()
            if not line:
                continue
            total += 1
            rec = json.loads(line)
            before_meta = rec.get("meta") or {}
            before = (
                before_meta.get("zipf_swapped_position_aggregates"),
                before_meta.get("zipf_swapped_position_deltas"),
            )
            rec2 = add_zipf_aggregates(rec)
            after_meta = rec2.get("meta") or {}
            after = (
                after_meta.get("zipf_swapped_position_aggregates"),
                after_meta.get("zipf_swapped_position_deltas"),
            )
            if before != after:
                changed += 1
            w.write(json.dumps(rec2, ensure_ascii=False) + "\n")
    return total, changed, tmp_path


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Add Zipf aggregates (mean/median/min) over swapped positions to processed rare-BLIMP JSONL files."
    )
    ap.add_argument(
        "--pattern",
        default="data/processed/*rare_blimp*.jsonl",
        help="Glob pattern for JSONL files to update.",
    )
    ap.add_argument(
        "--inplace",
        action="store_true",
        help="Replace files in-place (writes via a temporary file then renames).",
    )
    ap.add_argument(
        "--backup-suffix",
        default=".bak",
        help="If --inplace, write a backup copy with this suffix (set to empty to disable).",
    )
    args = ap.parse_args()

    paths = _iter_paths(args.pattern)
    for path in paths:
        total, changed, tmp_path = _process_file(path)
        if args.inplace:
            if args.backup_suffix:
                backup_path = path.with_suffix(path.suffix + args.backup_suffix)
                backup_path.write_bytes(path.read_bytes())
            tmp_path.replace(path)
            out_path = path
        else:
            out_path = path.with_name(path.stem + ".with_zipf_aggs.jsonl")
            tmp_path.replace(out_path)
        print(f"{path} -> {out_path} ({changed}/{total} records updated)")


if __name__ == "__main__":
    main()
