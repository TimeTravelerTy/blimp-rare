import csv
import re
from pathlib import Path
from typing import Dict, Set

_GENDER_NORMALIZE = {
    "f": "F",
    "female": "F",
    "feminine": "F",
    "woman": "F",
    "women": "F",
    "girl": "F",
    "girls": "F",
    "m": "M",
    "male": "M",
    "masculine": "M",
    "man": "M",
    "men": "M",
    "boy": "M",
    "boys": "M",
    "neutral": "N",
    "either": "N",
    "unisex": "N",
    "both": "N",
    "any": "N",
    "person": "N",
}


def _normalize_gender_token(token: str):
    norm = token.strip().lower()
    if not norm:
        return None
    return _GENDER_NORMALIZE.get(norm)


def _parse_gender_values(raw: str):
    tokens = re.split(r"[;,/]", raw)
    out = set()
    for token in tokens:
        norm = _normalize_gender_token(token)
        if norm:
            out.add(norm)
    return out


def load_wiki_person_genders(path) -> Dict[str, Set[str]]:
    """
    Load a TSV mapping from lemma to {M,F,N} gender categories derived from wiki data.
    Expected columns: lemma and gender (additional columns are ignored).
    """
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(path)

    mapping: Dict[str, Set[str]] = {}
    with file_path.open(encoding="utf-8") as f:
        peek = f.readline()
        f.seek(0)
        has_header = "lemma" in peek.lower() and "gender" in peek.lower()
        if has_header:
            reader = csv.DictReader(f, delimiter="\t")
        else:
            reader = csv.DictReader(f, delimiter="\t", fieldnames=["lemma", "gender"])
        for row in reader:
            if not row:
                continue
            lemma = (row.get("lemma") or "").strip().lower()
            if not lemma or lemma.startswith("#"):
                continue
            raw_gender = row.get("gender") or ""
            genders = _parse_gender_values(raw_gender)
            if not genders:
                continue
            entry = mapping.setdefault(lemma, set())
            entry.update(genders)
    return mapping


__all__ = ["load_wiki_person_genders"]
