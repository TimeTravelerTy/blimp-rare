from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

DEFAULT_GENDER_LEXICON_PATH = Path("data/processed/wiktionary_gender_lemmas.json")


@dataclass(frozen=True)
class GenderLexicon:
    """
    Thin wrapper that exposes gender lookups for lemmas harvested from Wiktionary.
    """

    female: Tuple[str, ...]
    male: Tuple[str, ...]
    metadata: Dict[str, object]

    def __post_init__(self):
        object.__setattr__(self, "_female_set", frozenset(self.female))
        object.__setattr__(self, "_male_set", frozenset(self.male))

    def has_data(self) -> bool:
        return bool(self.female or self.male)

    def lemma_gender(self, lemma: Optional[str]) -> Optional[str]:
        norm = (lemma or "").strip().lower()
        if not norm:
            return None
        if norm in self._female_set:
            return "female"
        if norm in self._male_set:
            return "male"
        return None

    def filter_by_gender(self, lemmas: Sequence[str], gender: Optional[str]) -> Tuple[str, ...]:
        if not lemmas or not gender:
            return tuple()
        if gender == "female":
            allowed = self._female_set
        elif gender == "male":
            allowed = self._male_set
        else:
            return tuple()
        return tuple(lemma for lemma in lemmas if lemma in allowed)

    def iter_gender(self, gender: Optional[str]) -> Tuple[str, ...]:
        if gender == "female":
            return self.female
        if gender == "male":
            return self.male
        return tuple()


def _normalize_payload(data: Dict[str, object]) -> Dict[str, Tuple[str, ...]]:
    if "lemmas" in data and isinstance(data["lemmas"], dict):
        female = tuple(data["lemmas"].get("female", []))
        male = tuple(data["lemmas"].get("male", []))
    else:
        female = tuple(data.get("female", []))
        male = tuple(data.get("male", []))
    return {"female": female, "male": male}


def load_gender_lexicon(path: Optional[Path | str] = None) -> GenderLexicon:
    target = Path(path) if path else DEFAULT_GENDER_LEXICON_PATH
    if not target.exists():
        return GenderLexicon(tuple(), tuple(), {"path": str(target), "loaded": False})
    with target.open("r", encoding="utf-8") as f:
        data = json.load(f)
    buckets = _normalize_payload(data)
    metadata = {
        "path": str(target),
        "loaded": True,
        "dump": data.get("dump"),
        "generated_at": data.get("generated_at"),
        "counts": data.get("counts"),
    }
    return GenderLexicon(buckets["female"], buckets["male"], metadata)
