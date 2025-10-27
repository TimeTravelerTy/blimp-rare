import random
from .rarity import is_rare_lemma
from .inflect import inflect_noun
from typing import Optional

def candidate_nouns(doc):
    # Only content nouns; skip PROPN, NE chunks, and ROOT (prevents verb mis-swaps like "reference")
    return [
        t for t in doc
        if t.pos_ == "NOUN"
        and t.tag_ in {"NN", "NNS"}
        and t.is_alpha
        and len(t.text) > 2
        and t.ent_type_ == ""
        and t.dep_ != "ROOT"  # skip main verb even if mis-tagged
        and not (t.head == t and t.dep_ == "ROOT")  # extra guard: skip if token is its own head and ROOT
    ]

def noun_swap_all(
    doc,
    rare_lemmas,
    noun_mode="all",
    k=2,
    zipf_thr=3.4,
    becl_map=None,
    req=None,
    rng: Optional[random.Random]=None,
):
    """
    rare_lemmas: list[str] of real rare noun lemmas to sample from
    becl_map: lemma->CountClass (optional, used when req in {"COUNT","MASS"})
    req: "COUNT" | "MASS" | None
    rng: pass a pre-seeded Random to make Good/Bad use the same sequence
    """
    if rng is None:
        rng = random

    toks = [t.text for t in doc]
    swaps = []

    targets = candidate_nouns(doc)
    # Deterministic order across Good/Bad
    targets.sort(key=lambda t: t.i)

    if not targets:
        return None, swaps

    if noun_mode == "k":
        targets = targets[:max(0, min(k, len(targets)))]

    # Pre-filter pool by rarity (and countability if needed)
    pool = [w for w in rare_lemmas if is_rare_lemma(w, zipf_thr)]
    if req == "COUNT" and becl_map:
        pool = [w for w in pool if str(becl_map.get(w.lower(), "")).endswith(("COUNT", "FLEX"))]
    elif req == "MASS" and becl_map:
        pool = [w for w in pool if str(becl_map.get(w.lower(), "")).endswith(("MASS", "FLEX"))]

    if not pool:
        return None, swaps

    # Choose in a deterministic sequence using rng
    for t in targets:
        lemma = rng.choice(pool)
        form = inflect_noun(lemma, t.tag_)
        if not form:
            continue
        toks[t.i] = form
        swaps.append({"i": t.i, "old": t.text, "new": form, "tag": t.tag_})

    if not swaps:
        return None, swaps

    text = " ".join(toks)
    # light detokenization to clean spaces before punctuation
    text = (
        text.replace(" .", ".")
            .replace(" ,", ",")
            .replace(" !", "!")
            .replace(" ?", "?")
            .replace(" ;", ";")
            .replace(" :", ":")
            .replace(" n't", "n't")
    )
    return text, swaps