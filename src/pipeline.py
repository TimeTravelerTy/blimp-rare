import random, spacy, yaml
from .io import load_blimp, write_jsonl
from .becl import load_becl_tsv
from .quantifier import load_quant_rules, requirement
from .edits import noun_swap_all
from .invariants import tag_preserved

def build_pilot(tier_cfg_path, becl_path, quant_cfg_path, out_path,
                noun_mode="all", k=2, zipf_thr=3.4, rare_lemmas=None, seed=0):
    base_rng = random.Random(seed)
    nlp = spacy.load("en_core_web_sm")
    with open(tier_cfg_path, encoding="utf-8") as f:
        tasks_cfg = yaml.safe_load(f)
    becl_map = load_becl_tsv(becl_path)
    qrules = load_quant_rules(quant_cfg_path)
    records = []

    for group_name, meta in tasks_cfg.items():
        phenomenon = meta["phenomenon"]
        configs = meta.get("configs", [])
        if not configs:
            continue
        for cfg in configs:
            ds = load_blimp(cfg)
            for i, r in enumerate(ds):
                g, b = r["sentence_good"], r["sentence_bad"]
                gdoc, bdoc = nlp(g), nlp(b)

                # Quantifier requirement per sentence
                req_g = requirement(gdoc, qrules)  # None/COUNT/MASS
                req_b = requirement(bdoc, qrules)

                # Per-pair RNG so Good/Bad use the same sequence of choices
                pair_seed = hash((group_name, cfg, i)) & 0xFFFFFFFF
                rng = random.Random(pair_seed)

                g_rare, g_swaps = noun_swap_all(
                    gdoc, rare_lemmas,
                    noun_mode=noun_mode, k=k, zipf_thr=zipf_thr,
                    becl_map=becl_map, req=req_g, rng=rng
                )
                # Reuse the SAME rng sequence for Bad by re-seeding with the same seed
                rng = random.Random(pair_seed)
                b_rare, b_swaps = noun_swap_all(
                    bdoc, rare_lemmas,
                    noun_mode=noun_mode, k=k, zipf_thr=zipf_thr,
                    becl_map=becl_map, req=req_b, rng=rng
                )

                g_ok = tag_preserved(gdoc, nlp(g_rare)) if g_rare else False
                b_ok = tag_preserved(bdoc, nlp(b_rare)) if b_rare else False

                # Only set good_rare and bad_rare if both are valid
                if g_ok and b_ok:
                    good_rare_val = g_rare
                    bad_rare_val = b_rare
                else:
                    good_rare_val = None
                    bad_rare_val = None

                records.append({
                    "group": group_name,
                    "phenomenon": phenomenon,
                    "subtask": cfg,
                    "idx": i,
                    "good_typical": g,
                    "bad_typical": b,
                    "good_rare": good_rare_val,
                    "bad_rare":  bad_rare_val,
                    "meta": {
                        "g_swaps": g_swaps,
                        "b_swaps": b_swaps,
                        "noun_mode": noun_mode,
                        "k": k,
                        "zipf_thr": zipf_thr,
                        "req_good": req_g,
                        "req_bad": req_b
                    }
                })
    write_jsonl(out_path, records)