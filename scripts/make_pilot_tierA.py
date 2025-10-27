import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import argparse, json
from src.pipeline import build_pilot

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier_cfg", default="configs/tierA.yaml")
    ap.add_argument("--becl_path", default="data/external/becl_lemma.tsv")
    ap.add_argument("--quant_cfg", default="configs/quantifier_map.yaml")
    ap.add_argument("--out", default="data/processed/pilot_tierA.jsonl")
    ap.add_argument("--noun_mode", choices=["all","k"], default="all")
    ap.add_argument("--k", type=int, default=2)
    ap.add_argument("--zipf", type=float, default=3.4)
    ap.add_argument("--rare_lemmas", default="[]")  # JSON list
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rare = json.loads(args.rare_lemmas) if args.rare_lemmas else []
    build_pilot(args.tier_cfg, args.becl_path, args.quant_cfg, args.out,
                noun_mode=args.noun_mode, k=args.k, zipf_thr=args.zipf,
                rare_lemmas=rare, seed=args.seed)