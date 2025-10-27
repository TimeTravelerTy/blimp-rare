from datasets import load_dataset
import json, pathlib

def load_blimp(config_name):
    # BLiMP requires a config like 'regular_plural_subject_verb_agreement_1'
    return load_dataset("nyu-mll/blimp", config_name)["train"]

def write_jsonl(path, records):
    p = pathlib.Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
