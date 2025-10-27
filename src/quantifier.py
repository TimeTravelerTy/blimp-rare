import yaml

def load_quant_rules(path):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)

def requirement(doc, rules):
    count_triggers = set(w.lower() for w in rules["count_triggers"])
    mass_triggers  = set(w.lower() for w in rules["mass_triggers"])
    toks = {t.text.lower() for t in doc}
    if toks & count_triggers:
        return "COUNT"
    if toks & mass_triggers:
        return "MASS"
    return None