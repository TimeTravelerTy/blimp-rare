from enum import Enum

class CountClass(Enum):
    COUNT = "COUNT"
    MASS = "MASS"
    FLEX = "FLEX"
    UNKNOWN = "UNKNOWN"

def load_becl_tsv(path):
    mp = {}
    with open(path, encoding="utf-8") as f:
        header = f.readline()
        for line in f:
            if not line.strip():
                continue
            lemma, cls = line.strip().split("\t")
            cls = cls.upper()
            if cls not in {"COUNT","MASS","FLEX"}:
                cls = "UNKNOWN"
            mp[lemma.lower()] = CountClass(cls)
    return mp

def class_of(lemma, mp):
    return mp.get(lemma.lower(), CountClass.UNKNOWN)