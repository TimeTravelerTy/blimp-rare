from lemminflect import getInflection

def inflect_noun(lemma, tag):
    # tag is "NN" or "NNS"
    out = getInflection(lemma, tag=tag)
    return out[0] if out else None