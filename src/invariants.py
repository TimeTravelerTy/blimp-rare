def tag_preserved(orig_doc, pert_doc):
    if pert_doc is None or len(orig_doc) != len(pert_doc):
        return False
    for a, b in zip(orig_doc, pert_doc):
        if a.text != b.text:
            # only nouns change; keep tag NN/NNS identical
            if not (a.pos_=="NOUN" and b.pos_=="NOUN" and a.tag_==b.tag_ and a.tag_ in {"NN","NNS"}):
                return False
    return True