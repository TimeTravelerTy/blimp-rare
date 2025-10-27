from wordfreq import zipf_frequency

def is_rare_lemma(lemma, thr=3.4):
    return zipf_frequency(lemma, "en") < thr