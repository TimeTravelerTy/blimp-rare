from wordfreq import zipf_frequency


def is_rare_lemma(lemma, thr=3.4, zipf_min=None):
    """
    Return True when lemma falls below an upper Zipf threshold, and above a
    lower bound if provided.
    """
    z = zipf_frequency(lemma, "en")
    if zipf_min is not None and z < zipf_min:
        return False
    if thr is None:
        return True
    return z < thr
