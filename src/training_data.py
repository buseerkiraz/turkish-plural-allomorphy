# -*- coding: utf-8 -*-
"""Training and test data shared by every model trained on Turkish.

Stages 09 (ALCOVE) and 11 (RULEX) both take their vocabularies, nonce items and
comparison figures from here, so the two models are trained and tested on
identical inputs. Learner k in a condition gets the same vocabulary sample in
both stages, because the sample is the first thing drawn from a generator
seeded by (vocabulary size, lateral condition, k).
"""
import collections
import math
import os
import random

import common as C

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "output")
FREQ = os.path.join(HERE, "..", "data", "tr_full.txt")

VOCAB_SIZES = [1000, 2000, 5000]
LATERALS = ["with", "without"]
# Frequency = bare form plus plural forms in BOTH harmony series, so the weight
# does not favour either class. Homographs (e.g. 'sat', also 'sell!') inflate
# some counts; that is noise, not bias toward -ler.
FREQ_FORMS = ["", "lar", "ları", "larda", "lardan", "ların",
              "ler", "leri", "lerde", "lerden", "lerin"]
LEVELS = ["hiatus", "/h/", "velar dorsal", "no cue"]
AT_GROUPS = [("hiatus", "D2_onset_vowel"), ("/h/", "D3_onset_h"),
             ("dorsal", "D4_onset_dorsal"), ("none", None)]


def vec(r):
    return tuple(int(r[d]) for d in C.DIMENSION_NAMES)


def at_group(r):
    for name, dim in AT_GROUPS:
        if dim is None or r[dim] == "1":
            return name


def weighted_sample(items, weights, n, rng):
    """n items without replacement, P proportional to weight
    (Efraimidis & Spirakis 2006: keep the n largest u ** (1 / w))."""
    keyed = [(math.log(rng.random()) / w, i) for i, w in enumerate(weights)]
    keyed.sort(reverse=True)
    return [items[i] for _, i in keyed[:n]]


def learner(pool, vocab_size, laterals, k):
    """(rng, vocabulary) for simulated learner k. The rng is returned so the
    model can keep drawing from it for trial order and its own randomness."""
    rng = random.Random("%d-%s-%d" % (vocab_size, laterals, k))
    words, weights = pool[laterals]
    return rng, weighted_sample(words, weights, vocab_size, rng)


def load():
    """(all words, words heard in the corpus, pool). The pool holds the
    sampling frame per lateral condition, the nonce vectors per cue level and
    every back-vowel /at/ word."""
    rows = C.read_tsv(os.path.join(OUT, "06_model_matrix.tsv"))
    freq = C.load_freq(FREQ)
    words = []
    for r in rows:
        f = sum(freq.get(v + s, 0) for v in C.orth_variants(r["lexeme"])
                for s in FREQ_FORMS)
        words.append(dict(lexeme=r["lexeme"], vec=vec(r), y=int(r["plural_ler"]),
                          exc=int(r["is_exception"]), freq=f,
                          lateral=r["neighbourhood"] == "lateral",
                          at=r["analysis_population"] == "Y",
                          at_group=at_group(r)))
    heard = [w for w in words if w["freq"] > 0]
    pool = {}
    for lat in LATERALS:
        ws = [w for w in heard if lat == "with" or not w["lateral"]]
        pool[lat] = (ws, [w["freq"] for w in ws])

    nonce = {}
    for r in C.read_tsv(os.path.join(OUT, "07_nonce_items.tsv")):
        if r["accepted"] == "Y":
            nonce.setdefault(r["cue_level"], vec(r))
    pool["nonce"] = {lv: nonce[lv] for lv in LEVELS}
    pool["at_all"] = [w for w in words if w["at"]]
    return words, heard, pool


def lexicon_rates():
    """Lexical -ler rate (%) per nonce cue level, pooled over the segments each
    level uses, from Stage 03's cleaned /at/ class (the same source as Stage 07)."""
    segs = collections.defaultdict(set)
    for r in C.read_tsv(os.path.join(OUT, "07_nonce_items.tsv")):
        segs[r["cue_level"]].add(r["transcription"][-3])   # segment before -at
    at = C.read_tsv(os.path.join(OUT, "03_at_class.tsv"))
    rates = {}
    for lv in LEVELS:
        hit = [r for r in at if r["pre_final_v"] in segs[lv]]
        rates[lv] = 100 * sum(r["status"] == "EXCEPTION" for r in hit) / len(hit)
    return rates


def vocab_summary(vocab):
    """Counts of /at/ words in a vocabulary by onset group and label."""
    return dict(collections.Counter((w["at_group"], w["y"]) for w in vocab if w["at"]))
