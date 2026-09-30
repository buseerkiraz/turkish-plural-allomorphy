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


def corpus_tokens(lexeme, freq):
    """Corpus frequency of a lexeme: the bare form plus its plural forms in both
    harmony series (FREQ_FORMS), summed over spelling variants. This is THE
    token count of the project: vocabulary sampling and token presentation
    (Stages 09-15) and the type-vs-token rates (Stages 14-15) all use it.

    Plural forms of both series are included so that the count cannot favour
    either class. Case forms are left out; adding them (as an earlier Stage 14
    did) would raise most counts, but it would change what the models were
    trained on, so any such refinement has to change it here, for everyone."""
    return sum(freq.get(v + s, 0) for v in C.orth_variants(lexeme) for s in FREQ_FORMS)


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


WEIGHTINGS = ["type", "token", "logtoken"]


def epoch_sampler(vocab, weighting):
    """Return f(rng) -> the words presented in one epoch.

    type      every word once, in random order (the scheme of Stages 09 and 11)
    token     len(vocab) draws with replacement, P proportional to corpus
              frequency: common words are practised more, as in real exposure
    logtoken  the same with P proportional to log(1 + frequency), the usual
              compromise between the two

    "type" consumes the generator exactly as Stages 09 and 11 always have, so
    their results are unchanged by this option existing.
    """
    if weighting == "type":
        def sample(rng):
            order = vocab[:]
            rng.shuffle(order)
            return order
        return sample
    if weighting == "token":
        w = [float(x["freq"]) for x in vocab]
    elif weighting == "logtoken":
        w = [math.log1p(x["freq"]) for x in vocab]
    else:
        raise ValueError("unknown weighting: %s" % weighting)
    cum, total = [], 0.0
    for x in w:
        total += x
        cum.append(total)
    return lambda rng: rng.choices(vocab, cum_weights=cum, k=len(vocab))


def load():
    """(all words, words heard in the corpus, pool). The pool holds the
    sampling frame per lateral condition, the nonce vectors per cue level and
    every back-vowel /at/ word."""
    rows = C.read_tsv(os.path.join(OUT, "06_model_matrix.tsv"))
    freq = C.load_freq(FREQ)
    words = []
    for r in rows:
        f = corpus_tokens(r["lexeme"], freq)
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


def level_segments():
    """{cue level: set of TELL segments before -at} as used by the nonce items
    (Stage 07): hiatus {a}, /h/ {h}, velar dorsal {k}, no cue {s, m, ɾ}. The
    definition every lexical comparison uses, so lexicon and nonce items match."""
    segs = collections.defaultdict(set)
    for r in C.read_tsv(os.path.join(OUT, "07_nonce_items.tsv")):
        segs[r["cue_level"]].add(r["transcription"][-3])   # segment before -at
    return segs


def lexicon_counts():
    """{cue level: (N words, N taking -ler)} over Stage 03's cleaned /at/ class,
    pooled over the segments each nonce level uses (level_segments)."""
    segs = level_segments()
    at = C.read_tsv(os.path.join(OUT, "03_at_class.tsv"))
    counts = {}
    for lv in LEVELS:
        hit = [r for r in at if r["pre_final_v"] in segs[lv]]
        counts[lv] = (len(hit), sum(r["status"] == "EXCEPTION" for r in hit))
    return counts


def lexicon_rates():
    """Lexical -ler rate (%) per nonce cue level (see lexicon_counts)."""
    return {lv: 100 * e / n for lv, (n, e) in lexicon_counts().items()}


def vocab_summary(vocab):
    """Counts of /at/ words in a vocabulary by onset group and label."""
    return dict(collections.Counter((w["at_group"], w["y"]) for w in vocab if w["at"]))
