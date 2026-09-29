#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Stage 15 - does training on word TOKENS change what the models predict?

Stages 09 and 11 present every word in a learner's vocabulary once per epoch
(type presentation): saat and a word heard once a year get equal practice.
Real exposure is token-based. Among the /at/ words this matters, because the
cue levels order differently by tokens than by types:

    by types   vowel > /h/ > /k/     (57% / 39% / 17% -ler)
    by tokens  vowel > /k/ > /h/     (saat and dikkat are very frequent -ler
                                      words; rahat is a very frequent -lar one)

Humans in the survey reportedly follow the token order. So the question for
each model is whether token presentation moves its nonce profile from the type
order to the token order. An exemplar model can in principle follow exposure;
whether the rule-plus-exception model can is tested here, not assumed.

Design: 5,000-word vocabularies with lateral-final words (the settled
condition and the design default), main parameters, the same learners as
Stages 09 and 11 (learner k gets the same vocabulary), three presentation
schemes from training_data.epoch_sampler: type, token, and log-token (the
usual compromise). Only presentation changes; which words a learner knows does
not. ALCOVE 20 learners and RULEX 100 per scheme, as in Stages 09 and 11.

Convergence. Checked to 240 epochs for token and log-token ALCOVE. Their means
wobble by up to about .05 between checkpoints without trend (token /k/: .235,
.324, .253, .325 at epochs 60, 80, 160, 240), because each token epoch is a
fresh random draw in which rare words may or may not appear. More training does
not remove this. The orderings held at every checkpoint: token puts /k/ above
/h/, log-token keeps /h/ above /k/. Read token-trained means as +-.05.

Caveat on the token counts: the corpus counts word forms, not nouns, so some
frequencies include other parts of speech (fakat is mostly the conjunction
"but", rahat mostly the adjective "comfortable", kat also a verb). This
inflates some regular /at/ tokens; it does not change the order above.
"""
import collections
import importlib
import math
import multiprocessing
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C
import training_data as T
from training_data import LEVELS, WEIGHTINGS

S09 = importlib.import_module("09_alcove_turkish")
S11 = importlib.import_module("11_rulex_turkish")

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "output")

VOCAB = 5000
LATERALS = "with"
N_ALCOVE = S09.LEARNERS
N_RULEX = S11.LEARNERS


def _init(pool):
    S09._init(pool)
    S11._init(pool)


def run(job):
    model, k, weighting = job
    if model == "ALCOVE":
        r = S09.run_learner((VOCAB, LATERALS, k, S09.PARAMS, weighting))
    else:
        r = S11.run_learner((VOCAB, LATERALS, k, S11.MAIN, weighting))
    return model, weighting, k, r["nonce"], r["halfway"]


def lexical_profiles(words):
    """-ler rate per nonce level among the cleaned /at/ words (Stage 03), by
    type, by token and by log-token, pooled over each level's segments."""
    freq = {}
    for w in words:
        freq.setdefault(w["lexeme"], w["freq"])
    segs = collections.defaultdict(set)
    for r in C.read_tsv(os.path.join(OUT, "07_nonce_items.tsv")):
        segs[r["cue_level"]].add(r["transcription"][-3])
    at = C.read_tsv(os.path.join(OUT, "03_at_class.tsv"))
    prof = {wt: {} for wt in WEIGHTINGS}
    for lv in LEVELS:
        hit = [r for r in at if r["pre_final_v"] in segs[lv]]
        for wt in WEIGHTINGS:
            f = [(1.0 if wt == "type" else float(freq.get(r["lexeme"], 0)) if wt == "token"
                  else math.log1p(freq.get(r["lexeme"], 0)), r["status"] == "EXCEPTION")
                 for r in hit]
            tot = sum(x for x, _ in f)
            prof[wt][lv] = sum(x for x, e in f if e) / tot if tot else float("nan")
    return prof


def rmsd(a, b):
    return (sum((a[lv] - b[lv]) ** 2 for lv in LEVELS) / len(LEVELS)) ** 0.5


def main():
    words, heard, pool = T.load()
    print("STAGE 15 -- type vs token presentation, ALCOVE and RULEX\n")
    print("  %d-word vocabularies, %s lateral-final words, main parameters, %d epochs"
          % (VOCAB, LATERALS, S09.EPOCHS))

    prof = lexical_profiles(words)
    print("\n  REAL /at/ WORDS: -ler rate per nonce level (cleaned class, Stage 03)")
    print("    %-10s %s   h vs k" % ("counted by", "  ".join("%-12s" % lv for lv in LEVELS)))
    for wt in WEIGHTINGS:
        p = prof[wt]
        print("    %-10s %s   %s" % (wt, "  ".join("%-12.3f" % p[lv] for lv in LEVELS),
                                     "/h/ > k" if p["/h/"] > p["velar dorsal"] else "k > /h/"))

    jobs = ([("ALCOVE", k, wt) for wt in WEIGHTINGS for k in range(N_ALCOVE)]
            + [("RULEX", k, wt) for wt in WEIGHTINGS for k in range(N_RULEX)])
    with multiprocessing.Pool(initializer=_init, initargs=(pool,)) as mp:
        results = mp.map(run, jobs, chunksize=2)

    by = collections.defaultdict(list)
    for model, wt, k, nonce, half in results:
        by[(model, wt)].append((k, nonce, half))

    print("\n  NONCE P(-ler), mean over learners")
    print("    %-7s %-9s %s   %-8s %9s %9s %7s" % (
        "model", "training", "  ".join("%-12s" % lv for lv in LEVELS), "h vs k",
        "RMSD type", "RMSD tok", "drift"))
    rows = []
    for model in ("ALCOVE", "RULEX"):
        for wt in WEIGHTINGS:
            rs = by[(model, wt)]
            m = {lv: statistics.mean(r[1][lv] for r in rs) for lv in LEVELS}
            half = {lv: statistics.mean(r[2][lv] for r in rs) for lv in LEVELS}
            drift = max(abs(m[lv] - half[lv]) for lv in LEVELS)
            order = "/h/ > k" if m["/h/"] > m["velar dorsal"] else "k > /h/"
            print("    %-7s %-9s %s   %-8s %9.3f %9.3f %7.3f" % (
                model, wt, "  ".join("%-12.3f" % m[lv] for lv in LEVELS), order,
                rmsd(m, prof["type"]), rmsd(m, prof["token"]), drift))
            for k, nonce, _ in rs:
                for lv in LEVELS:
                    rows.append(dict(model=model, training=wt, learner=k, cue_level=lv,
                                     p_ler="%.4f" % nonce[lv]))

    print("\n  PER-LEARNER ANSWERS FOR the vowel (hiatus) items (RULEX: -lar / guess / -ler / mixed)")
    for wt in WEIGHTINGS:
        print("    RULEX %-9s %s" % (wt, S11.split([r[1]["hiatus"] for r in by[("RULEX", wt)]])))

    print("\n  RMSD type / tok = distance of the four means from the real words'")
    print("  type and token profiles. drift = change in a mean over the last %d epochs."
          % (S09.EPOCHS - S09.CHECKPOINT))
    print("  Token-trained means wobble by about .05 without trend (checked to 240")
    print("  epochs); the h-vs-k orderings held at every checkpoint.")
    C.write_tsv(os.path.join(OUT, "15_token_training.tsv"), rows)
    print("\n  wrote output/15_token_training.tsv (%d rows)" % len(rows))


if __name__ == "__main__":
    main()
