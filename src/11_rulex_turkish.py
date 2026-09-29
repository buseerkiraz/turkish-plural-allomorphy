#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Stage 11 - train RULEX on Turkish and test it on the nonce items.

Identical design to Stage 09 (ALCOVE), through training_data: the same
frequency-weighted vocabularies (learner k gets the same words in both
stages), the same with/without-laterals conditions, the same number of passes
over the vocabulary, the same nonce vectors, the same label plural_ler.

Parameters. Main setting = the values Nosofsky, Palmeri & McKinley (1994) fit
to the Shepard et al. (1961) learning data, the same benchmark ALCOVE's
parameters come from: pstor = .80, scrit = .75, capac = .40. Test windows are
the paper's defaults (lwind = number of training items, uwind = 2 x lwind).
branch is left at its default of 1.0 (imperfect single-dimension rules
before conjunctions): with ten dimensions there are 45 pairs, each tested for
up to 2 x vocabulary trials, so conjunction-first search would take most of
training just to exhaust. Stated as a deviation from the SHJ fit (branch .10).

Because capac and pstor control how many exceptions RULEX can keep, and
exceptions are the only route by which RULEX could say -ler to a nonce item,
the stage also runs pstor in {.55, .80} x capac in {.40, 1.0} and reports
whether the result depends on them.

RULEX learners are nearly deterministic after training, so besides the mean
P(-ler) the stage reports how each cue level splits into learners who say -lar
(P = 0), guess (P = .5) and say -ler (P = 1). That per-learner split is the
signature that separates RULEX from ALCOVE even where the means agree.
"""
import collections
import multiprocessing
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C
import training_data as T
from rulex import Rulex
from training_data import LATERALS, LEVELS, VOCAB_SIZES

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "output")

LEARNERS = 100           # per condition; RULEX is cheap and highly idiosyncratic
EPOCHS = 80              # same as Stage 09. Checked to 160: no condition mean
                         # for teşaat moved more than .04 between 80 and 160.
CHECKPOINT = 60          # convergence = change over the last quarter. Small
                         # vocabularies are still settling at epoch 40, so a
                         # half-way comparison overstates late drift.
# RULEX never stops moving completely: learners keep storing exceptions and
# discarding them when they fail, so individual learners flip between -lar and
# -ler on some items indefinitely. A mean over LEARNERS all-or-none learners
# therefore jumps by about sqrt(.25 / LEARNERS) between any two snapshots even at
# steady state. The tolerance is three times that noise level (.15 at 100
# learners); a real trend was ruled out by running to 160 epochs (above).
CONVERGENCE_TOL = 3 * (0.25 / LEARNERS) ** 0.5
MAIN = dict(pstor=.80, scrit=.75, capac=.40)
SENSITIVITY = [dict(pstor=p, scrit=.75, capac=c)
               for p in (.55, .80) for c in (.40, 1.0)]
SHORT = [d.split("_")[0] for d in C.DIMENSION_NAMES]

_POOL = {}


def _init(pool):
    _POOL.update(pool)


def tag(params):
    return "pstor=%.2f capac=%.2f" % (params["pstor"], params["capac"])


def run_learner(job):
    vocab_size, laterals, k, params = job
    rng, vocab = T.learner(_POOL, vocab_size, laterals, k)
    m = Rulex(len(C.DIMENSION_NAMES), rng, lwind=vocab_size, **params)
    halfway = None
    for ep in range(EPOCHS):
        order = vocab[:]
        rng.shuffle(order)
        for w in order:
            m.train(w["vec"], w["y"])
        if ep + 1 == CHECKPOINT:
            halfway = {lv: m.p_category(v)[1] for lv, v in _POOL["nonce"].items()}
    nonce = {lv: m.p_category(v)[1] for lv, v in _POOL["nonce"].items()}
    return dict(vocab=vocab_size, laterals=laterals, learner=k, params=tag(params),
                nonce=nonce, halfway=halfway,
                rule=tuple(SHORT[d] for d in m.rule["dims"]) if m.rule else None,
                n_exceptions=len(m.exceptions),
                fit=statistics.mean(m.p_category(w["vec"])[w["y"]] for w in vocab),
                at_in_vocab=T.vocab_summary(vocab))


def split(values):
    """How many learners say -lar, guess, -ler, or something in between."""
    c = collections.Counter("lar" if v == 0 else "ler" if v == 1 else
                            "guess" if v == .5 else "mixed" for v in values)
    return "%3d/%3d/%3d/%3d" % (c["lar"], c["guess"], c["ler"], c["mixed"])


def main():
    words, heard, pool = T.load()
    print("STAGE 11 -- RULEX trained on Turkish, tested on nonce items\n")
    print("  lexicon: %d words, %d heard in the corpus (only these are sampled)"
          % (len(words), len(heard)))
    print("  %d learners per condition, %d epochs; main parameters %s, scrit=.75,"
          " branch=1.0" % (LEARNERS, EPOCHS, tag(MAIN)))

    jobs = [(n, lat, k, p) for p in SENSITIVITY for lat in LATERALS
            for n in VOCAB_SIZES for k in range(LEARNERS)]
    with multiprocessing.Pool(initializer=_init, initargs=(pool,)) as mp:
        results = mp.map(run_learner, jobs, chunksize=4)

    lex_rate = T.lexicon_rates()
    by = collections.defaultdict(list)
    for r in results:
        by[(r["params"], r["laterals"], r["vocab"])].append(r)
    main_tag = tag(MAIN)

    print("\n  WHAT RULEX LEARNED (main parameters)")
    print("    %-8s %6s  %-24s %11s %9s" % ("laterals", "vocab", "rule adopted (share)",
                                             "exceptions", "fit"))
    for lat in LATERALS:
        for n in VOCAB_SIZES:
            rs = by[(main_tag, lat, n)]
            rules = collections.Counter(r["rule"] for r in rs).most_common(2)
            desc = ", ".join("%s %.0f%%" % ("+".join(k) if k else "none",
                                            100 * v / len(rs)) for k, v in rules)
            print("    %-8s %6d  %-24s %11.1f %9.3f"
                  % (lat, n, desc, statistics.mean(r["n_exceptions"] for r in rs),
                     statistics.mean(r["fit"] for r in rs)))

    print("\n  NONCE P(-ler), main parameters: mean, and learners -lar/guess/-ler/mixed")
    print("    %-8s %6s  %s" % ("laterals", "vocab", "  ".join("%-22s" % lv for lv in LEVELS)))
    print("    %-8s %6s  %s" % ("lexicon", "", "  ".join(
        "%-22s" % ("%.1f%%" % lex_rate[lv]) for lv in LEVELS)))
    out_rows = []
    for lat in LATERALS:
        for n in VOCAB_SIZES:
            rs = by[(main_tag, lat, n)]
            print("    %-8s %6d  %s" % (lat, n, "  ".join(
                "%-22s" % ("%.3f  %s" % (statistics.mean(r["nonce"][lv] for r in rs),
                                         split([r["nonce"][lv] for r in rs])))
                for lv in LEVELS)))

    print("\n  SENSITIVITY: mean nonce P(-ler) by parameter setting (5,000 words)")
    print("    %-24s %-8s  %s" % ("parameters", "laterals",
                                  "  ".join("%-12s" % lv for lv in LEVELS)))
    for p in SENSITIVITY:
        for lat in LATERALS:
            rs = by[(tag(p), lat, 5000)]
            print("    %-24s %-8s  %s   exceptions %.1f" % (tag(p), lat, "  ".join(
                "%-12.3f" % statistics.mean(r["nonce"][lv] for r in rs) for lv in LEVELS),
                statistics.mean(r["n_exceptions"] for r in rs)))

    print("\n  CONVERGENCE (largest change in a nonce mean over the last %d epochs)"
          % (EPOCHS - CHECKPOINT))
    worst = 0.0
    for key, rs in sorted(by.items()):
        d = max(abs(statistics.mean(r["nonce"][lv] for r in rs)
                    - statistics.mean(r["halfway"][lv] for r in rs)) for lv in LEVELS)
        worst = max(worst, d)
    print("    worst over all %d conditions: %.3f" % (len(by), worst))
    if worst > CONVERGENCE_TOL:
        print("    WARNING: RULEX means still moving by more than %.2f (the noise"
              " band); raise EPOCHS." % CONVERGENCE_TOL)
    else:
        print("    Within the noise band of %.2f: RULEX means are stable; %d epochs"
              " is enough." % (CONVERGENCE_TOL, EPOCHS))

    for r in results:
        for lv in LEVELS:
            out_rows.append(dict(params=r["params"], laterals=r["laterals"],
                                 vocab=r["vocab"], learner=r["learner"], cue_level=lv,
                                 p_ler="%.3f" % r["nonce"][lv],
                                 rule="+".join(r["rule"]) if r["rule"] else "",
                                 n_exceptions=r["n_exceptions"]))
    C.write_tsv(os.path.join(OUT, "11_rulex_nonce.tsv"), out_rows)
    print("\n  wrote output/11_rulex_nonce.tsv (%d rows: setting x learner x cue level)"
          % len(out_rows))


if __name__ == "__main__":
    main()
