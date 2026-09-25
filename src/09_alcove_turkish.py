#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Stage 09 - train ALCOVE on Turkish and test it on the nonce items.

Each simulated learner:
  1. samples a vocabulary of VOCAB words from TELL, without replacement, with
     probability proportional to corpus frequency (a learner knows the words it
     hears, not the whole dictionary; words absent from the corpus are never
     sampled);
  2. builds ALCOVE with one hidden node per distinct feature vector in that
     vocabulary (Kruschke's covering map: identical exemplars share a node);
  3. trains for EPOCHS passes over its vocabulary, each word once per pass in
     random order, on the label plural_ler (1 = -ler);
  4. is tested, without learning, on the four nonce cue levels of Stage 07.

Conditions, fully crossed:
  VOCAB     1,000 / 2,000 / 5,000 words
  LATERALS  with / without. Lateral-final words (Stage 04) cannot be learned from
            these features, since the lateral-quality cue is circular and
            excluded, so they act as label noise on voiced-final words. The
            design says to keep them as vocabulary; they are run both ways so the
            result can be shown not to depend on that choice. "Without" removes
            the whole lateral neighbourhood, regular and exceptional alike, so it
            does not distort the -ler rate of what remains.

Parameters are alcove.SHJ_HUMAN_FIT: standard ALCOVE's best fit to human
learning of the Shepard et al. (1961) types (Nosofsky et al. 1994), validated
in Stage 08. There are no human Turkish data to fit them to; Stage 12 checks
that the result survives halving and doubling each of them.

Frequency is the corpus count of the bare form plus its plural forms in BOTH
harmony series, so the weight does not favour either class. Homographs (e.g.
'sat', also 'sell!') inflate some counts; this is noise, not bias toward -ler.
"""
import collections
import multiprocessing
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C
import training_data as T
from alcove import Alcove, SHJ_HUMAN_FIT
from training_data import AT_GROUPS, LATERALS, LEVELS, VOCAB_SIZES

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "output")

LEARNERS = 20            # per condition; each has its own vocabulary and order
EPOCHS = 80              # checked to 320 for 1,000 and 2,000 words: no condition
                         # mean moved more than .031 between epochs 80 and 320.
CHECKPOINT = 60          # convergence = change over the last quarter. Small
                         # vocabularies are still learning at epoch 40, so a
                         # half-way comparison mistakes learning for drift.
PARAMS = SHJ_HUMAN_FIT   # fitted to human SHJ learning; validated in Stage 08
CONVERGENCE_TOL = 0.05   # max change in a CONDITION MEAN after CHECKPOINT.
# Single learners never settle exactly: with a fixed learning rate ALCOVE keeps
# moving by about 0.1 with trial order even at 120 epochs. That is noise that
# averages out across learners, so convergence is judged on the means.

_POOL = {}               # filled per worker by _init


def _init(pool):
    _POOL.update(pool)


def run_learner(job):
    """job = (vocab size, lateral condition, learner k[, parameters]). The
    optional parameters default to PARAMS; Stage 12 passes other settings."""
    vocab_size, laterals, k = job[:3]
    params = job[3] if len(job) > 3 else PARAMS
    rng, vocab = T.learner(_POOL, vocab_size, laterals, k)

    net = Alcove(sorted({w["vec"] for w in vocab}), 2, **params)
    halfway = None
    for ep in range(EPOCHS):
        order = vocab[:]
        rng.shuffle(order)
        for w in order:
            net.train(w["vec"], w["y"])
        if ep + 1 == CHECKPOINT:
            halfway = {lv: net.predict(v)[1] for lv, v in _POOL["nonce"].items()}

    nonce = {lv: net.predict(v)[1] for lv, v in _POOL["nonce"].items()}
    # How much of each nonce item's hidden activation comes from exemplars with
    # an IDENTICAL feature vector, as opposed to merely similar ones. Near 1
    # means ALCOVE is answering by exact lookup, not by similarity.
    exact_share = {}
    for lv, v in _POOL["nonce"].items():
        act = net._hidden(v)
        total = sum(act)
        same = sum(a for a, h in zip(act, net.h) if tuple(int(x) for x in h) == v)
        exact_share[lv] = same / total if total else float("nan")
    fit = statistics.mean(net.predict(w["vec"])[w["y"]] for w in vocab)
    at_pred = collections.defaultdict(list)
    for w in _POOL["at_all"]:
        at_pred[w["at_group"]].append(net.predict(w["vec"])[1])
    at_in_vocab = T.vocab_summary(vocab)
    return dict(vocab=vocab_size, laterals=laterals, learner=k, nonce=nonce,
                halfway=halfway, exact_share=exact_share,
                fit=fit, alpha=list(net.alpha), n_nodes=len(net.h),
                at_pred={g: statistics.mean(v) for g, v in at_pred.items()},
                at_in_vocab=dict(at_in_vocab),
                n_exc=sum(w["exc"] for w in vocab))


def sd(xs):
    return statistics.stdev(xs) if len(xs) > 1 else 0.0


def main():
    words, heard, pool = T.load()
    print("STAGE 09 -- ALCOVE trained on Turkish, tested on nonce items\n")
    print("  lexicon: %d words, %d heard in the corpus (only these are sampled)"
          % (len(words), len(heard)))
    print("  %d learners per condition, %d epochs, parameters %s"
          % (LEARNERS, EPOCHS, ", ".join("%s=%g" % kv for kv in PARAMS.items())))

    jobs = [(n, lat, k) for lat in LATERALS for n in VOCAB_SIZES
            for k in range(LEARNERS)]
    with multiprocessing.Pool(initializer=_init, initargs=(pool,)) as mp:
        results = mp.map(run_learner, jobs)

    lex_rate = T.lexicon_rates()
    by = collections.defaultdict(list)
    for r in results:
        by[(r["laterals"], r["vocab"])].append(r)

    print("\n  WHAT THE LEARNERS SAW (means per learner)")
    print("    %-8s %6s %6s %9s   %s" % ("laterals", "vocab", "nodes",
                                          "exc", "/at/ words in vocab: -ler of N by onset"))
    for (lat, n), rs in sorted(by.items()):
        cells = []
        for g, _ in AT_GROUPS:
            ler = statistics.mean(r["at_in_vocab"].get((g, 1), 0) for r in rs)
            tot = ler + statistics.mean(r["at_in_vocab"].get((g, 0), 0) for r in rs)
            cells.append("%s %.1f/%.1f" % (g, ler, tot))
        print("    %-8s %6d %6.1f %9.1f   %s"
              % (lat, n, statistics.mean(r["n_nodes"] for r in rs),
                 statistics.mean(r["n_exc"] for r in rs), "  ".join(cells)))

    print("\n  NONCE P(-ler), mean (SD) across learners")
    print("    %-8s %6s  %s   %s" % ("laterals", "vocab",
                                      "  ".join("%-14s" % lv for lv in LEVELS),
                                      "ordered*"))
    print("    %-8s %6s  %s" % ("lexicon", "", "  ".join(
        "%-14s" % ("%.1f%%" % lex_rate[lv]) for lv in LEVELS)))
    out_rows = []
    for (lat, n), rs in sorted(by.items()):
        cells = []
        for lv in LEVELS:
            xs = [r["nonce"][lv] for r in rs]
            cells.append("%-14s" % ("%.3f (%.3f)" % (statistics.mean(xs), sd(xs))))
        ordered = sum(1 for r in rs
                      if all(r["nonce"][a] > r["nonce"][b]
                             for a, b in zip(LEVELS, LEVELS[1:])))
        print("    %-8s %6d  %s   %d/%d" % (lat, n, "  ".join(cells), ordered, len(rs)))
        for r in rs:
            for lv in LEVELS:
                out_rows.append(dict(laterals=lat, vocab=n, learner=r["learner"],
                                     cue_level=lv, p_ler="%.4f" % r["nonce"][lv]))
    print("    * learners whose P(-ler) falls strictly hiatus > /h/ > dorsal > no cue")

    print("\n  REAL /at/ WORDS: mean model P(-ler) by onset (all 375, trained or not)")
    real = collections.Counter((w["at_group"], w["y"]) for w in pool["at_all"])
    print("    %-8s %6s  %s" % ("lexicon", "", "  ".join(
        "%-8s %5.1f%%" % (g, 100 * real[(g, 1)] / (real[(g, 1)] + real[(g, 0)]))
        for g, _ in AT_GROUPS)))
    for (lat, n), rs in sorted(by.items()):
        print("    %-8s %6d  %s" % (lat, n, "  ".join(
            "%-8s %5.1f%%" % (g, 100 * statistics.mean(r["at_pred"][g] for r in rs))
            for g, _ in AT_GROUPS)))

    print("\n  ATTENTION after training (mean across learners; starts at %.3f each)"
          % (1 / len(C.DIMENSION_NAMES)))
    short = [d.split("_")[0] for d in C.DIMENSION_NAMES]
    print("    %-8s %6s  %s" % ("laterals", "vocab", " ".join("%-6s" % s for s in short)))
    for (lat, n), rs in sorted(by.items()):
        print("    %-8s %6d  %s" % (lat, n, " ".join(
            "%-6.3f" % statistics.mean(r["alpha"][i] for r in rs)
            for i in range(len(short)))))

    print("\n  TRAINING FIT AND CONVERGENCE")
    worst = 0.0
    for (lat, n), rs in sorted(by.items()):
        d = max(abs(statistics.mean(r["nonce"][lv] for r in rs)
                    - statistics.mean(r["halfway"][lv] for r in rs)) for lv in LEVELS)
        worst = max(worst, d)
        print("    %-8s %6d  mean P(correct) on own vocabulary %.3f; "
              "largest change in a nonce mean over the last %d epochs %.3f"
              % (lat, n, statistics.mean(r["fit"] for r in rs),
                 EPOCHS - CHECKPOINT, d))
    if worst > CONVERGENCE_TOL:
        print("    WARNING: nonce means still moving by more than %.2f; "
              "raise EPOCHS before reading the shape." % CONVERGENCE_TOL)
    else:
        print("    Nonce means are stable; %d epochs is enough." % EPOCHS)

    C.write_tsv(os.path.join(OUT, "09_alcove_nonce.tsv"), out_rows)
    print("\n  wrote output/09_alcove_nonce.tsv (%d rows: learner x cue level)"
          % len(out_rows))


if __name__ == "__main__":
    main()
