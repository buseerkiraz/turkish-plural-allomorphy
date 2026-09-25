#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Stage 12 - does ALCOVE's nonce result depend on its parameter values?

Stage 09 trains ALCOVE with one parameter set (Kruschke's SHJ values, validated
in Stage 08). With no human data there is nothing to fit, so the question is
whether the RESULT survives other reasonable values. Stage 11 already asks this
of RULEX.

Design: one parameter at a time, halved and doubled around the main setting,
plus attention learning switched off (lambda_a = 0), the ablation Stage 08
showed is what gives ALCOVE its SHJ ordering. Ten settings in all, including
the main one. Everything else is as Stage 09: same learner function, same
vocabularies (learner k sees the same words in every setting), same 80 epochs.
Only the 5,000-word, with-laterals condition is run: it is the one that
settled in Stage 09 and the design default for laterals.

What counts as "the result", read off each setting:
  ORDER     mean P(-ler) falls hiatus > /h/ > velar dorsal > no cue
  KUNAKAT   velar dorsal above no cue: the item that separates ALCOVE from the
            Tolerance Principle, which puts it level with no cue
  FIT       RMSD from the dictionary rates of the four levels
"""
import collections
import importlib
import multiprocessing
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C
import training_data as T
from training_data import LEVELS

S09 = importlib.import_module("09_alcove_turkish")

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "output")

VOCAB = 5000
LATERALS = "with"
LEARNERS = 10
MAIN = S09.PARAMS
SETTINGS = [("main", MAIN)]
for name in ("c", "phi", "lambda_w", "lambda_a"):
    for factor, word in ((0.5, "halved"), (2.0, "doubled")):
        SETTINGS.append(("%s %s" % (name, word), dict(MAIN, **{name: MAIN[name] * factor})))
SETTINGS.append(("attention off", dict(MAIN, lambda_a=0.0)))


def main():
    words, heard, pool = T.load()
    print("STAGE 12 -- ALCOVE parameter sensitivity\n")
    print("  %d-word vocabularies, %s lateral-final words, %d learners per setting, "
          "%d epochs" % (VOCAB, LATERALS, LEARNERS, S09.EPOCHS))
    print("  main setting: %s" % ", ".join("%s=%g" % kv for kv in MAIN.items()))

    jobs = [(VOCAB, LATERALS, k, p) for _, p in SETTINGS for k in range(LEARNERS)]
    with multiprocessing.Pool(initializer=S09._init, initargs=(pool,)) as mp:
        results = mp.map(S09.run_learner, jobs)

    lex = {lv: r / 100 for lv, r in T.lexicon_rates().items()}
    rows, summary = [], []
    print("\n  %-18s %s  %6s %6s %8s %7s %6s" % (
        "setting", "  ".join("%-7s" % lv[:7] for lv in LEVELS),
        "order", "k>none", "RMSD", "drift", "fit"))
    print("  %-18s %s" % ("dictionary", "  ".join("%-7.3f" % lex[lv] for lv in LEVELS)))
    for i, (name, params) in enumerate(SETTINGS):
        rs = results[i * LEARNERS:(i + 1) * LEARNERS]
        mean = {lv: statistics.mean(r["nonce"][lv] for r in rs) for lv in LEVELS}
        half = {lv: statistics.mean(r["halfway"][lv] for r in rs) for lv in LEVELS}
        ordered = all(mean[a] > mean[b] for a, b in zip(LEVELS, LEVELS[1:]))
        k_above = sum(r["nonce"]["velar dorsal"] > r["nonce"]["no cue"] for r in rs)
        rmsd = (sum((mean[lv] - lex[lv]) ** 2 for lv in LEVELS) / len(LEVELS)) ** 0.5
        drift = max(abs(mean[lv] - half[lv]) for lv in LEVELS)
        fit = statistics.mean(r["fit"] for r in rs)
        summary.append((name, ordered, k_above, drift))
        print("  %-18s %s  %6s %3d/%-2d %8.3f %7.3f %6.3f" % (
            name, "  ".join("%-7.3f" % mean[lv] for lv in LEVELS),
            "yes" if ordered else "NO", k_above, len(rs), rmsd, drift, fit))
        for r in rs:
            for lv in LEVELS:
                rows.append(dict(setting=name, learner=r["learner"], cue_level=lv,
                                 p_ler="%.4f" % r["nonce"][lv],
                                 **{k: "%g" % v for k, v in params.items()}))

    print("\n  order  = mean P(-ler) falls hiatus > /h/ > velar dorsal > no cue")
    print("  k>none = learners whose kunakat P(-ler) is above their no-cue P(-ler)")
    print("  RMSD   = distance of the four means from the dictionary rates")
    print("  drift  = largest change in a mean over the last %d epochs"
          % (S09.EPOCHS - S09.CHECKPOINT))

    held = [n for n, o, k, _ in summary if o and k == LEARNERS]
    unsettled = [n for n, _, _, d in summary if d > S09.CONVERGENCE_TOL]
    print("\n  VERDICT")
    print("    ordering and kunakat > no cue hold for every learner in %d of %d "
          "settings" % (len(held), len(SETTINGS)))
    for n, o, k, _ in summary:
        if n not in held:
            print("      not in: %-16s (order %s, kunakat above no cue in %d/%d)"
                  % (n, "holds" if o else "breaks", k, LEARNERS))
    if unsettled:
        print("    still moving by more than %.2f after %d epochs: %s"
              % (S09.CONVERGENCE_TOL, S09.EPOCHS, ", ".join(unsettled)))

    C.write_tsv(os.path.join(OUT, "12_alcove_sensitivity.tsv"), rows)
    print("\n  wrote output/12_alcove_sensitivity.tsv (%d rows)" % len(rows))


if __name__ == "__main__":
    main()
