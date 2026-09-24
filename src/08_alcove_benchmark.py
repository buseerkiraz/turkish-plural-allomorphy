#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Stage 08 - validate the ALCOVE implementation on a classic benchmark.

Before ALCOVE is trained on Turkish, it has to reproduce a published result, or
nothing it says about Turkish can be trusted. The benchmark is the six category
structures of Shepard, Hovland & Jenkins (1961): eight stimuli on three binary
dimensions, split 4/4 in the six possible ways. People learn them in the order

    I  <  II  <  III, IV, V  <  VI        (fewest errors first)

(Shepard et al. 1961; replicated by Nosofsky, Gluck, Palmeri, McKinley & Glauthier
1994). Kruschke (1992) showed ALCOVE reproduces this, and that it does so
BECAUSE of attention learning: with attention frozen, Type II (two relevant
dimensions) loses its advantage over Type IV (three dimensions, linearly
separable). Both halves are checked here. The second half is the stronger test,
because it shows the attention mechanism is doing the work, not just that the
numbers came out in order.

The type table was checked independently: enumerating all 70 4/4 splits of the
cube under its 48 symmetries gives exactly six classes, and the six below cover
all of them with the textbook properties (I one relevant dimension, II an XOR,
IV the only linearly separable three-dimension type, VI parity).
"""
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C
from alcove import Alcove

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "output")

STIMULI = [tuple(int(b) for b in format(i, "03b")) for i in range(8)]
# Category A members; the other four are B. Matches the published SHJ table
# with value 1 -> 0 and 2 -> 1.
TYPES = {
    "I":   {"000", "001", "010", "011"},
    "II":  {"000", "001", "110", "111"},
    "III": {"000", "001", "010", "101"},
    "IV":  {"000", "001", "010", "100"},
    "V":   {"000", "001", "010", "111"},
    "VI":  {"000", "011", "101", "110"},
}

# Parameter values Kruschke (1992) reports for his SHJ simulation (Figure 5).
# Check them against the paper before citing; the checks below test the
# qualitative ordering, which does not hinge on the exact values.
PARAMS = dict(c=6.5, phi=2.0, lambda_w=0.03, lambda_a=0.0033)
EPOCHS = 50          # one epoch = each of the 8 stimuli once, in random order
LEARNERS = 200       # simulated learners per type, each with its own trial order
BLOCK = 5            # epochs per reported block
SEED = 1


def label(stim, members):
    return 0 if "".join(map(str, stim)) in members else 1


def simulate(members, params, rng):
    """Mean P(error) per epoch, averaged over LEARNERS simulated learners."""
    curve = [0.0] * EPOCHS
    for _ in range(LEARNERS):
        net = Alcove(STIMULI, 2, **params)
        for ep in range(EPOCHS):
            order = STIMULI[:]
            rng.shuffle(order)
            for s in order:
                y = label(s, members)
                curve[ep] += 1.0 - net.train(s, y)[y]
    return [v / (LEARNERS * len(STIMULI)) for v in curve]


def run(params, tag):
    rng = random.Random(SEED)
    curves = {t: simulate(m, params, rng) for t, m in TYPES.items()}
    means = {t: sum(c) / len(c) for t, c in curves.items()}

    print("\n  %s" % tag)
    print("    %-5s %s   %s" % ("type", " ".join("ep%-3d" % (b * BLOCK + 1)
                                                for b in range(EPOCHS // BLOCK)),
                                  "mean"))
    for t, c in curves.items():
        blocks = [sum(c[b * BLOCK:(b + 1) * BLOCK]) / BLOCK
                  for b in range(EPOCHS // BLOCK)]
        print("    %-5s %s   %.3f" % (t, " ".join("%.3f" % v for v in blocks),
                                      means[t]))
    return curves, means


def main():
    print("STAGE 08 -- ALCOVE benchmark: Shepard, Hovland & Jenkins (1961)")
    print("  %d learners x %d epochs per type; P(error) per 5-epoch block"
          % (LEARNERS, EPOCHS))
    print("  parameters: %s" % ", ".join("%s=%g" % kv for kv in PARAMS.items()))

    curves, m = run(PARAMS, "WITH attention learning")
    mid = [m["III"], m["IV"], m["V"]]
    checks = [
        ("I < II", m["I"] < m["II"]),
        ("II < III, IV, V", m["II"] < min(mid)),
        ("III, IV, V < VI", max(mid) < m["VI"]),
    ]

    frozen = dict(PARAMS, lambda_a=0.0)
    curves0, m0 = run(frozen, "WITHOUT attention learning (lambda_a = 0)")
    checks.append(("attention off: II loses its lead over IV", m0["II"] >= m0["IV"]))

    print("\n  CHECKS")
    for name, ok in checks:
        print("    %-44s %s" % (name, "PASS" if ok else "FAIL"))

    rows = []
    for tag, cs in (("attention", curves), ("frozen", curves0)):
        for t, c in cs.items():
            for ep, v in enumerate(c, 1):
                rows.append(dict(condition=tag, type=t, epoch=ep,
                                 p_error="%.4f" % v))
    C.write_tsv(os.path.join(OUT, "08_alcove_shj.tsv"), rows)
    print("\n  wrote output/08_alcove_shj.tsv")

    if not all(ok for _, ok in checks):
        print("\n  ALCOVE does not reproduce the benchmark. Do not train it on")
        print("  Turkish until this passes.")
        sys.exit(1)
    print("\n  ALCOVE reproduces the SHJ ordering, and attention learning is what")
    print("  produces it. The implementation is cleared for the Turkish data.")


if __name__ == "__main__":
    main()
