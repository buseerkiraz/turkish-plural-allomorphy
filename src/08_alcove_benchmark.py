#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Stage 08 - validate ALCOVE against human learning data before using it.

Before ALCOVE is trained on Turkish, it has to reproduce how PEOPLE learn a
classic benchmark, or nothing it says about Turkish can be trusted.

Benchmark: the six category structures of Shepard, Hovland & Jenkins (1961),
eight stimuli on three binary dimensions split 4/4 in the six possible ways,
as replicated by Nosofsky, Gluck, Palmeri, McKinley & Glauthier (1994):
40 participants per type, 16 blocks of 16 trials (each stimulus twice per
block). Their mean error per block (their Table 1, as distributed in the R
package catlearn as `nosof94`) is reproduced below as HUMAN.

Parameters: alcove.SHJ_HUMAN_FIT, the best fit of standard ALCOVE to exactly
these data (catlearn's nosof94exalcove_opt). They are the parameters used for
Turkish in Stages 09 and 12, so the model that learns Turkish is the one that
matches human category learning here.

Checks:
  1. FIT       RMSD to the 96 human data points (6 types x 16 blocks) <= .06.
  2. ORDER     I < II < III, IV, V <= VI in mean error, as in people.
  3. ATTENTION with attention learning frozen, Type II loses its advantage over
               Type IV. Kruschke (1992) showed this is WHY ALCOVE gets the
               ordering: it shows the mechanism works, not just the numbers.

Known limitation, reported rather than hidden: with these parameters ALCOVE
learns Type VI almost as fast as Types III-V (about .125 vs .122 mean error),
whereas people find it clearly hardest (.195). The ordering holds, the size of
the Type VI gap does not.

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
from alcove import Alcove, SHJ_HUMAN_FIT

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
# Nosofsky et al. (1994) Table 1, first 16 blocks: mean P(error) per block.
HUMAN = {
    "I":   [.211, .025, .003, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "II":  [.378, .156, .083, .056, .031, .027, .028, .016, .016, .008, 0, .002,
            .005, .003, .002, 0],
    "III": [.459, .286, .223, .145, .081, .078, .063, .033, .023, .016, .019, .009,
            .008, .013, .009, .013],
    "IV":  [.422, .295, .222, .172, .148, .109, .089, .063, .025, .031, .019, .025,
            .005, 0, 0, 0],
    "V":   [.472, .331, .230, .139, .106, .081, .067, .078, .048, .045, .050, .036,
            .031, .027, .016, .014],
    "VI":  [.498, .341, .284, .245, .217, .192, .192, .177, .172, .128, .139, .117,
            .103, .098, .106, .106],
}
PARAMS = SHJ_HUMAN_FIT
BLOCKS = 16
LEARNERS = 200       # simulated learners per type, each with its own trial order
RMSD_MAX = 0.06
SEED = 7


def label(stim, members):
    return 0 if "".join(map(str, stim)) in members else 1


def simulate(members, params, rng):
    """Mean P(error) per block of 16 trials, averaged over LEARNERS."""
    curve = [0.0] * BLOCKS
    for _ in range(LEARNERS):
        net = Alcove(STIMULI, 2, **params)
        for b in range(BLOCKS):
            order = STIMULI * 2
            rng.shuffle(order)
            for s in order:
                y = label(s, members)
                curve[b] += 1.0 - net.train(s, y)[y]
    return [v / (LEARNERS * 2 * len(STIMULI)) for v in curve]


def run(params, tag):
    rng = random.Random(SEED)
    curves = {t: simulate(m, params, rng) for t, m in TYPES.items()}
    means = {t: sum(c) / len(c) for t, c in curves.items()}
    print("\n  %s" % tag)
    print("    %-5s %s   %s   %s" % ("type", " ".join("b%-4d" % (b + 1)
                                                    for b in range(0, BLOCKS, 2)),
                                     "mean", "people"))
    for t, c in curves.items():
        print("    %-5s %s   %.3f   %.3f" % (t, " ".join("%.3f" % c[b] for b in
                                                        range(0, BLOCKS, 2)),
                                           means[t], sum(HUMAN[t]) / BLOCKS))
    return curves, means


def main():
    print("STAGE 08 -- ALCOVE against human SHJ learning (Nosofsky et al. 1994)")
    print("  %d learners x %d blocks of 16 trials per type; P(error) every "
          "second block" % (LEARNERS, BLOCKS))
    print("  parameters (fitted to these data): %s"
          % ", ".join("%s=%g" % kv for kv in PARAMS.items()))

    curves, m = run(PARAMS, "WITH attention learning")
    sse = sum((curves[t][b] - HUMAN[t][b]) ** 2 for t in TYPES for b in range(BLOCKS))
    rmsd = (sse / (len(TYPES) * BLOCKS)) ** 0.5
    print("    fit to people: SSE %.3f (catlearn's fit: .142), RMSD %.3f" % (sse, rmsd))
    mid = [m["III"], m["IV"], m["V"]]
    checks = [
        ("fit: RMSD to human curves <= %.2f" % RMSD_MAX, rmsd <= RMSD_MAX),
        ("order: I < II", m["I"] < m["II"]),
        ("order: II < III, IV, V", m["II"] < min(mid)),
        ("order: III, IV, V <= VI", max(mid) <= m["VI"]),
    ]

    curves0, m0 = run(dict(PARAMS, lambda_a=0.0), "WITHOUT attention learning (lambda_a = 0)")
    checks.append(("attention off: II loses its lead over IV", m0["II"] >= m0["IV"]))

    print("\n  CHECKS")
    for name, ok in checks:
        print("    %-44s %s" % (name, "PASS" if ok else "FAIL"))
    print("\n  Limitation: Type VI mean error %.3f vs Types III-V %.3f-%.3f; people "
          "%.3f vs %.3f-%.3f." % (m["VI"], min(mid), max(mid), sum(HUMAN["VI"]) / BLOCKS,
                                 min(sum(HUMAN[t]) / BLOCKS for t in ("III", "IV", "V")),
                                 max(sum(HUMAN[t]) / BLOCKS for t in ("III", "IV", "V"))))

    rows = []
    for t in TYPES:
        for b in range(BLOCKS):
            rows.append(dict(type=t, block=b + 1, human="%.3f" % HUMAN[t][b],
                             alcove="%.4f" % curves[t][b],
                             alcove_frozen="%.4f" % curves0[t][b]))
    C.write_tsv(os.path.join(OUT, "08_alcove_shj.tsv"), rows)
    print("\n  wrote output/08_alcove_shj.tsv")

    if not all(ok for _, ok in checks):
        print("\n  ALCOVE does not reproduce the benchmark. Do not train it on")
        print("  Turkish until this passes.")
        sys.exit(1)
    print("\n  ALCOVE matches human SHJ learning, and attention learning is what")
    print("  produces the ordering. Cleared for the Turkish data.")


if __name__ == "__main__":
    main()
