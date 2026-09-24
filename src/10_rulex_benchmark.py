#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Stage 10 - validate the RULEX implementation on the paper's own benchmarks.

RULEX has no reference implementation to compare against (unlike ALCOVE, which
was checked trial-by-trial against catlearn in Stage 08). So it is checked
against the numbers Nosofsky, Palmeri & McKinley (1994) published.

BENCHMARK 1 - Medin & Schaffer (1978) Experiment 3, the 5-4 structure.
  Four binary dimensions, five A and four B training items, seven transfer
  items. Paper's fitted parameters (their Table 4): pstor = .55, scrit = .55,
  ccrit = 1.0, weights .43/.33/.19/.05; 16 blocks, each training item once per
  block; 5,000 simulated subjects. Paper's Table 3 gives observed P(A) and
  RULEX's predicted P(A) for all 16 items; their RMSD to the observed data is
  .048. The paper also reports that about 60% of subjects adopt a Dimension-1
  rule, about 30% a Dimension-3 rule, and about 20% learn perfectly.

BENCHMARK 2 - Shepard, Hovland & Jenkins (1961), the six types (as Stage 08).
  People: I < II < III, IV, V < VI in errors. Paper's parameters: pstor = .80,
  scrit = .75, branch = .10, capac = .40, and "uwind = 4". The paper does not
  say what unit that 4 is in. Read as 4 trials, Type II ties with Type IV; read
  as the default window (2 x 8 items = 16 trials) or as 4 blocks (64 trials),
  the ordering is reproduced. The default window is used here and the
  ambiguity is reported, not hidden.
"""
import math
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C
from rulex import Rulex

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "output")
SEED = 1

# ---------------------------------------------------------------- 5-4 data
# Paper's Table 1 / Table 3, values 1 and 2 recoded 0 and 1. Category A = 0.
MS_TRAIN = [("A1", "1112", 0), ("A2", "1212", 0), ("A3", "1211", 0),
            ("A4", "1121", 0), ("A5", "2111", 0), ("B1", "1122", 1),
            ("B2", "2112", 1), ("B3", "2221", 1), ("B4", "2222", 1)]
MS_TRANSFER = [("T1", "1221"), ("T2", "1222"), ("T3", "1111"), ("T4", "2212"),
               ("T5", "2121"), ("T6", "2211"), ("T7", "2122")]
# item: (paper's RULEX prediction, observed), both P(Category A)
MS_TABLE3 = {"A1": (.950, .970), "A2": (.974, .970), "A3": (.997, .920),
             "A4": (.867, .810), "A5": (.734, .720), "B1": (.391, .330),
             "B2": (.210, .280), "B3": (.026, .030), "B4": (.001, .050),
             "T1": (.726, .720), "T2": (.486, .560), "T3": (.991, .980),
             "T4": (.251, .230), "T5": (.299, .270), "T6": (.477, .390),
             "T7": (.045, .090)}
MS_PARAMS = dict(pstor=.55, scrit=.55, ccrit=1.0, lwind=9,
                 weights=[.43, .33, .19, .05])
MS_SUBJECTS = 5000
MS_BLOCKS = 16

# ---------------------------------------------------------------- SHJ data
SHJ_STIMULI = [tuple(int(b) for b in format(i, "03b")) for i in range(8)]
SHJ_TYPES = {                          # same table as Stage 08
    "I":   {"000", "001", "010", "011"},
    "II":  {"000", "001", "110", "111"},
    "III": {"000", "001", "010", "101"},
    "IV":  {"000", "001", "010", "100"},
    "V":   {"000", "001", "010", "111"},
    "VI":  {"000", "011", "101", "110"},
}
SHJ_PARAMS = dict(pstor=.80, scrit=.75, lwind=8, branch=.10, capac=.40)
SHJ_SUBJECTS = 2000
SHJ_BLOCKS = 16        # each block = every stimulus twice, as in the replication


def code(s):
    return tuple(int(ch) - 1 for ch in s)


def medin_schaffer(rng):
    p_a = {name: 0.0 for name in MS_TABLE3}
    rule_dims = {}
    perfect = 0
    items = [(n, code(s)) for n, s, _ in MS_TRAIN] + [(n, code(s)) for n, s in MS_TRANSFER]
    for _ in range(MS_SUBJECTS):
        m = Rulex(4, rng, **MS_PARAMS)
        for _ in range(MS_BLOCKS):
            order = MS_TRAIN[:]
            rng.shuffle(order)
            for _, s, y in order:
                m.train(code(s), y)
        for n, x in items:
            p_a[n] += m.p_category(x)[0]
        key = m.rule["dims"] if m.rule else None
        rule_dims[key] = rule_dims.get(key, 0) + 1
        perfect += all(m.p_category(code(s))[y] == 1.0 for _, s, y in MS_TRAIN)
    p_a = {n: v / MS_SUBJECTS for n, v in p_a.items()}
    return p_a, rule_dims, perfect / MS_SUBJECTS


def shj(rng):
    means = {}
    for t, members in SHJ_TYPES.items():
        err = 0.0
        for _ in range(SHJ_SUBJECTS):
            m = Rulex(3, rng, **SHJ_PARAMS)
            for _ in range(SHJ_BLOCKS):
                order = SHJ_STIMULI * 2
                rng.shuffle(order)
                for s in order:
                    y = 0 if "".join(map(str, s)) in members else 1
                    err += 1.0 - m.train(s, y)[y]
        means[t] = err / (SHJ_SUBJECTS * SHJ_BLOCKS * 16)
    return means


def main():
    rng = random.Random(SEED)
    print("STAGE 10 -- RULEX benchmarks (Nosofsky, Palmeri & McKinley 1994)\n")

    print("  BENCHMARK 1: Medin & Schaffer (1978) Exp. 3, %d simulated subjects"
          % MS_SUBJECTS)
    p_a, rule_dims, perfect = medin_schaffer(rng)
    print("    %-4s %7s %11s %9s" % ("item", "ours", "paper RULEX", "observed"))
    rows = []
    for n in MS_TABLE3:
        pr, ob = MS_TABLE3[n]
        print("    %-4s %7.3f %11.3f %9.3f" % (n, p_a[n], pr, ob))
        rows.append(dict(benchmark="medin_schaffer", item=n, ours="%.4f" % p_a[n],
                         paper=pr, observed=ob))
    rmsd_obs = math.sqrt(sum((p_a[n] - MS_TABLE3[n][1]) ** 2 for n in p_a) / len(p_a))
    rmsd_paper = math.sqrt(sum((p_a[n] - MS_TABLE3[n][0]) ** 2 for n in p_a) / len(p_a))
    share = {k: v / MS_SUBJECTS for k, v in rule_dims.items()}
    print("    RMSD to observed %.3f (paper's RULEX: .048); RMSD to paper's "
          "predictions %.3f" % (rmsd_obs, rmsd_paper))
    print("    rule on Dimension 1: %.0f%% (paper ~60%%); Dimension 3: %.0f%% "
          "(paper ~30%%); perfect learners %.0f%% (paper ~20%%)"
          % (100 * share.get((0,), 0), 100 * share.get((2,), 0), 100 * perfect))

    print("\n  BENCHMARK 2: Shepard, Hovland & Jenkins (1961), %d subjects per type"
          % SHJ_SUBJECTS)
    m = shj(rng)
    for t, v in m.items():
        print("    Type %-4s mean P(error) %.3f" % (t, v))
        rows.append(dict(benchmark="shj", item=t, ours="%.4f" % v, paper="",
                         observed=""))
    mid = [m["III"], m["IV"], m["V"]]

    checks = [
        ("5-4: RMSD to observed <= .07", rmsd_obs <= .07),
        ("5-4: RMSD to paper's predictions <= .05", rmsd_paper <= .05),
        ("5-4: Dimension-1 rule most common, then Dimension 3",
         share.get((0,), 0) > share.get((2,), 0) >
         max(share.get((1,), 0), share.get((3,), 0))),
        ("SHJ: I < II", m["I"] < m["II"]),
        ("SHJ: II < III, IV, V", m["II"] < min(mid)),
        ("SHJ: III, IV, V < VI", max(mid) < m["VI"]),
    ]
    print("\n  CHECKS")
    for name, ok in checks:
        print("    %-52s %s" % (name, "PASS" if ok else "FAIL"))

    C.write_tsv(os.path.join(OUT, "10_rulex_benchmarks.tsv"), rows)
    print("\n  wrote output/10_rulex_benchmarks.tsv")
    if not all(ok for _, ok in checks):
        print("\n  RULEX does not reproduce the published results. Do not train it")
        print("  on Turkish until this passes.")
        sys.exit(1)
    print("\n  RULEX reproduces both published benchmarks. Cleared for Turkish.")


if __name__ == "__main__":
    main()
