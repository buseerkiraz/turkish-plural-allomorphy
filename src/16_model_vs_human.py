#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Stage 16 - models against people: the means, and the individuals.

Reads the human responses (Stage 13) and the model predictions of Stage 15
(5,000-word vocabularies, type / token / log-token training, ALCOVE and RULEX).

PART 1, MEANS (plan item 10). For each model and training scheme, the distance
(RMSD) and Pearson r between its four cue-level P(-ler) and the human rates,
for all participants and for those without linguistics training (the more
conservative estimate of a naive speaker; Stage 13). Four points per profile,
so r is descriptive and the h-vs-k ordering is reported separately.

PART 2, INDIVIDUALS (plan item 11). Means cannot separate the models: a
population of all-or-none RULEX learners who found different rules can average
to the same rate as graded ALCOVE learners. The shape of the individual
responses can. For each cue level, what share of individuals answer all -lar,
all -ler, or a mix across that level's four items?

People give four yes/no answers per level, so each model learner is scored as
a participant would be: with P(-ler) = p on each of the four items, it is
all -lar with probability (1 - p)^4, all -ler with p^4, and mixed otherwise
(exact expectation, no sampling). The no-cue level has twelve items and is
scored the same way with 12.

RULEX is deterministic after learning, so as published it can never give a
mixed answer. Nosofsky, Palmeri & McKinley (1994) include a response-error
parameter for exactly this use (fitting individual-subject data): with
probability rerr the learner gives the opposite of its rule. RULEX is therefore
also scored with rerr set to the human -ler rate on no-cue items, the clearest
available estimate of how often people give the non-default answer where no
cue is present. Without it the comparison would be unfair to RULEX.
"""
import collections
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C
from training_data import LEVELS, WEIGHTINGS

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "output")
CUE_LEVELS = LEVELS[:3]            # the three cue levels; "no cue" is the baseline


def human_data():
    """{participant: {"ling": bool, level: [0/1 responses]}}."""
    people = {}
    for r in C.read_tsv(os.path.join(OUT, "13_human_responses.tsv")):
        p = people.setdefault(r["participant"], {"ling": r["linguistics"] == "Y"})
        p.setdefault(r["cue_level"], []).append(int(r["response_ler"]))
    return people


def human_means(people, keep):
    out = {}
    for lv in LEVELS:
        xs = [x for p in people.values() if keep(p) for x in p[lv]]
        out[lv] = sum(xs) / len(xs)
    return out


def model_learners():
    """{(model, training): [{level: p}]} from Stage 15."""
    by = collections.defaultdict(dict)
    for r in C.read_tsv(os.path.join(OUT, "15_token_training.tsv")):
        by[(r["model"], r["training"])].setdefault(r["learner"], {})[r["cue_level"]] = \
            float(r["p_ler"])
    return {k: list(v.values()) for k, v in by.items()}


def rmsd(a, b):
    return (sum((a[lv] - b[lv]) ** 2 for lv in LEVELS) / len(LEVELS)) ** 0.5


def pearson(a, b):
    xs, ys = [a[lv] for lv in LEVELS], [b[lv] for lv in LEVELS]
    mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den = (sum((x - mx) ** 2 for x in xs) * sum((y - my) ** 2 for y in ys)) ** 0.5
    return num / den if den else float("nan")


def shape_human(people, lv):
    """(all -lar, all -ler, mixed) shares of participants on one level."""
    n = len(people)
    lar = sum(1 for p in people.values() if sum(p[lv]) == 0) / n
    ler = sum(1 for p in people.values() if sum(p[lv]) == len(p[lv])) / n
    return lar, ler, 1 - lar - ler


def shape_model(learners, lv, n_items, rerr=0.0):
    """Expected (all -lar, all -ler, mixed) shares if each learner answered
    n_items items independently with its P(-ler), after response error."""
    lar = ler = 0.0
    for L in learners:
        p = rerr + (1 - 2 * rerr) * L[lv]
        lar += (1 - p) ** n_items
        ler += p ** n_items
    lar, ler = lar / len(learners), ler / len(learners)
    return lar, ler, max(0.0, 1 - lar - ler)


def main():
    people = human_data()
    models = model_learners()
    n_ling = sum(p["ling"] for p in people.values())
    print("STAGE 16 -- models against people\n")
    print("  %d participants (%d with linguistics training); models from Stage 15"
          % (len(people), n_ling))

    targets = {"all": human_means(people, lambda p: True),
               "no ling.": human_means(people, lambda p: not p["ling"])}
    print("\n=== PART 1: CUE-LEVEL MEANS (plan item 10) ===")
    print("  %-17s %s" % ("", "  ".join("%-12s" % lv for lv in LEVELS)))
    for name, t in targets.items():
        print("  %-17s %s" % ("people, " + name, "  ".join("%-12.3f" % t[lv] for lv in LEVELS)))
    print("\n  %-7s %-9s %s   %-7s %16s %16s" % (
        "model", "training", "  ".join("%-12s" % lv for lv in LEVELS), "h vs k",
        "RMSD / r (all)", "(no ling.)"))
    rows = []
    for model in ("ALCOVE", "RULEX"):
        for wt in WEIGHTINGS:
            L = models[(model, wt)]
            m = {lv: sum(x[lv] for x in L) / len(L) for lv in LEVELS}
            fits = [(rmsd(m, t), pearson(m, t)) for t in targets.values()]
            print("  %-7s %-9s %s   %-7s %9.3f / %+.2f %9.3f / %+.2f" % (
                model, wt, "  ".join("%-12.3f" % m[lv] for lv in LEVELS),
                "k > h" if m["velar dorsal"] > m["/h/"] else "h > k",
                fits[0][0], fits[0][1], fits[1][0], fits[1][1]))
            rows.append(dict(part="means", model=model, training=wt,
                             rmsd_all="%.4f" % fits[0][0], r_all="%.4f" % fits[0][1],
                             rmsd_noling="%.4f" % fits[1][0], r_noling="%.4f" % fits[1][1],
                             **{lv: "%.4f" % m[lv] for lv in LEVELS}))
    print("  people order: %s" % ("k > h" if targets["all"]["velar dorsal"] >
                                  targets["all"]["/h/"] else "h > k"))

    rerr = targets["all"]["no cue"]
    print("\n=== PART 2: INDIVIDUALS (plan item 11) ===")
    print("  share of individuals answering a level's 4 items all -lar / all -ler / mixed")
    print("  RULEX+err: RULEX with response error rerr = %.3f (people's no-cue -ler rate)" % rerr)
    print("\n  %-26s %s" % ("", "  ".join("%-20s" % lv for lv in CUE_LEVELS)))
    shapes = [("people", lambda lv: shape_human(people, lv))]
    for model in ("ALCOVE", "RULEX"):
        for wt in WEIGHTINGS:
            L = models[(model, wt)]
            shapes.append(("%s %s" % (model, wt),
                           lambda lv, L=L: shape_model(L, lv, 4)))
            if model == "RULEX":
                shapes.append(("RULEX+err %s" % wt,
                               lambda lv, L=L: shape_model(L, lv, 4, rerr)))
    for name, fn in shapes:
        cells = []
        for lv in CUE_LEVELS:
            lar, ler, mix = fn(lv)
            cells.append("%3.0f /%3.0f /%3.0f%%     " % (100 * lar, 100 * ler, 100 * mix))
            rows.append(dict(part="individuals", model=name, training="", level=lv,
                             all_lar="%.4f" % lar, all_ler="%.4f" % ler, mixed="%.4f" % mix))
        print("  %-26s %s" % (name, "".join(cells)))

    # How much response error would RULEX need to match people's mixed share?
    # An all-or-none learner with error e is mixed on 4 items with probability
    # 1 - (1-e)^4 - e^4; solve for e by bisection.
    mixed_people = sum(shape_human(people, lv)[2] for lv in CUE_LEVELS) / len(CUE_LEVELS)
    lo, hi = 0.0, 0.5
    for _ in range(60):
        e = (lo + hi) / 2
        if 1 - (1 - e) ** 4 - e ** 4 < mixed_people:
            lo = e
        else:
            hi = e
    print("\n  people are mixed on %.0f%% of cue levels on average. RULEX would need a"
          % (100 * mixed_people))
    print("  response error of %.2f to match that, against %.3f estimated from the"
          % (e, rerr))
    print("  no-cue items: %.1f times the error people actually show." % (e / rerr))

    print("""
  Reading PART 2: RULEX without response error can only produce 0% mixed, so
  any mixed human answers count against it only if they exceed what response
  error alone produces (RULEX+err rows). ALCOVE's mixed share comes from its
  graded P(-ler), before any response error.""")

    C.write_tsv(os.path.join(OUT, "16_model_vs_human.tsv"), rows,
                fieldnames=["part", "model", "training", "level", "all_lar", "all_ler",
                            "mixed", "rmsd_all", "r_all", "rmsd_noling", "r_noling"]
                + LEVELS)
    print("\n  wrote output/16_model_vs_human.tsv")


if __name__ == "__main__":
    main()
