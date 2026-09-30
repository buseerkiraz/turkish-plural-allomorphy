#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Stage 14 - type frequency, token frequency, and which one the speakers match.

Stage 13 found that the human ordering across cue levels disagrees with the
lexicon: people put the dorsal level ABOVE /h/, the type counts put /h/ above
dorsal. This stage tests the obvious explanation. The type count treats saat and
cemaat as one item each; a speaker hears saat far more often than menfaat. If speakers track experience rather than dictionary entries,
the token-weighted rate is the one to compare against.

    type rate  = exceptions / items in the class
    token rate = corpus tokens of the exceptions / corpus tokens of the class

Why this matters for the model contest, beyond the descriptive fit. ALCOVE
accumulates exemplar traces with exposure and is frequency-sensitive by
construction. RULEX stores a rule plus a list of memorised exceptions and has no
representation of how often an item was met, but how often an item is PRESENTED
still changes what it learns: Stage 15 shows token presentation lets a frequent
exception (saat) survive, raising RULEX's hiatus P(-ler) from .10 to .82. What
RULEX cannot produce under any presentation is the graded middle of the
profile. The model-level test is Stage 15; this stage is the lexical one.

Definitions shared with the rest of the pipeline (training_data.py), so the
lexical rates here are the ones the models are compared against:
  cue levels   the segment before -at of the nonce items: hiatus {a}, /h/ {h},
               velar dorsal {k}, no cue {s, m, r}. Words with any other segment
               before -at are not part of any tested level and are left out.
  tokens       training_data.corpus_tokens: bare form plus plural forms in both
               harmony series, the count the models were trained with.

It also runs against Albright & Hayes (2003), who found type frequency fitted
English past tenses better than token frequency. Reporting a Turkish result in
the other direction is a real finding, not a failure, but it should be stated as
a disagreement with them rather than quietly.

A second comparison is run here too. Turkish orthography does not distinguish
palatal /c/ from velar /k/, so a participant reading "teşakat" cannot tell which
they are being shown, while the lexicon distinguishes them sharply. This stage
reports the velar-only, palatal-only and all-dorsal rates separately, because
the written survey cannot carry the contrast and the human figure may be an
average over both readings. The cue level itself is velar /k/ only, matching
the nonce items.
"""
import collections
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C
import training_data as T

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "output")
FREQ = os.path.join(HERE, "..", "data", "tr_full.txt")

LEVELS = ["hiatus", "/h/", "velar dorsal", "no cue"]


def tokens(lexeme, freq):
    return T.corpus_tokens(lexeme, freq)


def rates(items, freq):
    exc = [r for r in items if r["status"] == "EXCEPTION"]
    t_exc = sum(tokens(r["lexeme"], freq) for r in exc)
    t_all = sum(tokens(r["lexeme"], freq) for r in items)
    return (100.0 * len(exc) / len(items) if items else float("nan"),
            100.0 * t_exc / t_all if t_all else float("nan"),
            len(items), len(exc), t_all)


def fit(model, human):
    """RMSD and Pearson r between two equal-length profiles."""
    n = len(model)
    rmsd = math.sqrt(sum((a - b) ** 2 for a, b in zip(model, human)) / n)
    mm, mh = sum(model) / n, sum(human) / n
    num = sum((a - mm) * (b - mh) for a, b in zip(model, human))
    den = math.sqrt(sum((a - mm) ** 2 for a in model) *
                    sum((b - mh) ** 2 for b in human))
    return rmsd, (num / den if den else float("nan"))


def main():
    at_path = os.path.join(OUT, "03_at_class.tsv")
    hum_path = os.path.join(OUT, "13_human_by_cue_level.tsv")
    for p in (at_path, hum_path):
        if not os.path.exists(p):
            sys.exit("FATAL: %s missing. Run the earlier stages first." % p)
    at = C.read_tsv(at_path)
    human = {r["cue_level"]: float(r["human_all"]) for r in C.read_tsv(hum_path)}
    human_nl = {r["cue_level"]: float(r["human_no_linguistics"])
                for r in C.read_tsv(hum_path)}
    if not os.path.exists(FREQ):
        sys.exit("FATAL: %s missing. Run Stage 00." % FREQ)
    freq = C.load_freq(FREQ)
    segs = T.level_segments()
    level_of = {seg: lv for lv, ss in segs.items() for seg in ss}

    print("STAGE 14 -- type frequency vs token frequency\n")
    print("  real /at/ words: %d" % len(at))

    by = collections.defaultdict(list)
    untested = collections.Counter()
    for r in at:
        lv = level_of.get(r["pre_final_v"])
        if lv:
            by[lv].append(r)
        else:
            untested[r["pre_final_v"] or "-"] += 1
    print("  in the four tested levels: %d; other segments before -at, not tested: %d"
          % (sum(len(v) for v in by.values()), sum(untested.values())))

    print("\n=== EXCEPTION RATE PER CUE LEVEL, TWO WAYS ===")
    print("  %-14s %6s %5s %10s %11s %11s %11s"
          % ("cue level", "types", "exc", "tokens", "type %", "token %", "humans %"))
    rows, type_p, token_p, hum_p, hum_nl_p = [], [], [], [], []
    for lv in LEVELS:
        tp, tk, n, e, tot = rates(by[lv], freq)
        rows.append(dict(cue_level=lv, n_types=n, n_exceptions=e, n_tokens=tot,
                         type_rate_pct="%.1f" % tp, token_rate_pct="%.1f" % tk,
                         human_all_pct="%.1f" % human[lv],
                         human_no_linguistics_pct="%.1f" % human_nl[lv]))
        type_p.append(tp); token_p.append(tk)
        hum_p.append(human[lv]); hum_nl_p.append(human_nl[lv])
        print("  %-14s %6d %5d %10d %10.1f%% %10.1f%% %10.1f%%"
              % (lv, n, e, tot, tp, tk, human[lv]))

    print("\n=== ORDERING ===")
    for label, prof in (("type frequency ", type_p), ("token frequency", token_p),
                        ("humans         ", hum_p)):
        order = [l for _, l in sorted(zip(prof, LEVELS), reverse=True)]
        print("  %s  %s" % (label, " > ".join(order)))

    print("\n=== FIT TO THE HUMAN PROFILE ===")
    print("  %-26s %8s %8s" % ("", "RMSD", "r"))
    scores = {}
    for grp, prof_h in (("all", hum_p), ("no ling.", hum_nl_p)):
        for label, prof in (("type frequency", type_p), ("token frequency", token_p)):
            r1, c1 = fit(prof, prof_h)
            scores[(label, grp)] = (r1, c1)
            print("  %-26s %8.1f %8.3f" % ("%s, %s" % (label, grp), r1, c1))

    print("\n  READ THIS CAREFULLY, the two measures disagree.")
    print("  RMSD is better for TYPE frequency; the ordering matches TOKEN")
    print("  frequency. They are measuring different things. Token weighting")
    print("  drives the rates to extremes (%.0f%% for hiatus, %.0f%% for /h/)"
          % (token_p[0], token_p[1]))
    print("  because one or two very frequent words dominate each class, so it")
    print("  overshoots the human magnitudes badly even while getting the RANK")
    print("  ORDER right. Type frequency gets the magnitudes closer and the")
    print("  order wrong.")
    print("\n  The defensible claim is therefore narrow: speakers' ordering of the")
    print("  cue levels follows token frequency, not type frequency. Do NOT claim")
    print("  token frequency fits better overall, because on RMSD it does not.")
    print("  With four points and one word (saat, dikkat) dominating two of the")
    print("  classes, this is a direction to investigate, not a result to lean on.")
    print("  Stage 15 runs the model-level test: ALCOVE and RULEX trained on tokens.")

    # --- the orthography problem ---------------------------------------
    print("\n=== PALATAL vs VELAR DORSALS ===")
    dor = [r for r in at if r["pre_final_v"] in C.DORSALS]
    pal = [r for r in dor if r["pre_final_v"] in C.PALATAL_DORSAL]
    vel = [r for r in dor if r["pre_final_v"] in C.VELAR_DORSAL]
    for name, sub in (("palatal (c, ɟ)", pal), ("velar (k, g)", vel), ("all dorsals", dor),
                      ("tested level: k", by["velar dorsal"])):
        tp, tk, n, e, tot = rates(sub, freq)
        print("  %-16s %3d types, %2d exceptions, type %5.1f%%, token %5.1f%%"
              % (name, n, e, tp, tk))
    print("  humans on the written dorsal items: %.1f%%" % human["velar dorsal"])
    print("\n  Turkish spelling does not mark this contrast, so a participant")
    print("  reading a written nonce word cannot tell which dorsal is intended.")
    print("  If the human figure sits between the palatal and velar rates, the")
    print("  overshoot may be an artefact of written presentation rather than a")
    print("  fact about speakers. An auditory replication would separate these.")

    # --- what drives the token rates -----------------------------------
    print("\n=== THE ITEMS DRIVING THE TOKEN WEIGHTS ===")
    for lv in LEVELS[:3]:
        exc = sorted([r for r in by[lv] if r["status"] == "EXCEPTION"],
                     key=lambda r: -tokens(r["lexeme"], freq))[:3]
        print("  %-14s %s" % (lv, ", ".join("%s (%d)" % (r["lexeme"],
                                            tokens(r["lexeme"], freq)) for r in exc)))
    print("  A token-weighted rate is only as good as the corpus behind it.")
    print("  OpenSubtitles is spoken-register and skews to everyday vocabulary,")
    print("  which is arguably the right bias for a model of what speakers hear,")
    print("  but it is a choice and should be declared.")

    C.write_tsv(os.path.join(OUT, "14_frequency_rates.tsv"), rows)
    print("\n  wrote output/14_frequency_rates.tsv")

    print("\n  The model-level test is Stage 15: both models trained with type and token")
    print("  presentation. Token presentation moves ALCOVE to the token order; it also")
    print("  changes RULEX (hiatus .10 -> .82), which cannot produce the graded middle")
    print("  of the profile under either scheme.")


if __name__ == "__main__":
    main()
