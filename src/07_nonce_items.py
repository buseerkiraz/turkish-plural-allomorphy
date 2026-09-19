#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Stage 07 - build and screen the wug set.

Why nonce items rather than held-out real words.  Albright & Hayes (2003) ran
exactly the held-out design over 4,253 English verb pairs with ten-fold cross
validation.  Both a rule-based and an analogical model returned the regular
output as first choice for essentially every held-out item, which discriminated
nothing.  Their diagnosis, after Ling & Marinov: real speakers have memorised
their exceptions and the models have not, so testing on withheld real words asks
the models to do something no speaker does.  Their fix, adopted here, is to give
models and humans the same invented words.

Design: 4 stems x 6 segments before the -at ending, fully crossed.  Within a
stem block the ONLY thing that varies is the cue segment, which is the
Albright & Hayes matched-set logic.

Screening: every candidate is checked against the complete TELL lexeme list and
against the ~2.0m-type corpus.  Any hit is rejected, because a "nonce" word that
turns out to be real destroys the item.
"""
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "output")
TELL = os.path.join(HERE, "..", "data", "tell")
FREQ = os.path.join(HERE, "..", "data", "tr_full.txt")

STEMS = ["kuna", "tıza", "bora", "nuda"]
# Lexical exception rate before -at, from stage 03, drives the three cue levels.
# The middle level is the one that discriminates the models: RULEX has no way to
# represent "somewhat", ALCOVE does.
ENDINGS = [
    ("",  "hiatus",            "strong cue", 70.6, "saat, cemaat, kanaat, ziraat, itaat"),
    ("h", "/h/",               "strong cue", 40.9, "kabahat, seyahat, nasihat, sarahat"),
    ("k", "velar dorsal",      "weak cue",   20.8, "dikkat, rikkat, sirkat, ifakat"),
    ("s", "coronal fricative", "no cue",      0.0, "sanat, ticaret, kısmet, hiddet"),
    ("m", "labial nasal",      "no cue",      0.0, "alamet, kıyamet, selamet"),
    ("r", "rhotic",            "no cue",      3.1, "ibaret, imaret, maharet, ziyaret"),
]


def load_tell_lexemes():
    lex = set()
    for table in ("MASTER.db.txt", "ELICIT.db.txt"):
        path = os.path.join(TELL, table)
        with open(path, encoding="utf-8", errors="replace") as f:
            for r in csv.reader(f, delimiter="\t"):
                for cell in r[:5]:
                    c = cell.strip().lower()
                    if c and c.isalpha():
                        lex.add(c)
    return lex


def load_freq_keys():
    keys = set()
    with open(FREQ, encoding="utf-8", errors="replace") as f:
        for line in f:
            p = line.split()
            if len(p) == 2:
                keys.add(p[0])
    return keys


def main():
    lex = load_tell_lexemes()
    freq = load_freq_keys()
    print("STAGE 07 -- nonce item set")
    print("  screening against %d TELL lexeme strings and %d corpus types\n"
          % (len(lex), len(freq)))

    items, rejected = [], []
    for stem in STEMS:
        for seg, seglabel, cls, lexrate, models in ENDINGS:
            form = stem + seg + "at"
            in_tell = form in lex
            in_corpus = any(form + s in freq for s in
                            ("", "lar", "ler", "ı", "i", "ta", "te"))
            rec = dict(form=form, stem=stem, pre_ending_segment=seg or "(vowel)",
                       segment_type=seglabel, cue_class=cls,
                       D1_palatal_final_syl=0,
                       D2_guttural_onset=1 if seg in ("", "h") else 0,
                       lexical_exception_rate_pct=lexrate,
                       cue_predicts="-ler" if cls != "no cue" else "-lar",
                       real_word_models=models,
                       in_TELL="Y" if in_tell else "N",
                       in_corpus="Y" if in_corpus else "N",
                       accepted="N" if (in_tell or in_corpus) else "Y")
            (rejected if rec["accepted"] == "N" else items).append(rec)

    print("  %-12s %-18s %-12s %7s %8s %9s" % ("form", "segment type", "cue class",
                                                "lex %", "in TELL", "in corpus"))
    for r in items + rejected:
        print("  %-12s %-18s %-12s %6.1f%% %8s %9s%s"
              % (r["form"], r["segment_type"], r["cue_class"],
                 r["lexical_exception_rate_pct"], r["in_TELL"], r["in_corpus"],
                 "   <-- REJECTED" if r["accepted"] == "N" else ""))

    print("\n  accepted: %d    rejected: %d" % (len(items), len(rejected)))
    if rejected:
        print("  Replace any rejected item with a new stem and re-run before use.")

    print("\n" + "=" * 72)
    print("PRE-REGISTERED PREDICTIONS")
    print("=" * 72)
    print("""
  The three cue sub-types have lexical exception rates of 71%, 41% and 21%.
  How a model treats that ordering is the whole experiment.

  RULEX     A sharp, near-categorical split between cue and no-cue, with the
            three cue sub-types treated ALIKE. RULEX stores a rule plus an
            exception list; it has no representation of "somewhat exceptional",
            so it should not reproduce the 71/41/21 ordering.

  ALCOVE    A graded response tracking summed similarity to stored exceptions,
            so it SHOULD reproduce the ordering: kunaat > kunahat > kunakat.
            Recovering that ordering is the clearest possible evidence for
            exemplar-based generalisation here.

  TOLERANCE Inside the cue neighbourhood the default rule is not productive
  PRINCIPLE (stage 03: N=84, e=34, threshold 19.0), so a learner should form a
            local sub-rule covering the whole neighbourhood. Prediction: a HIGH
            and UNIFORM -ler rate across all three sub-types, i.e. the same
            flat profile as RULEX but at a much higher level.

  So: flat-and-low = RULEX without the rule; flat-and-high = a sub-rule;
  ordered 71/41/21 = ALCOVE. Three distinguishable signatures from one 24-item
  set, which is the point of crossing the design.
""")

    C.write_tsv(os.path.join(OUT, "07_nonce_items.tsv"), items + rejected)
    print("  wrote output/07_nonce_items.tsv")


if __name__ == "__main__":
    main()
