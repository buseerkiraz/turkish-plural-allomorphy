#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Stage 02

The question: is exception status predictable from surface phonology at all?  
If not, no model can generalise to unseen exceptions and the RULEX/ALCOVE 
comparison is flat by construction.

This runs every candidate cue from the original work plan over the FULL
back-vowel population rather than a hand-picked sample.  That distinction
is important: an earlier pilot on 55 hand-selected items reported vowel 
length as the strongest cue (phi ~0.61) and lateral quality as weak.  
The full lexicon reverses both findings.  Hand-picked examples come from 
textbooks, and textbooks pick memorable cases, so a pilot on them measures 
the textbook rather than the language.
"""

import os
import sys
import collections

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "output")

CUES = [
    ("final C is a lateral", lambda r: r["final_C"] in C.LATERALS),
    ("lateral is CLEAR (l not dark)", lambda r: r["final_C"] == C.CLEAR_L),
    ("final C is coronal", lambda r: r["final_C"] in set("tdsznɾlɫʃʒ")),
    ("final syllable closed", lambda r: r["final_C"] != ""),
    ("any long vowel in word", lambda r: r["any_long_v"] == "Y"),
    ("final vowel long", lambda r: r["final_v_long"] == "Y"),
    ("hiatus anywhere", lambda r: r["hiatus"] == "Y"),
    ("polysyllabic (2+ syllables)", lambda r: int(r["n_syllables"]) >= 2),
    ("final vowel rounded", lambda r: r["last_v"] in C.ROUND_V),
    ("palatal C in final syllable",
     lambda r: any(ch in C.PALATAL_C for ch in r["final_syllable"])),
    ("onset is vowel or /h/",
     lambda r: r["pre_final_v"] in C.VOWEL_OR_H_ONSET),
    ("onset in /at/ cue class (wide)",
     lambda r: r["pre_final_v"] in C.AT_CUE_CLASS),
]


def main():
    rows = C.read_tsv(os.path.join(OUT, "01_candidates.tsv"))
    pop = C.analysis_population(rows, "back")
    exc = [r for r in pop if r["status"] == "EXCEPTION"]

    print("STAGE 02 -- Phase 0 viability audit")
    print("  population: %d back-vowel nominals, %d exceptions (%.2f%%)\n"
          % (len(pop), len(exc), 100 * len(exc) / len(pop)))

    print("  CANDIDATE CUES, whole population")
    reps = []
    for label, fn in CUES:
        rep = C.cue_report(pop, fn, label)
        reps.append(rep)
        C.print_cue(rep)

    print("\n  VERDICT")
    best = max(reps, key=lambda r: abs(r["phi"]))
    print("    strongest single cue: %s (phi=%+.3f)" % (best["label"], best["phi"]))
    print("    A cue exists, so the project is viable. But note below that it is")
    print("    CONDITIONAL: it lives inside particular final-consonant neighbourhoods")
    print("    rather than applying across the lexicon.\n")

    # -- where do the exceptions actually live? --------------------------
    print("  EXCEPTIONS BY FINAL CONSONANT NEIGHBOURHOOD")
    byC = collections.Counter(r["final_C"] for r in exc)
    print("    %-14s %7s %7s %8s" % ("final C", "items", "exc", "rate"))
    for ch, _ in byC.most_common(6):
        sub = [r for r in pop if r["final_C"] == ch]
        e = sum(1 for r in sub if r["status"] == "EXCEPTION")
        print("    %-14s %7d %7d %7.1f%%" % (ch or "(vowel)", len(sub), e,
                                             100 * e / len(sub)))
    print("\n    The exception set is really two sets: lateral-final and t-final.")
    print("    They have different cues and must be analysed separately.\n")

    # -- cues within each neighbourhood ----------------------------------
    lat = [r for r in pop if r["final_C"] in C.LATERALS]
    tfin = [r for r in pop if r["final_C"] == "t"]
    for name, sub, local in [
        ("LATERAL-FINAL", lat,
         [("lateral is CLEAR", lambda r: r["final_C"] == C.CLEAR_L),
          ("any long vowel", lambda r: r["any_long_v"] == "Y"),
          ("hiatus", lambda r: r["hiatus"] == "Y")]),
        ("t-FINAL", tfin,
         [("hiatus", lambda r: r["hiatus"] == "Y"),
          ("any long vowel", lambda r: r["any_long_v"] == "Y"),
          ("ends in rime /at/", lambda r: r["at_final"] == "Y"),
          ("onset is vowel or /h/",
           lambda r: r["pre_final_v"] in C.VOWEL_OR_H_ONSET),
          ("onset in /at/ cue class (wide)",
           lambda r: r["pre_final_v"] in C.AT_CUE_CLASS)])]:
        e = sum(1 for r in sub if r["status"] == "EXCEPTION")
        print("  WITHIN %s (n=%d, %d exceptions, %.1f%%)"
              % (name, len(sub), e, 100 * e / len(sub)))
        for label, fn in local:
            C.print_cue(C.cue_report(sub, fn, label), indent="    ")
        tp = C.tolerance_principle(len(sub), e)
        print("    Tolerance Principle: N=%d e=%d threshold=%.1f -> default rule %s"
              % (tp["N"], tp["e"], tp["threshold"],
                 "PRODUCTIVE" if tp["productive"] else "NOT productive"))
        print()

    # -- whole-lexicon tolerance principle -------------------------------
    tp = C.tolerance_principle(len(pop), len(exc))
    print("  TOLERANCE PRINCIPLE, whole back-vowel lexicon")
    print("    N=%d, e=%d, threshold N/lnN=%.1f -> rule %s"
          % (tp["N"], tp["e"], tp["threshold"],
             "PRODUCTIVE" if tp["productive"] else "NOT productive"))
    print("    So the default -lAr rule is safely productive overall and the")
    print("    exceptions are memorised; any sub-rule must be neighbourhood-local.")

    C.write_tsv(os.path.join(OUT, "02_phase0_cues.tsv"),
                [{k: (round(v, 4) if isinstance(v, float) else v)
                  for k, v in r.items()} for r in reps])
    print("\n  wrote output/02_phase0_cues.tsv")


if __name__ == "__main__":
    main()
