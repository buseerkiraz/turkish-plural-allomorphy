#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Stage 03 - is the t-final cue an artefact of the Arabic -At suffix?

Becker, Ketrez & Nevins (2011, Language 87) warn that many t-final Turkish nouns
carry the Arabic feminine suffix -At, and that lexical trends measured over
t-final nouns may therefore be morphological facts wearing a phonological
costume.  They controlled for it by crossing TELL with a morphologically parsed
wordlist.  We have no parser, so we run the equivalent test directly: hold the
suffix constant and ask what still predicts exception status.

Three cleaning levels are reported so that the reader can see the effect sizes
do not depend on our judgement calls:

  L0  raw
  L1  duplicate transcriptions collapsed (orthographic variants of one word,
      e.g. belagat / belaagat -> one /bela:ɟat/)
  L2  transparent compounds and place names also removed, BY HAND

Automatic compound detection was tried and rejected.  Requiring the final
element to be an attested free-standing nominal wrongly flags Arabic words that
merely end in another word (cerahat contains rahat, mucazat contains azat).  A
short hand list is the honest solution and is printed below in full.
"""
import os
import sys
import collections

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "output")

# Hand-identified transparent compounds and place names among the /at/ items.
# Given as TELL transcriptions with length marks stripped.
HAND_COMPOUNDS = {
    "tʃaɫaɾsaat":      "calar + saat (alarm clock)",
    "anasaat":         "ana + saat (master clock)",
    "ampeɾsaat":       "amper + saat (ampere-hour)",
    "dʒymbyɾdʒemaat":  "cumbur + cemaat (the whole crowd)",
    "kɯtcanaat":       "kit + kanaat (frugal)",
    "aktʃaabat":       "place name, Akcaabat",
    "bojabat":         "place name, Boyabat",
    "edʒeabat":        "place name, Eceabat",
    "abɯhajat":        "abihayat, water of life (ab + hayat)",
    "heɾtʃibadabat":   "Persian phrase, herci bad abad",
    "saɾɯkanat":       "sari + kanat (a bird name)",
    "beɾmutat":        "ber + mutat (as usual)",
    "feɫdispat":       "feldspar, a Latinate compound",
    "hamhaɫat":        "ham + halat (clumsy)",
}

CUES = [
    ("hiatus (...V-at)", lambda r: r["pre_final_v"] in C.VOWELS),
    ("preceding /h/", lambda r: r["pre_final_v"] == "h"),
    ("preceding any dorsal", lambda r: r["pre_final_v"] in C.DORSALS),
    ("preceding PALATAL dorsal", lambda r: r["pre_final_v"] in C.PALATAL_DORSAL),
    ("any long vowel in word", lambda r: r["any_long_v"] == "Y"),
    ("CLASS: /a/, /h/ or dorsal",
     lambda r: r["pre_final_v"] in C.AT_CUE_CLASS),
    ("CLASS incl. any vowel", lambda r: r["pre_final_v"] in C.GUTTURAL_ONSET),
]

IN_CLASS = lambda r: r["pre_final_v"] in C.AT_CUE_CLASS


def main():
    rows = C.read_tsv(os.path.join(OUT, "01_candidates.tsv"))
    pop = C.analysis_population(rows, "back")
    tfin = [r for r in pop if r["final_C"] == "t"]
    at = [r for r in tfin if r["at_final"] == "Y"]

    print("STAGE 03 -- Arabic -At control\n")
    print("STEP 1. Is the suffix on its own the whole story?")
    e_t = sum(1 for r in tfin if r["status"] == "EXCEPTION")
    e_at = sum(1 for r in at if r["status"] == "EXCEPTION")
    print("  t-final back-vowel nominals:        %d" % len(tfin))
    print("  ... ending in the rime /at/:        %d" % len(at))
    print("  t-final exceptions:                 %d" % e_t)
    print("  ... ending in /at/:                 %d  (%.0f%% recall)"
          % (e_at, 100 * e_at / e_t))
    print("  REGULAR nouns also ending in /at/:  %d" % (len(at) - e_at))
    C.print_cue(C.cue_report(tfin, lambda r: r["at_final"] == "Y",
                             "ends in /at/ [within t-final]"))
    print("\n  -> NECESSARY but NOT SUFFICIENT. Five out of six words carrying the")
    print("     suffix are perfectly regular, so the morphology cannot be the cue.\n")

    # ---------------- cleaning levels ----------------
    L0 = at
    seen = {}
    for r in at:
        seen.setdefault(C.strip_length(r["citation"]), []).append(r)
    L1 = [v[0] for v in seen.values()]
    L2 = [r for r in L1 if C.strip_length(r["citation"]) not in HAND_COMPOUNDS]

    print("=" * 72)
    print("STEP 2. What splits the /at/ class, with the suffix held constant?\n")
    for name, S in [("L0 raw           ", L0), ("L1 deduplicated  ", L1),
                    ("L2 + compounds   ", L2)]:
        e = sum(1 for r in S if r["status"] == "EXCEPTION")
        print("  %s n=%3d  exceptions=%2d (%.1f%%)" % (name, len(S), e, 100 * e / len(S)))
    removed = {C.strip_length(r["citation"]) for r in L1} - \
              {C.strip_length(r["citation"]) for r in L2}
    unmatched = set(HAND_COMPOUNDS) - removed
    if unmatched:
        raise AssertionError(
            "HAND_COMPOUNDS entries that never matched any /at/-class row "
            "(stale entry or a transcription mismatch): %s" % sorted(unmatched))
    print("\n  hand-removed (%d):" % len(removed))
    for k in sorted(removed):
        print("    %-18s %s" % (k, HAND_COMPOUNDS[k]))
    print()

    for label, fn in CUES:
        print("  --- %s" % label)
        for name, S in [("L0", L0), ("L1", L1), ("L2", L2)]:
            rep = C.cue_report(S, fn, "")
            print("    %s  phi=%+.3f  prec=%5.1f%%  rec=%5.1f%%"
                  % (name, rep["phi"], rep["precision"], rep["recall"]))
        print()

    # ---------------- the winning class ----------------
    print("=" * 72)
    print("STEP 3. The guttural/dorsal class in detail (L2)\n")
    tot = collections.Counter(r["pre_final_v"] for r in L2)
    exc = collections.Counter(r["pre_final_v"] for r in L2
                              if r["status"] == "EXCEPTION")
    print("  segment before -at    items   exc     rate")
    for ch, n in tot.most_common():
        mark = "  <-- in class" if ch in C.AT_CUE_CLASS else ""
        print("     %-4s              %5d %5d   %5.1f%%%s"
              % (ch or "-", n, exc[ch], 100 * exc[ch] / n, mark))

    ins = [r for r in L2 if IN_CLASS(r)]
    outs = [r for r in L2 if not IN_CLASS(r)]
    ei = sum(1 for r in ins if r["status"] == "EXCEPTION")
    eo = sum(1 for r in outs if r["status"] == "EXCEPTION")
    print("\n  INSIDE  the class: %3d items, %2d exceptions (%.1f%%)"
          % (len(ins), ei, 100 * ei / len(ins)))
    print("  OUTSIDE the class: %3d items, %2d exceptions (%.1f%%)"
          % (len(outs), eo, 100 * eo / len(outs)))
    print("  exceptions outside the class: %s"
          % [r["lexeme"] for r in outs if r["status"] == "EXCEPTION"])

    tp = C.tolerance_principle(len(ins), ei)
    print("\n  Tolerance Principle inside the neighbourhood:")
    print("    N=%d, -ler=%d, -lar=%d, threshold=%.1f -> %s"
          % (tp["N"], ei, len(ins) - ei, tp["threshold"],
             C.tolerance_verdict(len(ins), ei)))
    print("    Neither -lar nor a -ler sub-rule survives over the whole class, so")
    print("    the class as a unit licenses nothing. Stage 07 applies the test to")
    print("    the sub-classes the nonce items actually probe.")

    print("\n  Exception rate inside the class is %.1f%%: neither deterministic"
          % (100 * ei / len(ins)))
    print("  (unlike the laterals at 96.6%) nor random. That partial structure is")
    print("  precisely where RULEX and ALCOVE make different predictions.")

    for r in L2:
        r["in_guttural_class"] = "Y" if IN_CLASS(r) else "N"
    C.write_tsv(os.path.join(OUT, "03_at_class.tsv"),
                sorted(L2, key=lambda r: (r["status"], r["pre_final_v"], r["citation"])),
                fieldnames=["lexeme", "citation", "status", "pre_final_v",
                            "in_guttural_class", "hiatus", "any_long_v",
                            "n_syllables", "etymology"])
    print("\n  wrote output/03_at_class.tsv (%d rows)" % len(L2))


if __name__ == "__main__":
    main()
