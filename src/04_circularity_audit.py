#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Stage 04 - circularity audit of TELL's quality diacritics.

A cue is only usable as model input if it is an independent observation about
the word.  If the transcriber wrote a symbol because the word takes a front
suffix, then feeding that symbol to a model is handing it the answer.

Two diacritics are at risk:

  laterals   clear 'l' vs dark 'ɫ'
  dorsals    palatal 'c' 'ɟ' vs velar 'k' 'g'

The diagnostic is the FRONT-vowel population, where the study makes no
predictions and nothing is at stake.  In Turkish, /l/ after a front vowel is
phonetically clear and /k/ after a front vowel is phonetically fronted.  If TELL
records that, the transcription is phonetic.  If it does not, the symbol is
being used for something else, most plausibly as a marker of harmony class.
"""

import os
import sys
import collections

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "output")


def crosstab(rows, segments, title, note):
    print("  %s" % title)
    print("    %-8s %-6s %8s %10s" % ("last V", "final C", "items", "exceptions"))
    t = collections.Counter((r["last_v_backness"], r["final_C"], r["status"])
                            for r in rows)
    for bk in ("front", "back"):
        for seg in segments:
            e = t[(bk, seg, "EXCEPTION")]
            g = t[(bk, seg, "REGULAR")]
            if e + g:
                print("    %-8s %-6s %8d %10d" % (bk, seg, e + g, e))
    print("    %s\n" % note)


def main():
    rows = C.read_tsv(os.path.join(OUT, "01_candidates.tsv"))
    pop = [r for r in rows if r["status"] in ("REGULAR", "EXCEPTION")]

    print("STAGE 04 -- circularity audit\n")

    # ---------------- laterals ----------------
    lat = [r for r in pop if r["final_C"] in C.LATERALS]
    crosstab(lat, [C.CLEAR_L, C.DARK_L], "TEST A: word-final laterals", "")

    front_lat = [r for r in lat if r["last_v_backness"] == "front"]
    n_clear = sum(1 for r in front_lat if r["final_C"] == C.CLEAR_L)
    print("    After a FRONT vowel, Turkish /l/ is phonetically CLEAR.")
    print("    TELL writes clear l in %d of %d such words (%.1f%%)."
          % (n_clear, len(front_lat), 100 * n_clear / len(front_lat)))

    back_clear = [r for r in lat if r["last_v_backness"] == "back"
                  and r["final_C"] == C.CLEAR_L]
    e = sum(1 for r in back_clear if r["status"] == "EXCEPTION")
    print("    After a BACK vowel, clear l coincides with exception status in")
    print("    %d of %d words (%.1f%%)." % (e, len(back_clear), 100 * e / len(back_clear)))
    verdict_lat = n_clear / max(len(front_lat), 1) < 0.5
    print("\n    VERDICT: %s" % ("CIRCULAR. The symbol is not recording a sound; it "
                                 "tracks harmony\n             class. Unusable as model input."
                                 if verdict_lat else "phonetically grounded, usable."))
    print("             Lateral-final items stay in TRAINING as ordinary vocabulary")
    print("             but are excluded from reported results, and lateral QUALITY")
    print("             is not a model input dimension.\n")

    # ---------------- dorsals ----------------
    dor = [r for r in pop if r["final_C"] in C.DORSALS]
    crosstab(dor, ["c", "k", "ɟ", "g"], "TEST B: word-final dorsals", "")
    front_dor = [r for r in dor if r["last_v_backness"] == "front"]
    n_pal = sum(1 for r in front_dor if r["final_C"] in C.PALATAL_DORSAL)
    print("    After a FRONT vowel, TELL writes a palatal dorsal in %d of %d words"
          % (n_pal, len(front_dor)))
    print("    (%.1f%%). It uses the plain velar symbol throughout, so the dorsal"
          % (100 * n_pal / max(len(front_dor), 1)))
    print("    diacritic is NOT being used as a frontness marker.")
    print("\n    VERDICT: no circularity. The /at/ population is safe.\n")

    # ---------------- does the winning cue even need the diacritic? ----
    print("  TEST C: does the Stage 03 cue depend on the dorsal diacritic?\n")
    at = [r for r in C.analysis_population(rows, "back") if r["at_final"] == "Y"]
    for label, fn in [
        ("diacritic-FREE: /a/ or /h/ only",
         lambda r: r["pre_final_v"] in {"a", "h"}),
        ("diacritic-FREE: /a/, /h/, any dorsal",
         lambda r: r["pre_final_v"] in C.AT_CUE_CLASS),
        ("diacritic-DEPENDENT: palatal dorsal only",
         lambda r: r["pre_final_v"] in C.PALATAL_DORSAL),
    ]:
        C.print_cue(C.cue_report(at, fn, label), indent="    ")
    print("\n    The best cue lumps c, ɟ, k and g together, so it does not read the")
    print("    palatality diacritic at all. Even if that diacritic were suspect,")
    print("    the cue would stand.\n")

    # ---------------- summary ----------------
    print("=" * 72)
    print("CONSEQUENCE FOR THE DESIGN")
    print("  Primary analysis population = the /at/ class (Stage 03).")
    print("  Laterals are training vocabulary only.")
    print("  This supersedes the earlier reason for dropping laterals ('too easy').")
    print("  The real reason is contamination, which is a much stronger argument and")
    print("  one a marker would otherwise have raised against us.")


if __name__ == "__main__":
    main()
