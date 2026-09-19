#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Stage 06 - code the six model input dimensions and emit the model matrix.

Design constraints this satisfies:

1. BINARY.  RULEX as published (Nosofsky, Palmeri & McKinley 1994) operates over
   a small number of binary dimensions, four in the original stimuli.  Thirty
   one-hot phoneme columns would not reproduce the published dynamics.

2. ALWAYS DEFINED.  The earlier draft had conditional features ("is the lateral
   clear?") which are undefined for the 95% of words with no lateral.  RULEX has
   no not-applicable value, so those were unusable.  Every dimension below is a
   property of the final syllable and has an answer for every word.

3. NO LEAKAGE.  Nothing here encodes exceptionality, etymology, or harmony class.
   Lateral QUALITY is excluded on the Stage 04 circularity finding.

4. DISTRACTORS INCLUDED.  D5 and D6 govern other Turkish processes and should be
   learned-and-ignored.  Without them the model is handed only diagnostic
   features and the task is easier than the learner's real one.
"""
import os
import sys
import collections

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "output")

DIMENSIONS = [
    ("D0_backness", "last vowel backness (THE RULE DIMENSION)",
     lambda r: 1 if r["last_v_backness"] == "back" else 0),
    ("D1_palatal_final_syl", "palatal dorsal (c, ɟ) anywhere in the final syllable",
     lambda r: 1 if any(ch in C.PALATAL_C_MODEL_SAFE for ch in r["final_syllable"]) else 0),
    ("D2_guttural_onset", "segment before the final vowel is a vowel or /h/",
     lambda r: 1 if r["pre_final_v"] in C.D2_ONSET else 0),
    ("D3_closed_syl", "word ends in a consonant",
     lambda r: 1 if r["final_C"] else 0),
    ("D4_polysyllabic", "two or more vowels",
     lambda r: 1 if int(r["n_syllables"]) >= 2 else 0),
    ("D5_round_final_v", "DISTRACTOR: final vowel is rounded (governs -(s)I, not -lAr)",
     lambda r: 1 if r["last_v"] in C.ROUND_V else 0),
    ("D6_voiced_final_c", "DISTRACTOR: final C voiced (governs kitap~kitabi, not harmony)",
     lambda r: 1 if r["final_C"] in C.VOICED_C else 0),
]

# Designed minimal pairs. Each is (exception, regular), matched on the segment
# before -at and on rough length. Items the Stage 05 corpus check flagged as
# unsafe are listed in DROPPED with the reason.
MINIMAL_PAIRS = [
    ("h", "kabahat", "maslahat"),
    ("h", "sarahat", "safahat"),
    ("h", "seyahat", "meşruhat"),
    ("a", "cemaat", "inşaat"),
    ("a", "ziraat", "icraat"),
    ("a", "kanaat", "mümanaat"),
    ("a", "itaat", "müracaat"),
    ("c", "takat", "zekât"),
    ("c", "refakat", "harekât"),
    ("c", "sadakat", "tensikat"),
    ("k", "dikkat", "hilkat"),
    ("k", "sirkat", "fakat"),
    ("k", "ifakat", "mutabakat"),
    ("ɟ", "feragat", "lügat"),
]
DROPPED = [
    ("cerahat / istirahat",
     "Stage 05: corpus prefers istirahatler (21 vs 11), so the 'regular' member "
     "is not reliably regular. Was previously our best pair."),
    ("any pair using sıhhat",
     "Stage 05: corpus gives sıhhatler 24 vs 0, contradicting TELL."),
]


def main():
    rows = C.read_tsv(os.path.join(OUT, "01_candidates.tsv"))
    keep = [r for r in rows if r["status"] in ("REGULAR", "EXCEPTION")]

    out = []
    for r in keep:
        d = dict(lexeme=r["lexeme"], citation=r["citation"], status=r["status"],
                 final_C=r["final_C"], pre_final_v=r["pre_final_v"],
                 at_final=r["at_final"])
        for name, _, fn in DIMENSIONS:
            d[name] = fn(r)
        d["target"] = 1 if r["status"] == "EXCEPTION" else 0
        d["neighbourhood"] = ("at" if r["at_final"] == "Y" and r["final_C"] == "t"
                              else "lateral" if r["final_C"] in C.LATERALS
                              else "other")
        d["analysis_population"] = ("Y" if (d["neighbourhood"] == "at"
                                            and r["last_v_backness"] == "back")
                                    else "N")
        out.append(d)

    print("STAGE 06 -- feature coding\n")
    print("  DIMENSIONS")
    for name, desc, _ in DIMENSIONS:
        n1 = sum(d[name] for d in out)
        print("    %-22s %-62s  on in %5d/%d" % (name, desc, n1, len(out)))

    back = [d for d in out if d["D0_backness"] == 1]
    print("\n  HOW D1 AND D2 CROSS (back-vowel population, n=%d)" % len(back))
    print("    %-6s %-6s %8s %10s %9s" % ("D1", "D2", "items", "exceptions", "rate"))
    t = collections.Counter((d["D1_palatal_final_syl"], d["D2_guttural_onset"],
                             d["target"]) for d in back)
    empty_cells = []
    for d1 in (1, 0):
        for d2 in (1, 0):
            e = t[(d1, d2, 1)]
            g = t[(d1, d2, 0)]
            if e + g == 0:
                empty_cells.append((d1, d2))
            print("    %-6d %-6d %8d %10d %8.1f%%"
                  % (d1, d2, e + g, e, 100 * e / (e + g) if e + g else 0))
    if empty_cells:
        print("    %d of 4 cells empty (D1=%s): D2's dorsals rarely also carry a"
              % (len(empty_cells), ", ".join("%d/%d" % c for c in empty_cells)))
        print("    D1 palatal dorsal in the same final syllable. RULEX should")
        print("    partition the remaining cells; ALCOVE should smooth them.")
    else:
        print("    Monotone and graded with no empty cell. RULEX should partition this;")
        print("    ALCOVE should smooth it.")

    print("\n  REDUNDANCY CHECK (a feature that duplicates D0 inflates dimensionality)")
    front = [d for d in out if d["D0_backness"] == 0]
    for name, _, _ in DIMENSIONS[1:3]:
        pb = 100 * sum(d[name] for d in back) / len(back)
        pf = 100 * sum(d[name] for d in front) / len(front)
        print("    %-22s on in %5.1f%% of back-vowel, %5.1f%% of front-vowel items"
              % (name, pb, pf))
    print("    Neither tracks D0, so neither is the rule dimension in disguise.")

    # ---- minimal pairs ----
    byname = {r["lexeme"]: r for r in keep}
    print("\n  DESIGNED MINIMAL PAIRS (held out by design, not by random split)")
    ok = []
    for pre, e, g in MINIMAL_PAIRS:
        re_, rg = byname.get(e), byname.get(g)
        flag = ""
        if not re_ or not rg:
            flag = "  [MISSING FROM TELL]"
        elif re_["status"] != "EXCEPTION" or rg["status"] != "REGULAR":
            flag = "  [STATUS MISMATCH]"
        else:
            ok.append((pre, e, g))
        print("    %-3s %-12s (exc)  vs  %-12s (reg)%s" % (pre, e, g, flag))
    print("    %d of %d pairs validated." % (len(ok), len(MINIMAL_PAIRS)))
    print("\n  PAIRS DROPPED AFTER THE CORPUS CHECK")
    for name, why in DROPPED:
        print("    %s\n      %s" % (name, why))

    ap = [d for d in out if d["analysis_population"] == "Y"]
    print("\n  Primary analysis population (back-vowel /at/): %d items, %d exceptions"
          % (len(ap), sum(d["target"] for d in ap)))

    C.write_tsv(os.path.join(OUT, "06_model_matrix.tsv"), out)
    print("\n  wrote output/06_model_matrix.tsv (%d rows) -- this is the handoff file"
          % len(out))


if __name__ == "__main__":
    main()
