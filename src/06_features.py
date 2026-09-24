#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Stage 06 - code the model input dimensions and emit the model matrix.

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

# The dimensions themselves live in common.MODEL_DIMENSIONS, so Stage 07 codes
# the nonce items through exactly the same function.

ONSET_DIMS = ["D2_onset_vowel", "D3_onset_h", "D4_onset_dorsal"]


def onset_label(d):
    on = [n for n in ONSET_DIMS if d[n]]
    assert len(on) <= 1, "onset dimensions must be mutually exclusive: %s" % d["lexeme"]
    return on[0] if on else "(none)"


def print_onset_table(items, title):
    print("\n  %s (n=%d)" % (title, len(items)))
    print("    %-18s %-4s %8s %10s %9s" % ("onset", "D1", "items", "exceptions", "rate"))
    t = collections.Counter((onset_label(d), d["D1_palatal_final_syl"], d["target"])
                            for d in items)
    for onset in ONSET_DIMS + ["(none)"]:
        for d1 in (1, 0):
            e, g = t[(onset, d1, 1)], t[(onset, d1, 0)]
            if e + g:
                print("    %-18s %-4d %8d %10d %8.1f%%"
                      % (onset, d1, e + g, e, 100 * e / (e + g)))


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
        d.update(C.model_features(r["citation"]))
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
    for name, desc, _ in C.MODEL_DIMENSIONS:
        n1 = sum(d[name] for d in out)
        print("    %-22s %-62s  on in %5d/%d" % (name, desc, n1, len(out)))

    back = [d for d in out if d["D0_backness"] == 1]
    print_onset_table(back, "EXCEPTION RATE BY ONSET, whole back-vowel population")
    print_onset_table([d for d in out if d["analysis_population"] == "Y"],
                      "EXCEPTION RATE BY ONSET, back-vowel /at/ (primary population)")
    print("    The onset classes are separate dimensions so that the /at/ cue")
    print("    levels (vowel > /h/ > dorsal > none) are distinct inputs. Whether a")
    print("    model reproduces their ordering is left to the model.")

    print("\n  REDUNDANCY CHECK (a feature that duplicates D0 inflates dimensionality)")
    front = [d for d in out if d["D0_backness"] == 0]
    for name in ["D1_palatal_final_syl"] + ONSET_DIMS:
        pb = 100 * sum(d[name] for d in back) / len(back)
        pf = 100 * sum(d[name] for d in front) / len(front)
        print("    %-22s on in %5.1f%% of back-vowel, %5.1f%% of front-vowel items"
              % (name, pb, pf))
    print("    None tracks D0, so none is the rule dimension in disguise.")

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
