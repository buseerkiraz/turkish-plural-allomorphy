#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Stage 01 -- classify every TELL nominal as REGULAR or EXCEPTION.

Method
------
Turkish backness harmony spreads from the final stem vowel onto the suffix.
TELL does not give the plural, but it gives three suffixed forms whose vowel is
a HIGH vowel harmonising for backness:

    accusative  -(y)I     possessive/predicative  -(I)m

Reading the backness of that suffix vowel gives the word's harmony class without
any appeal to orthography, dictionaries, or the analyst's intuition.

    stem final vowel BACK  + suffix vowel FRONT  -> EXCEPTION
    stem final vowel BACK  + suffix vowel BACK   -> REGULAR
    stem final vowel FRONT + suffix vowel FRONT  -> REGULAR
    stem final vowel FRONT + suffix vowel BACK   -> REVERSE   (should not occur)

Where the three suffixed forms disagree the item is marked VARIABLE and excluded.

Why not orthography: Turkish spelling does not reliably mark palatality or vowel
length (kagit, hala), so coding from written forms silently destroys exactly the
cues this study is looking for.
"""
import csv
import os
import sys
import collections

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C

HERE = os.path.dirname(os.path.abspath(__file__))
TELL = os.path.join(HERE, "..", "data", "tell")
OUT = os.path.join(HERE, "..", "output")


def suffix_backness(citation, suffixed):
    """Backness of the harmonising high vowel in a suffixed form, or None."""
    if not citation or not suffixed:
        return None
    stem = C.strip_length(citation)
    suf = C.strip_length(suffixed)
    # allow the final consonant to alternate (kitap ~ kitabi) by matching stem[:-1]
    if not suf.startswith(stem[:-1]):
        return None
    for ch in suf[len(stem) - 1:]:
        if ch in C.HIGH_BACK:
            return "back"
        if ch in C.HIGH_FRONT:
            return "front"
    return None


def load_etymology():
    ety = {}
    path = os.path.join(TELL, "ETYMA.db.txt")
    with open(path, encoding="utf-8", errors="replace") as f:
        for r in csv.reader(f, delimiter="\t"):
            if len(r) >= 3 and r[0] != "id":
                ety[r[0]] = r[2].strip()
    return ety


def main():
    os.makedirs(OUT, exist_ok=True)
    ety = load_etymology()

    path = os.path.join(TELL, "ELICIT.db.txt")
    with open(path, encoding="utf-8", errors="replace") as f:
        rows = list(csv.reader(f, delimiter="\t"))
    ix = {n: i for i, n in enumerate(rows[0])}

    out, skipped = [], collections.Counter()
    for r in rows[1:]:
        if len(r) < 10:
            skipped["short row"] += 1
            continue
        cit = r[ix["citation"]].strip()
        if not cit:
            skipped["no citation (speaker did not know the word)"] += 1
            continue
        if " " in cit or any(ch in cit for ch in C.JUNK_CHARS):
            skipped["editorial marks or space in citation"] += 1
            continue

        lv = C.last_vowel(cit)
        stem_back = C.backness(lv)
        if not stem_back:
            skipped["no vowel in citation"] += 1
            continue

        votes = [v for v in (suffix_backness(cit, r[ix[k]].strip())
                             for k in ("accusative", "possessive", "predicative"))
                 if v]
        if not votes:
            skipped["no usable suffixed form"] += 1
            continue
        if len(set(votes)) > 1:
            status = "VARIABLE"
        else:
            suf = votes[0]
            if stem_back == "back" and suf == "front":
                status = "EXCEPTION"
            elif stem_back == "front" and suf == "back":
                status = "REVERSE"
            else:
                status = "REGULAR"

        fc = C.final_consonant(cit)
        out.append(dict(
            id=r[ix["id"]], lexeme=r[ix["lexeme"]], citation=cit,
            accusative=r[ix["accusative"]].strip(),
            last_v=lv, last_v_backness=stem_back,
            suffix_class=votes[0] if len(set(votes)) == 1 else "mixed",
            n_suffix_votes=len(votes), status=status,
            final_C=fc,
            final_C_is_lateral="Y" if fc in C.LATERALS else "N",
            lateral_quality=("clear" if fc == C.CLEAR_L else
                             "dark" if fc == C.DARK_L else ""),
            n_syllables=len(C.vowels_of(cit)),
            final_v_long="Y" if C.final_vowel_long(cit) else "N",
            any_long_v="Y" if C.any_long_vowel(cit) else "N",
            hiatus="Y" if C.has_hiatus(cit) else "N",
            pre_final_v=C.pre_final_vowel(cit),
            final_syllable=C.final_syllable(cit),
            at_final="Y" if C.is_at_final(cit) else "N",
            etymology=ety.get(r[ix["id"]], ""),
        ))

    C.write_tsv(os.path.join(OUT, "01_candidates.tsv"), out)

    print("STAGE 01 -- extraction")
    print("  ELICIT rows read:        %d" % (len(rows) - 1))
    for k, v in skipped.most_common():
        print("    skipped, %-46s %5d" % (k + ":", v))
    print("  classified:              %d" % len(out))
    print()
    st = collections.Counter(r["status"] for r in out)
    for k in ("REGULAR", "EXCEPTION", "VARIABLE", "REVERSE"):
        print("    %-10s %6d" % (k, st[k]))
    print()

    back = C.analysis_population(out, "back")
    exc = [r for r in back if r["status"] == "EXCEPTION"]
    print("  ANALYSIS POPULATION (back final vowel): %d" % len(back))
    print("  exceptions:                             %d = %.2f%%"
          % (len(exc), 100 * len(exc) / len(back)))

    # One-directionality check. This must count over ALL front-vowel rows, not
    # over analysis_population(), which filters to REGULAR/EXCEPTION and so
    # excludes REVERSE by construction -- making the count vacuously zero.
    front_all = [r for r in out if r["last_v_backness"] == "front"]
    rev = [r for r in front_all if r["status"] == "REVERSE"]
    print("  front-vowel nominals (all statuses):    %d, of which REVERSE: %d (%.3f%%)"
          % (len(front_all), len(rev), 100 * len(rev) / len(front_all)))
    if rev:
        print("    the exception(s): %s"
              % ", ".join("%s /%s/" % (r["lexeme"], r["citation"]) for r in rev))
    print("    Harmony blocking is effectively one-directional (Kabak 2011): front")
    print("    stems essentially never take a back suffix. Report the exact count,")
    print("    not zero -- the one case above is a real datum and inspecting it is")
    print("    part of the claim.")
    print()
    print("  exceptions by final consonant:")
    for ch, n in collections.Counter(r["final_C"] for r in exc).most_common():
        print("    %-3s %4d" % (ch or "(vowel-final)", n))
    print("\n  wrote output/01_candidates.tsv")


if __name__ == "__main__":
    main()
