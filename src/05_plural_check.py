#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Stage 05 - does TELL's accusative-based class transfer to the PLURAL?

TELL elicited -(y)I and -(I)m, not -lAr.  The study is about -lAr.  The classes
almost always agree, but "almost always" is not a methods section.

Test: for each item, count corpus tokens of stem+lar-forms against stem+ler-forms
and compare the winner to the class TELL assigns.  The plural suffix is
consonant-initial, so it attaches to the bare citation form with no stem change,
which makes the string match safe.

Corpus: OpenSubtitles-2018 Turkish (hermitdave/FrequencyWords), ~2.0m types.
Spoken-register and noisy, but large and independent of TELL, which is what the
check needs.

Caveats built in:
  - circumflexes are stripped, since corpora rarely use them
  - items with fewer than MIN_TOKENS combined are reported as untestable rather
    than silently counted as agreements
  - multiword and hyphenated lexemes are skipped
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "output")
FREQ = os.path.join(HERE, "..", "data", "tr_full.txt")

MIN_TOKENS = 3
BACK_SUFFIXES = ["lar", "ları", "larda", "lardan", "ların"]
FRONT_SUFFIXES = ["ler", "leri", "lerde", "lerden", "lerin"]


def load_freq():
    return C.load_freq(FREQ)


orth_variants = C.orth_variants


def counts(lexeme, freq):
    back = front = 0
    for v in orth_variants(lexeme):
        back += sum(freq.get(v + s, 0) for s in BACK_SUFFIXES)
        front += sum(freq.get(v + s, 0) for s in FRONT_SUFFIXES)
    return back, front


def run(label, items, freq, minimum=MIN_TOKENS, show=12):
    agree = disagree = thin = 0
    bad = []
    detail = []
    for r in items:
        b, f = counts(r["lexeme"], freq)
        predicted = "front" if r["status"] == "EXCEPTION" else "back"
        if b + f < minimum:
            thin += 1
            observed = ""
            ok = ""
        else:
            observed = "front" if f > b else "back"
            ok = "Y" if observed == predicted else "N"
            if ok == "Y":
                agree += 1
            else:
                disagree += 1
                bad.append((r["lexeme"], r["status"], b, f, r["pre_final_v"]))
        detail.append(dict(lexeme=r["lexeme"], citation=r["citation"],
                           tell_status=r["status"], n_lar=b, n_ler=f,
                           corpus_class=observed, agrees=ok))
    tested = agree + disagree
    print("\n  %s" % label)
    print("    %d items; %d had too little corpus evidence; %d testable"
          % (len(items), thin, tested))
    if tested:
        print("    plural AGREES with TELL: %d/%d = %.1f%%"
              % (agree, tested, 100 * agree / tested))
    for lx, st, b, f, pre in sorted(bad, key=lambda x: -(x[2] + x[3]))[:show]:
        print("      MISMATCH %-12s TELL=%-9s -lar=%-5d -ler=%-5d (pre-at=%s)"
              % (lx, st, b, f, pre or "-"))
    return agree, tested, detail


def main():
    freq = load_freq()
    print("STAGE 05 -- plural spot-check")
    print("  corpus types loaded: %d" % len(freq))

    rows = C.read_tsv(os.path.join(OUT, "01_candidates.tsv"))
    pop = C.analysis_population(rows, "back")
    at = [r for r in pop if r["final_C"] == "t" and r["at_final"] == "Y"]
    lat = [r for r in pop if r["final_C"] in C.LATERALS]
    at_exc = [r for r in at if r["status"] == "EXCEPTION"]

    a1, t1, d1 = run("A. The /at/ class (primary analysis population)", at, freq)
    a2, t2, d2 = run("B. Lateral-final items (training vocabulary)", lat, freq)
    a3, t3, _ = run("C. /at/ EXCEPTIONS only (the items that matter most)",
                    at_exc, freq, minimum=1)

    print("\n" + "=" * 72)
    print("  OVERALL testable agreement: %d/%d = %.1f%%"
          % (a1 + a2, t1 + t2, 100 * (a1 + a2) / (t1 + t2)))
    print("=" * 72)
    print("\n  Report this figure in the methods section in place of the assumption")
    print("  that accusative class transfers to the plural.")
    print("\n  Inspect the /at/ mismatches above: where a mismatched item has /h/ or a")
    print("  vowel before the ending, the corpus is siding with the Stage 03 cue")
    print("  against TELL, which strengthens rather than weakens the cue. Any such")
    print("  item must be dropped from the designed minimal pairs in Stage 07.")

    C.write_tsv(os.path.join(OUT, "05_plural_check.tsv"), d1 + d2)
    print("\n  wrote output/05_plural_check.tsv")


if __name__ == "__main__":
    main()
