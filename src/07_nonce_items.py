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

Features: each item is transcribed into TELL's alphabet and coded by
common.model_features, the same function Stage 06 uses for real words. The run
fails if two cue levels share a feature vector, because a model cannot respond
differently to inputs it cannot tell apart.

Screening: every candidate is checked against the complete TELL lexeme list and
against the ~2.0m-type corpus.  Any hit is rejected, because a "nonce" word that
turns out to be real destroys the item.  So is any item within one insertion,
deletion or substitution of a corpus word seen at least NEAR_MIN times: a
participant could read it as a misspelling of that word and copy its plural.
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

# These are the four stems actually administered in the survey (Stage 13).
# They replaced an earlier set that was harmonically back-vowel throughout.
# That is the shape of a NATIVE Turkish root, whereas every real member of
# this exception class is an Arabic loan and 70% of them are disharmonic
# (seyahat, dikkat, cemaat, ziraat). Native-looking stimuli invite the default
# -lAr answer before the cue is consulted, risking a floor effect. All four
# below are disharmonic, none is a real word, none is within one edit of a
# real word, and none of their stems is a real word on its own. An earlier
# candidate, nefa-, failed the one-edit screen (nefaat ~ şefaat, nefahat ~
# sefahat, nefakat ~ refakat).
# The models are unaffected by the change: all stems code identically, since
# the feature dimensions read only the final syllable.
STEMS = ["teşa", "deşa", "şida", "yeşa"]
# Corpus frequency a one-edit neighbour needs before it counts as a near miss.
# Below this, the list is mostly typos and names no participant would know.
NEAR_MIN = 10
TR_LETTERS = "abcçdefgğhıijklmnoöprsştuüvyz"
# (spelling of the segment, its TELL symbol, label, class, real-word models).
# The lexical exception rate for each segment is read from Stage 03's cleaned
# /at/ table rather than typed in, so it cannot go stale. The middle levels are
# the ones that discriminate the models.
ENDINGS = [
    ("",  "a", "hiatus",            "strong cue", "saat, cemaat, kanaat, ziraat, itaat"),
    ("h", "h", "/h/",               "strong cue", "kabahat, seyahat, nasihat, sarahat"),
    ("k", "k", "velar dorsal",      "weak cue",   "dikkat, rikkat, sirkat, ifakat"),
    ("s", "s", "coronal fricative", "no cue",     "sanat, ticaret, kısmet, hiddet"),
    ("m", "m", "labial nasal",      "no cue",     "alamet, kıyamet, selamet"),
    ("r", "ɾ", "rhotic",            "no cue",     "ibaret, imaret, maharet, ziyaret"),
]


def at_class_counts():
    """{segment before -at: (items, exceptions)} over Stage 03's cleaned class."""
    counts = {}
    for r in C.read_tsv(os.path.join(OUT, "03_at_class.tsv")):
        n, e = counts.get(r["pre_final_v"], (0, 0))
        counts[r["pre_final_v"]] = (n + 1, e + (r["status"] == "EXCEPTION"))
    return counts


# Spelling -> TELL transcription, for the letters the nonce stems use. Every
# other letter is its own TELL symbol. Dorsals stay velar: all stems are
# back-vowel, where Turkish k/g are velar.
TO_TELL = {"ı": "ɯ", "r": "ɾ"}


def transcribe(form):
    return "".join(TO_TELL.get(ch, ch) for ch in form)


def check_distinct(items):
    """Each cue level must differ in model space from every other level."""
    by_level = {}
    for r in items:
        vec = tuple(r[n] for n in C.DIMENSION_NAMES)
        by_level.setdefault(r["cue_level"], set()).add(vec)
    clashes = [(a, b) for a in by_level for b in by_level
               if a < b and by_level[a] & by_level[b]]
    return by_level, clashes


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


def load_freq():
    counts = {}
    with open(FREQ, encoding="utf-8", errors="replace") as f:
        for line in f:
            p = line.split()
            if len(p) == 2:
                counts[p[0]] = int(p[1])
    return counts


def one_edit(form):
    """Every string one insertion, deletion or substitution away from form."""
    out = set()
    for i in range(len(form) + 1):
        a, b = form[:i], form[i:]
        if b:
            out.add(a + b[1:])
        for ch in TR_LETTERS:
            out.add(a + ch + b)
            if b:
                out.add(a + ch + b[1:])
    out.discard(form)
    return out


def near_misses(form, freq):
    """Corpus words within one edit of form, seen at least NEAR_MIN times."""
    return sorted((w for w in one_edit(form) if freq.get(w, 0) >= NEAR_MIN),
                  key=lambda w: -freq[w])


def main():
    lex = load_tell_lexemes()
    freq = load_freq()
    print("STAGE 07 -- nonce item set")
    print("  screening against %d TELL lexeme strings and %d corpus types,"
          % (len(lex), len(freq)))
    print("  and for one-edit neighbours seen at least %d times\n" % NEAR_MIN)

    counts = at_class_counts()
    items, rejected = [], []
    for stem in STEMS:
        for seg, tell_seg, seglabel, cls, models in ENDINGS:
            n, e = counts.get(tell_seg, (0, 0))
            lexrate = round(100 * e / n, 1) if n else float("nan")
            form = stem + seg + "at"
            in_tell = form in lex
            in_corpus = any(form + s in freq for s in
                            ("", "lar", "ler", "ı", "i", "ta", "te"))
            near = near_misses(form, freq)
            rec = dict(form=form, transcription=transcribe(form), stem=stem,
                       pre_ending_segment=seg or "(vowel)",
                       segment_type=seglabel, cue_class=cls,
                       cue_level=seglabel if cls != "no cue" else "no cue",
                       lexical_exception_rate_pct=lexrate,
                       cue_predicts="-ler" if cls != "no cue" else "-lar",
                       real_word_models=models,
                       in_TELL="Y" if in_tell else "N",
                       in_corpus="Y" if in_corpus else "N",
                       near_miss=", ".join(near),
                       accepted="N" if (in_tell or in_corpus or near) else "Y")
            rec.update(C.model_features(rec["transcription"]))
            (rejected if rec["accepted"] == "N" else items).append(rec)

    print("  %-12s %-18s %-12s %7s %8s %9s  %s" % ("form", "segment type", "cue class",
                                                    "lex %", "in TELL", "in corpus",
                                                    "one edit from"))
    for r in items + rejected:
        print("  %-12s %-18s %-12s %6.1f%% %8s %9s  %s%s"
              % (r["form"], r["segment_type"], r["cue_class"],
                 r["lexical_exception_rate_pct"], r["in_TELL"], r["in_corpus"],
                 r["near_miss"] or "-",
                 "   <-- REJECTED" if r["accepted"] == "N" else ""))

    print("\n  accepted: %d    rejected: %d" % (len(items), len(rejected)))
    if rejected:
        print("  Replace any rejected item with a new stem and re-run before use.")

    short = [n.split("_")[0] for n in C.DIMENSION_NAMES]
    print("\n  MODEL-SPACE CODING (common.model_features, same as Stage 06)")
    print("    %-18s %s" % ("cue level", " ".join("%-3s" % n for n in short)))
    by_level, clashes = check_distinct(items)
    for level, vecs in by_level.items():
        for v in sorted(vecs):
            print("    %-18s %s" % (level, " ".join("%-3d" % x for x in v)))
    if clashes:
        for a, b in clashes:
            print("  FAIL: cue levels '%s' and '%s' share a feature vector" % (a, b))
        print("  The models cannot distinguish these levels. Fix the dimensions in")
        print("  common.MODEL_DIMENSIONS before handing this set over.")
        sys.exit(1)
    print("    Each cue level has its own vector: the models can tell them apart.")
    print("    The four stems code identically (the dimensions only read the final")
    print("    syllable), so for the models they are replicates of one item per level.")

    print("\n" + "=" * 72)
    print("TOLERANCE PRINCIPLE BY NONCE CUE LEVEL (Stage 03 cleaned /at/ class)")
    print("=" * 72)
    print("  %-18s %-10s %5s %5s %5s %9s  %s"
          % ("cue level", "segments", "N", "-ler", "-lar", "threshold", "licenses"))
    levels = []
    for label in ("hiatus", "/h/", "velar dorsal", "no cue"):
        segs = [t for _, t, l, c, _ in ENDINGS
                if (l if c != "no cue" else "no cue") == label]
        n = sum(counts.get(t, (0, 0))[0] for t in segs)
        e = sum(counts.get(t, (0, 0))[1] for t in segs)
        th = C.tolerance_principle(n, e)["threshold"]
        levels.append((label, C.tolerance_verdict(n, e)))
        print("  %-18s %-10s %5d %5d %5d %9.1f  %s"
              % (label, ",".join(segs), n, e, n - e, th, levels[-1][1]))
    ins = [r for r in C.read_tsv(os.path.join(OUT, "03_at_class.tsv"))
           if r["in_guttural_class"] == "Y"]
    e = sum(r["status"] == "EXCEPTION" for r in ins)
    print("  %-18s %-10s %5d %5d %5d %9.1f  %s"
          % ("whole cue class", "a,h,dors", len(ins), e, len(ins) - e,
             C.tolerance_principle(len(ins), e)["threshold"],
             C.tolerance_verdict(len(ins), e)))
    print("""
  Counts are from an adult dictionary. A learner's vocabulary is smaller, and
  the verdicts for classes this size can flip with a handful of items, so
  treat the pattern as the prediction, not the exact thresholds.""")

    print("\n" + "=" * 72)
    print("PRE-REGISTERED PREDICTIONS")
    print("=" * 72)
    print("""
  The three cue sub-types have lexical exception rates of %s.
  How a model treats that ordering is the whole experiment.

  ALCOVE    A graded response tracking summed similarity to stored exceptions,
            so it SHOULD reproduce the ordering: teşaat > teşahat > teşakat >
            no-cue, with teşakat above the no-cue items.

  RULEX     Each simulated learner is categorical: it either holds a rule that
            sends a cue level to -ler or it does not, and stored exceptions are
            whole items that do not generalise to nonce words. The reported
            prediction is the AVERAGE over many learners, which is graded if
            different learners find different rules. So RULEX is not assumed
            flat: its profile has to be simulated. What it predicts that ALCOVE
            does not is the per-learner distribution: all-or-none responses per
            cue level, not intermediate ones.

  TOLERANCE The whole cue class licenses no rule, but its sub-classes differ
  PRINCIPLE (table above):
%s
            Prediction: a STEPPED profile, not a slope. Levels with a sub-rule
            come out high, levels where -lar holds come out as low as the
            no-cue items, and levels with no productive rule come out
            unstable (variable across and within speakers).

  So the discriminating item is teşakat. ALCOVE puts it above the no-cue items;
  the Tolerance Principle puts it with them. The hiatus-vs-/h/ gap separates a
  smooth slope from a step.
""" % (" / ".join("%.0f%%" % items[i]["lexical_exception_rate_pct"] for i in range(3)),
       "\n".join("              %-14s -> %s" % lv for lv in levels)))

    C.write_tsv(os.path.join(OUT, "07_nonce_items.tsv"), items + rejected)
    print("  wrote output/07_nonce_items.tsv")


if __name__ == "__main__":
    main()
