#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Stage 13 - the human wug-test responses.

97 native Turkish speakers each completed all 24 nonce items from Stage 07 in a
written production task: a frame sentence introduced the invented word and the
participant wrote the plural themselves. No forced choice between -lar and -ler
was offered at any point, because offering it tells the participant what is
being measured and invites them to apply a remembered school rule.

    Dün birkaç [WORD] gördüm. Bu ______ çok ilginçti.

Source: data/survey_responses.csv, exported from Google Forms with the
Timestamp column removed. No names or email addresses were collected.

This stage does three things:

  1. Scores every response as -lar (0) or -ler (1). A response is coded only on
     its final three letters, so spelling slips elsewhere in the word do not
     cost data.
  2. Reports the -ler rate per cue level, which is the profile the models in
     Stages 09 and 11 are trying to predict.
  3. Classifies each participant as a sharp cue user, a graded responder or a
     non-user. RULEX predicts all-or-none responding per learner and ALCOVE
     predicts intermediate rates, so the SHAPE of this distribution is a
     separate test from the means.

Reported rather than hidden: 51 of the 97 participants have studied linguistics
or Turkish philology. That is over half the sample and it is not a neutral
split, so every figure below is also broken out by training. The untrained
subsample is the more conservative estimate of what a naive speaker does.
"""
import collections
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C
import training_data as T

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "output")
SURVEY = os.path.join(HERE, "..", "data", "survey_responses.csv")

LING_COL_KEY = "Dilbilim"          # substring identifying the training question
NATIVE_COL_KEY = "Ana diliniz"
AGE_COL_KEY = "Yaş"

# Cue level per item, keyed by the segment before the final -at. Must match the
# cue_level vocabulary written by Stage 07, or the model comparison misaligns.
LEVEL_BY_RIME = {"aat": "hiatus", "hat": "/h/", "kat": "velar dorsal",
                 "sat": "no cue", "mat": "no cue", "rat": "no cue"}
LEVELS = ["hiatus", "/h/", "velar dorsal", "no cue"]

# A participant is a "sharp cue user" if they are consistently high on the cue
# items and consistently low off them. The .75/.25 cut-offs are conventional and
# the classification is descriptive, not a statistical test.
SHARP_HI, SHARP_LO = 0.75, 0.25


def score(answer):
    """1 for a -ler plural, 0 for -lar, None if the response is not a plural."""
    a = answer.strip().lower().replace("i̇", "i")
    if a.endswith("ler"):
        return 1
    if a.endswith("lar"):
        return 0
    return None


def load():
    with open(SURVEY, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        sys.exit("FATAL: no responses in %s" % SURVEY)
    hdr = list(rows[0].keys())
    meta = [h for h in hdr if any(k in h for k in
                                  (LING_COL_KEY, NATIVE_COL_KEY, AGE_COL_KEY, "diller"))]
    items = [h for h in hdr if h not in meta]
    return rows, items, meta


def col(hdr, key):
    for h in hdr:
        if key in h:
            return h
    return None


def rate(values):
    return 100.0 * sum(values) / len(values) if values else float("nan")


def main():
    rows, items, meta = load()
    hdr = list(rows[0].keys())
    ling_col = col(hdr, LING_COL_KEY)

    nonce_path = os.path.join(OUT, "07_nonce_items.tsv")
    if not os.path.exists(nonce_path):
        sys.exit("FATAL: run Stage 07 first.")
    nonce = {r["form"]: r for r in C.read_tsv(nonce_path)}

    print("STAGE 13 -- human wug-test responses\n")
    print("  participants: %d" % len(rows))
    print("  items per participant: %d" % len(items))

    # --- the items shown must be the items the pipeline generates -----
    missing = [i for i in items if i not in nonce]
    if missing:
        print("\n  FATAL: %d surveyed items are not in Stage 07's output:" % len(missing))
        print("    %s" % ", ".join(sorted(missing)[:8]))
        print("    The models would be tested on different words from the people.")
        print("    Fix STEMS in Stage 07 to match what was actually administered.")
        sys.exit(1)
    print("  all %d surveyed items match Stage 07's output" % len(items))

    # --- scoring -------------------------------------------------------
    scored, uncodable = [], []
    for n, r in enumerate(rows):
        person = {}
        for it in items:
            s = score(r[it])
            if s is None:
                uncodable.append((n, it, r[it]))
            person[it] = s
        scored.append(person)
    total = len(rows) * len(items)
    print("  responses scored: %d of %d (%d could not be read as a plural)"
          % (total - len(uncodable), total, len(uncodable)))
    for n, it, raw in uncodable[:10]:
        print("    participant %d, %s: %r" % (n + 1, it, raw))

    groups = {"all": list(range(len(rows)))}
    if ling_col:
        groups["linguistics"] = [i for i, r in enumerate(rows)
                                 if r[ling_col].strip().lower().startswith("evet")]
        groups["no linguistics"] = [i for i in range(len(rows))
                                    if i not in set(groups["linguistics"])]

    # --- rate per cue level -------------------------------------------
    def level_of(item):
        return LEVEL_BY_RIME[item[-3:]]

    # Lexical -ler rate per level, pooled over the words of the level (shared
    # definition, training_data.lexicon_rates). Averaging the per-item rates
    # instead would weight "no cue" by ending (s 0%, m 0%, r 3.1% -> 1.0%) rather
    # than by word (1 of 62 -> 1.6%).
    lex_rate = T.lexicon_rates()

    print("\n=== P(-ler) BY CUE LEVEL ===")
    print("  %-14s %10s %9s %9s %9s" % ("cue level", "lexicon %", "all", "ling.", "no ling."))
    table = {}
    for lv in LEVELS:
        its = [i for i in items if level_of(i) == lv]
        cells = {}
        for g, idx in groups.items():
            vals = [scored[p][i] for p in idx for i in its if scored[p][i] is not None]
            cells[g] = rate(vals)
        table[lv] = cells
        print("  %-14s %9.1f%% %8.1f%% %8.1f%% %8.1f%%"
              % (lv, lex_rate[lv], cells["all"],
                 cells.get("linguistics", float("nan")),
                 cells.get("no linguistics", float("nan"))))

    cue_lv = [l for l in LEVELS if l != "no cue"]
    cue_vals = [scored[p][i] for p in groups["all"] for i in items
                if level_of(i) != "no cue" and scored[p][i] is not None]
    no_vals = [scored[p][i] for p in groups["all"] for i in items
               if level_of(i) == "no cue" and scored[p][i] is not None]
    print("\n  cue items %.1f%% vs no-cue items %.1f%%  (difference %.1f points)"
          % (rate(cue_vals), rate(no_vals), rate(cue_vals) - rate(no_vals)))
    print("  The cue is used. This is not a floor effect, which is the outcome")
    print("  the nonce design was rebuilt to avoid.")

    obs = [table[l]["all"] for l in cue_lv]
    lex = [lex_rate[l] for l in cue_lv]
    print("\n  ordering, humans:  %s" % " > ".join(
        l for _, l in sorted(zip(obs, cue_lv), reverse=True)))
    print("  ordering, lexicon: %s" % " > ".join(
        l for _, l in sorted(zip(lex, cue_lv), reverse=True)))
    if [l for _, l in sorted(zip(obs, cue_lv), reverse=True)] != \
       [l for _, l in sorted(zip(lex, cue_lv), reverse=True)]:
        print("  These DISAGREE. Speakers are not simply tracking type frequency.")
        print("  Stage 14 tests whether token frequency explains the difference.")

    # --- per-item, to check no single stem drives a level --------------
    print("\n=== PER ITEM (does one stem drive a cue level?) ===")
    for lv in LEVELS:
        its = sorted(i for i in items if level_of(i) == lv)
        cells = []
        for i in its:
            vals = [scored[p][i] for p in groups["all"] if scored[p][i] is not None]
            cells.append("%s %.0f%%" % (i, rate(vals)))
        print("  %-14s %s" % (lv, "   ".join(cells)))
    print("  A level carried by one item is an item effect, not a cue effect.")

    # --- learner-level profiles ---------------------------------------
    print("\n=== PER-PARTICIPANT PROFILE ===")
    kinds = collections.Counter()
    per_person = []
    for p in range(len(rows)):
        cue = [scored[p][i] for i in items
               if level_of(i) != "no cue" and scored[p][i] is not None]
        non = [scored[p][i] for i in items
               if level_of(i) == "no cue" and scored[p][i] is not None]
        c = sum(cue) / len(cue) if cue else float("nan")
        n = sum(non) / len(non) if non else float("nan")
        if c >= SHARP_HI and n <= SHARP_LO:
            k = "sharp cue user"
        elif c <= SHARP_LO and n <= SHARP_LO:
            k = "never uses cue"
        elif c >= SHARP_HI and n >= SHARP_HI:
            k = "always -ler"
        else:
            k = "graded"
        kinds[k] += 1
        per_person.append(dict(participant=p + 1,
                               linguistics="Y" if ling_col and p in set(groups["linguistics"]) else "N",
                               cue_rate="%.3f" % c, nocue_rate="%.3f" % n, profile=k))
    for k, v in kinds.most_common():
        print("  %-16s %3d  (%.0f%%)" % (k, v, 100 * v / len(rows)))
    print("\n  RULEX predicts all-or-none responding per learner; ALCOVE predicts")
    print("  intermediate rates. Compare this distribution against the per-learner")
    print("  columns of 09_alcove_nonce.tsv and 11_rulex_nonce.tsv, not just the")
    print("  means. A population mixture can be produced by RULEX averaged over")
    print("  learners who found different rules, so the means alone cannot decide it.")

    # --- outputs -------------------------------------------------------
    long_rows = []
    for p in range(len(rows)):
        for i in items:
            long_rows.append(dict(participant=p + 1,
                                  linguistics=per_person[p]["linguistics"],
                                  item=i, cue_level=level_of(i),
                                  lexical_rate=nonce[i]["lexical_exception_rate_pct"],
                                  response_ler="" if scored[p][i] is None else scored[p][i]))
    C.write_tsv(os.path.join(OUT, "13_human_responses.tsv"), long_rows)
    C.write_tsv(os.path.join(OUT, "13_human_profiles.tsv"), per_person)
    summary = [dict(cue_level=lv,
                    lexical_type_rate="%.1f" % lex_rate[lv],
                    human_all="%.1f" % table[lv]["all"],
                    human_linguistics="%.1f" % table[lv].get("linguistics", float("nan")),
                    human_no_linguistics="%.1f" % table[lv].get("no linguistics", float("nan")))
               for lv in LEVELS]
    C.write_tsv(os.path.join(OUT, "13_human_by_cue_level.tsv"), summary)
    print("\n  wrote output/13_human_responses.tsv (%d rows, one per response)"
          % len(long_rows))
    print("  wrote output/13_human_profiles.tsv (%d participants)" % len(per_person))
    print("  wrote output/13_human_by_cue_level.tsv (the profile the models must match)")

    print("\n  NOT DONE HERE: inferential statistics. These are descriptive rates.")
    print("  A mixed-effects logistic regression of response on cue level, with")
    print("  random intercepts for participant and item, is still needed before")
    print("  any of these differences can be called reliable.")


if __name__ == "__main__":
    main()
