# -*- coding: utf-8 -*-
"""Shared phonological definitions and statistics for the Turkish plural study.

Every script in this pipeline imports its phonology from here so that a change to
a feature definition propagates everywhere instead of drifting between analyses.

TELL transcription conventions used below
-----------------------------------------
vowels      a e i ɯ u o y ø
length      ':' following a vowel
laterals    'l' = clear/palatalised, 'ɫ' = dark/velarised
dorsals     'c' = palatal k, 'ɟ' = palatal g, 'k' = velar k, 'g' = velar g
affricate   'ʒ' (NOT 'c' -- this trips people up)
"""
import math
import sys

# The transcriptions printed by these scripts contain IPA (ɫ, ɟ, ɯ, ʃ). A Windows
# console defaults to cp1252 and raises UnicodeEncodeError on the first one, which
# kills the run. Force UTF-8 on the way out. Harmless everywhere else.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

VOWELS = set("aeiɯuoyø")
BACK_V = set("aɯou")
FRONT_V = set("eiyø")
HIGH_BACK = set("ɯu")          # the accusative/possessive suffix vowel, back
HIGH_FRONT = set("iy")         # the accusative/possessive suffix vowel, front
ROUND_V = set("uoyø")

CLEAR_L = "l"
DARK_L = "ɫ"
LATERALS = {CLEAR_L, DARK_L}
PALATAL_DORSAL = set("cɟ")
VELAR_DORSAL = set("kg")
DORSALS = PALATAL_DORSAL | VELAR_DORSAL
PALATAL_C = PALATAL_DORSAL | {CLEAR_L}     # segments TELL marks as palatal
VOICED_C = set("bdgɟvzʒʒʤdʒmnɾljɫ")

# PALATAL_C is not safe to reuse as a model input feature, so this is not the
# same set even though the name is similar.
#
# PALATAL_C (just above) includes clear 'l' because clear vs dark lateral
# quality is articulatorily a palatal contrast. That is fine as an EXPLORATORY
# candidate cue in Stage 02, which runs before the circularity audit exists.
#
# Stage 04 later finds that clear/dark lateral quality is not phonetic at all:
# TELL uses it to mark harmony class, so it is unusable as model input (see
# Stage 04's verdict and the "lateral is CLEAR" cue in Stage 02). Any dimension
# computed AFTER that finding must not read lateral quality. This is that
# dorsal-only set for Stage 06's D1 dimension.
PALATAL_C_MODEL_SAFE = PALATAL_DORSAL

# AT_CUE_CLASS is the Phase 0.5 finding: within /at/-final words, exceptions occur
# essentially only when this class stands before the ending. It is a local
# generalisation over one neighbourhood and includes the dorsals. It is an
# ANALYSIS set (stages 02-04), not a model input; see MODEL_DIMENSIONS below.
AT_CUE_CLASS = {"a", "h"} | DORSALS
GUTTURAL_ONSET = AT_CUE_CLASS | VOWELS   # legacy alias: the widest variant
# Exploratory Stage 02 cue only. This used to be the single model dimension
# D2, which lumped vowel and /h/ together and left velar dorsals uncoded. That
# made the nonce items kunaat and kunahat identical in model space, and kunakat
# identical to the no-cue items, so the 71/41/21 ordering the nonce test exists
# to probe was invisible to both models. Replaced by three onset dimensions.
VOWEL_OR_H_ONSET = VOWELS | {"h"}

# The three onset classes the model sees, one binary dimension each. They are
# mutually exclusive, so each /at/ cue level gets its own feature vector.
# Separate flags rather than one lumped class: lexicon-wide the dorsals are
# common, and lumping them with vowel and /h/ is what collapsed the earlier
# D1xD2 interaction. As its own dimension a model can learn to down-weight it.
ONSET_VOWEL = VOWELS
ONSET_H = {"h"}
ONSET_DORSAL = DORSALS

JUNK_CHARS = "~#?_;@34/-"      # stray editorial marks in TELL citation fields


# ----------------------------------------------------------------- strings
def strip_length(s):
    return s.replace(":", "")


def vowels_of(s):
    return [c for c in strip_length(s) if c in VOWELS]


def last_vowel(s):
    v = vowels_of(s)
    return v[-1] if v else ""


def backness(v):
    if v in BACK_V:
        return "back"
    if v in FRONT_V:
        return "front"
    return ""


def final_consonant(citation):
    b = strip_length(citation)
    return b[-1] if b and b[-1] not in VOWELS else ""


def final_syllable(citation):
    """From the segment before the last vowel to the end of the word.

    This is deliberately crude (no real syllabifier) but it is the domain the
    cue features are defined over, and it is defined for every word.
    """
    b = strip_length(citation)
    idx = [i for i, c in enumerate(b) if c in VOWELS]
    if not idx:
        return b
    prev = idx[-2] if len(idx) > 1 else -1
    return b[prev + 1:]


def pre_final_vowel(citation):
    """The single segment immediately before the final vowel ('' if word-initial).

    For an /at/-final word this is the segment before the Arabic feminine ending,
    which is the Phase 0.5 cue site.
    """
    b = strip_length(citation)
    idx = [i for i, c in enumerate(b) if c in VOWELS]
    if not idx or idx[-1] == 0:
        return ""
    return b[idx[-1] - 1]


def has_hiatus(citation):
    b = strip_length(citation)
    return any(b[i] in VOWELS and b[i + 1] in VOWELS for i in range(len(b) - 1))


def final_vowel_long(citation):
    idx = [i for i, ch in enumerate(citation) if ch in VOWELS]
    if not idx:
        return False
    i = idx[-1]
    return i + 1 < len(citation) and citation[i + 1] == ":"


def any_long_vowel(citation):
    return ":" in citation


def is_at_final(citation):
    return strip_length(citation).endswith("at")


# ----------------------------------------------------------------- model input
# Defined over the transcription alone so that Stage 06 (real words) and Stage 07
# (nonce words) code items through the same function and cannot drift apart.
MODEL_DIMENSIONS = [
    ("D0_backness", "last vowel backness (THE RULE DIMENSION)",
     lambda c: backness(last_vowel(c)) == "back"),
    ("D1_palatal_final_syl", "palatal dorsal (c, ɟ) anywhere in the final syllable",
     lambda c: any(ch in PALATAL_C_MODEL_SAFE for ch in final_syllable(c))),
    ("D2_onset_vowel", "segment before the final vowel is a vowel (hiatus)",
     lambda c: pre_final_vowel(c) in ONSET_VOWEL),
    ("D3_onset_h", "segment before the final vowel is /h/",
     lambda c: pre_final_vowel(c) in ONSET_H),
    ("D4_onset_dorsal", "segment before the final vowel is a dorsal (k, g, c, ɟ)",
     lambda c: pre_final_vowel(c) in ONSET_DORSAL),
    ("D5_closed_syl", "word ends in a consonant",
     lambda c: bool(final_consonant(c))),
    ("D6_polysyllabic", "two or more vowels",
     lambda c: len(vowels_of(c)) >= 2),
    ("D7_round_final_v", "DISTRACTOR: final vowel is rounded (governs -(s)I, not -lAr)",
     lambda c: last_vowel(c) in ROUND_V),
    ("D8_voiced_final_c", "DISTRACTOR: final C voiced (governs kitap~kitabi, not harmony)",
     lambda c: final_consonant(c) in VOICED_C),
    # Without D9 a nonce item like kunaat shares its vector with every back
    # hiatus word, most of which are not /at/-final, so a model sees -ler rates
    # of 34/18/6/0.6% instead of the /at/-internal 71/41/21/~1% the predictions
    # are stated over. D9 lets a model form the /at/ neighbourhood. It does not
    # leak the answer: Stage 03 shows /at/ alone is 11.7% precise.
    ("D9_at_final", "word ends in the rime /at/",
     lambda c: is_at_final(c)),
]
DIMENSION_NAMES = [name for name, _, _ in MODEL_DIMENSIONS]


def model_features(citation):
    """The binary model input vector for one TELL-style transcription."""
    return {name: int(fn(citation)) for name, _, fn in MODEL_DIMENSIONS}


# ----------------------------------------------------------------- stats
def contingency(items, predicate, positive="EXCEPTION"):
    """2x2 counts (a,b,c,d) = cue+/exc, cue+/reg, cue-/exc, cue-/reg."""
    a = b = c = d = 0
    for r in items:
        hit = predicate(r)
        exc = r["status"] == positive
        if hit and exc:
            a += 1
        elif hit:
            b += 1
        elif exc:
            c += 1
        else:
            d += 1
    return a, b, c, d


def phi(a, b, c, d):
    den = (a + b) * (c + d) * (a + c) * (b + d)
    return (a * d - b * c) / math.sqrt(den) if den else float("nan")


def cue_report(items, predicate, label, positive="EXCEPTION"):
    a, b, c, d = contingency(items, predicate, positive)
    p = phi(a, b, c, d)
    prec = 100 * a / (a + b) if a + b else 0.0
    rec = 100 * a / (a + c) if a + c else 0.0
    return dict(label=label, phi=p, precision=prec, recall=rec,
                a=a, b=b, c=c, d=d, n=a + b + c + d)


def print_cue(rep, indent="  "):
    print("%s%-34s phi=%+.3f  prec=%5.1f%%  rec=%5.1f%%   (%d hits / %d flagged)"
          % (indent, rep["label"], rep["phi"], rep["precision"], rep["recall"],
             rep["a"], rep["a"] + rep["b"]))


def tolerance_principle(n_items, n_exceptions):
    """Yang (2016). A rule over N items tolerates at most N/ln(N) exceptions."""
    if n_items < 2:
        return dict(N=n_items, e=n_exceptions, threshold=float("nan"), productive=None)
    th = n_items / math.log(n_items)
    return dict(N=n_items, e=n_exceptions, threshold=th, productive=n_exceptions <= th)


def tolerance_verdict(n_items, n_ler):
    """Which plural, if any, the Tolerance Principle licenses over one class.

    Both directions have to be checked. -lar over the class has n_ler
    exceptions; a local -ler sub-rule has the regulars as ITS exceptions. A class
    too mixed for either rule gets no productive rule at all.
    """
    if n_items < 2:
        return "too few items"
    if tolerance_principle(n_items, n_ler)["productive"]:
        return "-lar (default holds)"
    if tolerance_principle(n_items, n_items - n_ler)["productive"]:
        return "-ler (local sub-rule)"
    return "no productive rule"


# ----------------------------------------------------------------- corpus
DEACCENT = str.maketrans("âîûÂÎÛ", "aiuAIU")   # corpora rarely write circumflexes


def load_freq(path):
    """OpenSubtitles frequency list as {word form: token count}."""
    f = {}
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            p = line.split()
            if len(p) == 2:
                try:
                    f[p[0]] = int(p[1])
                except ValueError:
                    pass
    return f


def orth_variants(lexeme):
    v = {lexeme, lexeme.translate(DEACCENT)}
    return {x for x in v if x and " " not in x and "-" not in x}


# ----------------------------------------------------------------- io
def read_tsv(path):
    import csv
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def write_tsv(path, rows, fieldnames=None):
    import csv
    if not rows:
        raise ValueError("refusing to write an empty table: " + path)
    fieldnames = fieldnames or list(rows[0].keys())
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t",
                           extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def analysis_population(rows, backness_filter="back"):
    """Items where an exception is logically possible and the class is determinate."""
    out = [r for r in rows if r["status"] in ("REGULAR", "EXCEPTION")]
    if backness_filter:
        out = [r for r in out if r["last_v_backness"] == backness_filter]
    return out
