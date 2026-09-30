# Rule vs. Exemplar Learning of Turkish Plural Allomorphy

Course project for "From Neurons to Transformers: Cognitive Principles in
AI Research" (Heidelberg University, SoSe 2026).

Turkish chooses the plural suffix by vowel harmony, but a set of Arabic loans
ending in *-at* break the rule (*saat → saatler*). We built the lexical
evidence for that exception class from a phonemic dictionary, implemented two
rival models of category learning, a rule-plus-exception model (RULEX) and an
exemplar model (ALCOVE), trained both on realistic Turkish vocabularies, and
tested them and 97 native speakers on the same 24 invented words.

**In one sentence:** speakers treat the invented words the way an exemplar
learner exposed to real word frequencies would. They give graded, mixed
answers that follow token frequency, and not the all-or-none answers a
rule-plus-exception learner produces.

## Team

**Niyousha Mojoudi** and **Buse Erkiraz**.

- **Buse:** the lexical pipeline (stages 00–07), the nonce stimuli, and the
  survey design and data collection. Also the human analysis (stage 13) and
  the frequency analysis (stage 14).
- **Niyousha:** the model features and training labels (stage 06
  revisions), both model implementations and their validation (stages 08
  and 10), the Turkish training and robustness runs (09, 11, 12, 15), the
  model-vs-human comparison (16), the figures and the statistics.

## Contents

- [The research question](#the-research-question)
- [Headline results](#headline-results)
- [Running it](#running-it)
- [Pipeline stages](#pipeline-stages)
- [The human survey](#the-human-survey)
- [Figures](#figures)
- [Statistics](#statistics)
- [Outputs](#outputs)
- [Data sources](#data-sources)
- [Design decisions encoded in the code](#design-decisions-encoded-in-the-code)
- [Known limitations](#known-limitations)
- [References](#references)

## The research question

Turkish selects the plural suffix `-lAr` by the backness of the stem's final
vowel: back → `-lar`, front → `-ler`. A small set of nouns, mostly Arabic and
French loans, have a back final vowel but take `-ler` anyway (*saat → saatler*).

RULEX (Nosofsky, Palmeri & McKinley 1994) says learners acquire a rule and
memorise the exceptions as a separate list. ALCOVE (Kruschke 1992) says there
is no rule and no list, only stored exemplars and similarity-based
generalisation. The two agree on known words and diverge on novel ones, so the
test is on invented words: 4 stems (*teşa-, deşa-, şida-, yeşa-*) × 6 segments
before *-at*. Grouped by what the lexicon says about that segment, these give
four cue levels:

| cue level | example | segment before *-at* | real *-at* words taking *-ler* (types) |
|---|---|---|---|
| hiatus | *teşaat* | a vowel | 70.6% |
| /h/ | *teşahat* | *h* | 40.9% |
| velar dorsal | *teşakat* | *k* | 20.8% |
| no cue | *teşasat*, *teşamat*, *teşarat* | *s*, *m*, *r* | 1.6% |

A third account, Yang's (2016) Tolerance Principle, predicts a step rather
than a slope: a *-ler* sub-rule for hiatus, no productive rule for /h/, and
*-lar* for /k/ and no cue.

## Headline results

### The lexicon (stages 01–05)

- 15,398 nominals classified; **8,139** have a back final vowel, of which **278
  are exceptions (3.42%)**. This replaces the draft's guessed 17% base rate.
- **1 of 7,253** front-vowel nominals takes a back suffix (*duhul*, where TELL's
  citation form is itself irregular), so harmony blocking is one-directional to
  within 0.014% over a 7,253-item test.
- The exception set is two neighbourhoods, not one: **197 lateral-final** and
  **44 t-final**.
- Within t-final, the Arabic `-At` suffix has **100% recall but 11.7% precision**
  (331 regular nouns carry it), so the morphology is not the cue.
- The cue is the segment immediately before `-at`: a vowel, `/h/`, or a dorsal.
  Inside that class **34/84 items are exceptions (40.5%)**; outside it **2/210
  (1.0%)**. phi = 0.545, recall 94.4%.
- **The lateral cue is circular and unusable.** TELL writes dark `ɫ` after front
  vowels in 369 of 370 cases, which is phonetically wrong, so its `l`/`ɫ`
  contrast tracks harmony class rather than sound. Lateral-final items remain as
  training vocabulary but are excluded from reported results.
- The dorsal contrast is clean (0 palatal dorsals in 649 front-vowel items), and
  the winning cue does not read the palatality diacritic at all.
- **Types and tokens order the cue levels differently.** Counted by types, the
  real *-at* words give hiatus > /h/ > /k/; counted by corpus tokens, hiatus >
  /k/ > /h/ (98% / 17% / 35%). A few very frequent words drive this: *saat*
  and *dikkat* take *-ler*, *rahat* takes *-lar* (stages 14 and 15).
- Tolerance Principle, checked in both directions: the /at/ cue class as a whole
  (84 items, 34 `-ler`) licenses **no productive rule**. Its sub-classes differ:
  hiatus licenses a local `-ler` sub-rule (12/17), velar /k/ keeps `-lar` (5/24),
  and /h/ licenses neither (9/22). So the Tolerance Principle predicts a stepped
  profile on the nonce items, with *teşakat* falling with the no-cue items.
- Plural spot-check: **269/280 testable items agree** with TELL's
  accusative-based class (96.1%); 99/102 within the `/at/` class.

### The models (stages 08–12, 15)

Both models were validated against published results before any Turkish
training. ALCOVE matches human learning of the six Shepard, Hovland & Jenkins
(1961) types (RMSD .039) and is identical trial-for-trial to catlearn's
reference implementation. RULEX reproduces Nosofsky et al.'s (1994) published
5-4 predictions (RMSD .029) and the SHJ ordering.

P(-ler) on the nonce items, 5,000-word vocabularies:

| model | training | hiatus | /h/ | /k/ | no cue |
|---|---|---|---|---|---|
| ALCOVE | type | 0.64 | 0.50 | 0.10 | 0.02 |
| ALCOVE | token | 0.98 | 0.09 | 0.32 | 0.02 |
| RULEX | type | 0.10 | 0.05 | 0.00 | 0.01 |
| RULEX | token | 0.82 | 0.01 | 0.06 | 0.03 |

- **ALCOVE follows what it is exposed to.** Trained on types, it reproduces the
  type profile; trained on tokens, the token profile, including /k/ above /h/.
- **RULEX learns the vowel rule plus about one exception and says *-lar* to
  nearly everything.** Each learner answers all-or-none (for *teşaat*: 90% of
  learners always *-lar*, 10% always *-ler*). Token training makes the hiatus
  exception survive (0.82), but RULEX never produces the graded middle levels.
- The ALCOVE pattern survives halving and doubling every parameter; only φ
  (decisiveness) moves the /k/ level noticeably (stage 12).

### The people (stage 13)

97 native speakers, 2,328 responses, all scorable.

| cue level | hiatus | /h/ | /k/ | no cue |
|---|---|---|---|---|
| people, P(-ler) | **66.0%** | **33.0%** | **48.5%** | **6.6%** |

- People use the cue (49.1% *-ler* on cue items vs 6.6% on no-cue items).
- Their order is **hiatus > /k/ > /h/**: the token order, not the type order.
- Most individuals are graded: 58 graded, 21 sharp cue users, 18 who never use
  the cue.

### Models against people (stage 16)

- **Means.** ALCOVE is closer to people than RULEX under every training
  scheme. Token-trained ALCOVE matches their pattern best (r = .88; .91 for
  participants without linguistics training) but is too extreme (hiatus .98 vs
  .66). No model gets both the order and the magnitudes.
- **Individuals.** On each cue level, 52–64% of people give **mixed** answers
  across the four items. RULEX learners cannot mix. Even with the response-error
  parameter published for RULEX, set to people's own no-cue error rate (6.6%),
  RULEX produces only 24% mixed. Matching people would take an error rate of
  19%, about three times what people actually show.

### Statistics (`stats/`)

Mixed-effects logistic regression, response (*-ler* = 1) ~ cue level +
linguistics training + (1 | participant) + (1 | item):

| test | odds ratio [95% CI] | p |
|---|---|---|
| cue level matters at all (likelihood ratio, χ²(3) = 64.9) | | 5 × 10⁻¹⁴ |
| /k/ vs /h/, **the type-vs-token reversal** | **2.27 [1.38, 3.73]** | **.001** |
| hiatus vs /k/ | 2.55 [1.54, 4.21] | .0003 |
| linguistics training × /h/ | 3.48 [1.74, 6.95] | .0004 |
| linguistics training, overall | 1.71 [1.02, 2.86] | .043 (borderline) |

The reversal holds without the one participant under 18 (OR 2.29, p = .001)
and with by-participant random slopes (OR 2.08, p = .02).

## Running it

Python 3.7 or newer. The pipeline needs no packages beyond the standard
library. Stage 00 needs network access; every later stage is offline.

macOS / Linux:

```bash
bash run_all.sh
```

Windows (Command Prompt or PowerShell):

```
run_all.bat
```

Windows users with Git Bash or WSL can use `run_all.sh` instead.

**Timing.** A full run takes about an hour once the data is cached, plus
roughly a minute the first time for the 33 MB of downloads. Almost all of it
is model training in stages 09, 11, 12 and 15, which run in parallel on all
cores. Every other stage takes seconds.

To run a single stage, or to re-run one after editing it:

```bash
python3 src/03_at_control.py
```

Stages must run in order the first time, because each reads earlier stages'
outputs (02 onwards read `output/01_candidates.tsv`, 16 reads 13 and 15).
After that any stage can be re-run on its own.

Every stage prints its full findings to the terminal, and `run_all` also saves
them to `logs/`. If you only want the numbers quoted in the write-up, read the
logs rather than the TSVs.

The figures (matplotlib) and the regression (R, `lme4`) are optional extras
with their own set-up; see [Figures](#figures) and [Statistics](#statistics).
Both are committed, so neither needs re-running to be read.

## Pipeline stages

| Stage | Script | What it does |
|---|---|---|
| 00 | `00_fetch_data.py` | Downloads TELL and the OpenSubtitles Turkish frequency list |
| 01 | `01_extract.py` | Classifies every TELL nominal REGULAR / EXCEPTION from the accusative suffix vowel |
| 02 | `02_phase0_audit.py` | Phase 0 viability audit: is exception status predictable from surface phonology? |
| 03 | `03_at_control.py` | Phase 0.5: controls for the Arabic feminine `-At` suffix |
| 04 | `04_circularity_audit.py` | Checks TELL's quality diacritics are phonetic, not class markers |
| 05 | `05_plural_check.py` | Verifies the accusative-based class transfers to the plural |
| 06 | `06_features.py` | Codes the ten model input dimensions; emits the handoff matrix |
| 07 | `07_nonce_items.py` | Builds and screens the 24-item wug set |
| 08 | `08_alcove_benchmark.py` | Validates ALCOVE (`alcove.py`) against human learning of the Shepard, Hovland & Jenkins (1961) types (Nosofsky et al. 1994) |
| 09 | `09_alcove_turkish.py` | Trains ALCOVE on frequency-weighted Turkish vocabularies and tests it on the nonce items |
| 10 | `10_rulex_benchmark.py` | Validates the RULEX implementation (`rulex.py`) against Nosofsky, Palmeri & McKinley's (1994) published results |
| 11 | `11_rulex_turkish.py` | Trains RULEX on the same vocabularies as stage 09 (`training_data.py`) and tests it on the nonce items |
| 12 | `12_alcove_sensitivity.py` | Reruns the 5,000-word ALCOVE condition with c, φ and λw halved and doubled, with slow attention, and with attention off |
| 13 | `13_human_survey.py` | Scores the 97 human wug-test responses and reports P(-ler) per cue level and per participant |
| 14 | `14_frequency.py` | Type-weighted vs token-weighted lexical rates, against the human profile |
| 15 | `15_token_training.py` | Trains both models with type, token and log-token presentation of the same vocabularies (`training_data.epoch_sampler`) |
| 16 | `16_model_vs_human.py` | Compares both models with the human data: cue-level means (RMSD, r) and the share of individuals answering all -lar, all -ler or mixed |

Shared modules: `common.py` (phonology, feature definitions, statistics),
`alcove.py` and `rulex.py` (the models), and `training_data.py` (vocabulary
sampling, nonce items and presentation schemes, shared by every training
stage so both models see identical inputs).

## The human survey

- **Task.** A written production task on Google Forms. A frame sentence
  introduced each invented word and participants wrote the plural themselves:
  *Dün birkaç [WORD] gördüm. Bu ______ çok ilginçti.* There was no forced
  choice between *-lar* and *-ler*, since offering it would show what is being
  measured.
- **Items.** The 24 stage-07 nonce words, in one fixed order. Stage 13 fails if
  the surveyed words and the generated words ever disagree.
- **Stems.** *teşa-, deşa-, şida-, yeşa-*: disharmonic (front then back
  vowel), like the Arabic loans in the exception class (*seyahat, dikkat*) and
  unlike native roots. None is a real word or within one edit of one. The stems
  do not affect the models, whose features read only the final syllable.
- **Participants.** 97 native Turkish speakers, aged 17–59 (median 23). 51
  have studied linguistics or Turkish philology. One participant was 17 and is
  included. The regression reports every test with and without them, and
  without them the cue-level rates move by under one percentage point.
- **Data.** `data/survey_responses.csv`, the Google Forms export with the
  Timestamp column removed. No names or email addresses were collected. Unlike
  TELL, it cannot be re-downloaded, so it is committed as an exception in
  `.gitignore`.

## Figures

`figures/make_figures.py` draws the report figures from the pipeline's output
files. It is the only part of the project that needs matplotlib, so it runs in
its own environment and the pipeline stays standard-library only:

```bash
python3 -m venv .venv && .venv/bin/pip install matplotlib
.venv/bin/python figures/make_figures.py
```

| Figure | Shows |
|---|---|
| `fig1_alcove_benchmark` | People and ALCOVE learning the six SHJ types, with attention learning and with it frozen |
| `fig2_rulex_benchmark` | RULEX against Nosofsky et al.'s (1994) published 5-4 predictions, and its SHJ ordering |
| `fig3_nonce_profiles` | Both models on the nonce items, beside the dictionary rates and the Tolerance Principle, per vocabulary size and lateral condition |
| `fig4_learner_split` | *teşaat* per learner: ALCOVE's graded answers vs RULEX's all-or-none ones |
| `fig5_individuals` | People vs models: share of individuals answering each cue level all -lar, mixed or all -ler (stage 16) |

Each is written as PNG and PDF. Colours follow one rule across figures:
ALCOVE is always blue, RULEX always orange.

## Statistics

`stats/mixed_model.R` fits a mixed-effects logistic regression to the human
responses (stage 13): response (-ler = 1) by cue level and linguistics training,
with random intercepts for participant and item, using `lme4`. Its output is
committed as `stats/mixed_model_results.txt`, so it can be read without R.
Like the figures, it needs a package the pipeline does not, installed into a
project-local library (see the comments at the top of the script for the
R 4.3 install route):

```bash
Rscript stats/mixed_model.R
```

It tests:

- whether cue level matters at all
- each cue level against no cue
- the contrasts between cue levels, including /k/ vs /h/ (the type-vs-token
  reversal)
- whether linguistics training changes the cue effect
- two robustness checks: without the one participant under 18, and with
  by-participant random slopes for cue

## Outputs

| File | Contents |
|---|---|
| `output/01_candidates.tsv` | Every classified nominal with its phonological coding |
| `output/02_phase0_cues.tsv` | Effect size for each candidate cue |
| `output/03_at_class.tsv` | The cleaned 294-item `/at/` class |
| `output/05_plural_check.tsv` | Per-item corpus plural counts |
| `output/06_model_matrix.tsv` | **Handoff file.** Ten binary dimensions per item (D0–D9). Train on `plural_ler` (1 = -ler); `is_exception` is for analysis only |
| `output/07_nonce_items.tsv` | The screened 24-item wug set, coded on the same dimensions |
| `output/08_alcove_shj.tsv` | Human and ALCOVE learning curves on the six SHJ types, ALCOVE with and without attention learning |
| `output/09_alcove_nonce.tsv` | ALCOVE P(-ler) for each simulated learner and nonce cue level |
| `output/10_rulex_benchmarks.tsv` | RULEX on the Medin & Schaffer 5-4 structure and the six SHJ types, beside the published values |
| `output/11_rulex_nonce.tsv` | RULEX P(-ler) per learner and cue level, for the main and three sensitivity parameter settings |
| `output/12_alcove_sensitivity.tsv` | ALCOVE P(-ler) per learner and cue level for each of the nine parameter settings |
| `output/13_human_responses.tsv` | One row per response: participant, item, cue level, -ler coded 0/1 |
| `output/13_human_profiles.tsv` | Per-participant cue rate and profile type |
| `output/13_human_by_cue_level.tsv` | The human profile the models must match |
| `output/14_frequency_rates.tsv` | Type and token exception rates per cue level |
| `output/15_token_training.tsv` | ALCOVE and RULEX P(-ler) per learner and cue level under type, token and log-token training |
| `output/16_model_vs_human.tsv` | Model-vs-human fit per model and training scheme, and individual-level response shapes |

## Data sources

**TELL** (Turkish Electronic Living Lexicon), Inkelas, Küntay, Orgun & Sprouse,
UC Berkeley. Phonemic transcriptions of ~30k lexemes elicited from a native
speaker. Download URL is the HTTP tilde path
`http://linguistics.berkeley.edu/~tell/telldata_all.zip`; the obvious HTTPS
`/TELL/` path 404s.

**OpenSubtitles-2018 Turkish frequency list** from `hermitdave/FrequencyWords`,
~2.0m types. Uses:

- the plural spot-check (stage 05)
- screening the nonce items (stage 07)
- frequency-weighted vocabulary sampling and token presentation for the models
  (`training_data.py`: stages 09, 11, 12, 15)
- the type-vs-token rates (stage 14)

**The survey responses** (`data/survey_responses.csv`), collected for this
project; see [The human survey](#the-human-survey).

**Human SHJ learning curves** (Nosofsky, Gluck, Palmeri, McKinley & Glauthier
1994, 40 participants per type), as distributed in the R package `catlearn`,
reproduced in stage 08.

### Licensing of the two downloaded datasets

Neither dataset is redistributed by this repository; both are downloaded at
run time by stage 00 and gitignored locally. Nothing here changes if you
never run the pipeline.

- **TELL**: no licence or terms of use are stated on the Berkeley pages this
  project downloads from. If you use TELL, cite Inkelas, Küntay, Orgun &
  Sprouse's TELL papers as an academic courtesy.
- **FrequencyWords**: the generator code is MIT; the word-list content itself
  (what stage 00 downloads) is **CC BY-SA 4.0**, per the repository's own
  README. Attribution: hermitdave/FrequencyWords, OpenSubtitles-2018 Turkish
  list. Since the file is only downloaded and read, never redistributed or
  published in this repository, ShareAlike is not triggered. If you ever
  commit a copy of the raw list or a close derivative of it, that clause
  applies.

## Design decisions encoded in the code

**Testing is on nonce items, not held-out real words.** Albright & Hayes (2003)
ran the held-out design over 4,253 English verbs; both a rule model and an
analogical model returned the regular form for essentially every held-out item.
Real speakers have memorised their exceptions and models have not, so the
comparison is unfair. Stage 07 builds the replacement.

**Features are binary and always defined.** RULEX has no not-applicable value,
so conditional features like "is the lateral clear?" cannot be used. Every
dimension in stage 06 is a property of the final syllable.

**Models learn the plural, not exception status.** They train on `plural_ler`
(1 = -ler). Training on `is_exception` would put front-vowel words and
back-vowel regulars in the same class and hide the harmony rule itself.

**The segment before the final vowel is three dimensions, not one.**
`D2_onset_vowel`, `D3_onset_h` and `D4_onset_dorsal` are separate, mutually
exclusive flags. An earlier version had a single D2 (vowel or /h/) and no
velar-dorsal dimension. That coded the nonce items *teşaat* and *teşahat*
identically, and *teşakat* identically to the no-cue items, so neither model
could have reproduced the 71/41/21 ordering the nonce test is built to detect.
The features are now defined once, in `common.MODEL_DIMENSIONS`. Stages 06 and
07 both call that function, and stage 07 stops the run if any two cue levels
share a feature vector. `AT_CUE_CLASS` (vowel, /h/ or any dorsal) is still the
`/at/`-local analysis finding used in stages 02–04. It is not a model input.

**The models are told a word ends in /at/ (`D9_at_final`).** The predicted
levels (71/41/21%) and the Tolerance Principle verdicts are stated over /at/-final
words only. Without D9, a nonce item like *teşaat* shares its feature vector with
every back-vowel hiatus word, most of which do not end in /at/, and the -ler
rates a model can see for the four cue levels drop to 34/18/6/0.6%. A first
ALCOVE run without D9 tracked exactly those diluted rates. D9 does not leak the
answer: the /at/ rime on its own is only 11.7% precise (stage 03).

**Learners have realistic vocabularies.** Each simulated learner samples 1,000,
2,000 or 5,000 words from TELL, weighted by corpus frequency; the 3,398 words
that never occur in the corpus are never sampled. Learner *k* gets the same
words in every training stage, for both models, so the models are compared on
identical input. Every condition is run with and without the lateral-final
words; it makes little difference.

**ALCOVE is validated against people before it touches Turkish.** Its
parameters (`alcove.SHJ_HUMAN_FIT`: c = 5.82, φ = 1.95, λw = 0.075, λα = 0.986)
are standard ALCOVE's best fit to human learning of the Shepard, Hovland &
Jenkins (1961) types, from the replication by Nosofsky et al. (1994), 40
participants per type, as distributed in `catlearn`. Stage 08 re-derives that
fit (RMSD .039 over the 96 human data points), reproduces the difficulty
ordering (I < II < III–V ≤ VI), and reproduces Kruschke's (1992) ablation: with
attention learning frozen, Type II loses its advantage over Type IV. One
limitation is reported rather than hidden: ALCOVE makes Type VI only slightly
harder than Types III–V, while people find it clearly hardest. An earlier
version used parameter values recalled from Kruschke (1992) that could not be
checked against the paper and learned far too slowly compared with people
(RMSD .152); the Turkish result was the same under both. Separately,
`alcove.py` was run against the reference implementation `slpALCOVE` in the R
package `catlearn` (v1.1) on the same trial sequences (SHJ Types II and VI,
and a random 9-dimension, 3-category problem). Choice probabilities agreed to
within 1e-10 on every trial.

**With the fitted parameters, ALCOVE answers by exact lookup, and it does not
matter.** The attention rate was fitted to a 256-trial task. Over up to 400,000
Turkish trials attention grows without bound (about 25 per dimension, from 0.1),
so ALCOVE answers a nonce item only from exemplars with an identical feature
vector: they carry 100% of *teşaat*'s activation (stage 12, "exact"). This is
possible because the ten features are coarse enough that every nonce item has
real-word twins. Stage 12 shows the result does not depend on it: with slow
attention (λα = .0033) identical exemplars carry 87% and similar ones the rest;
with attention off, only 9% comes from identical exemplars. In all three the
cue levels come out in the same order, and the slow-attention means match the
main ones to within .002. So the comparison is between remembering how often
words like this take *-ler* (ALCOVE) and a rule with a few stored exceptions
(RULEX); it does not rest on ALCOVE generalising across different sounds.

**RULEX is validated against the published numbers.** No reference
implementation exists, so stage 10 checks `rulex.py` against Nosofsky, Palmeri &
McKinley (1994). On Medin & Schaffer's 5-4 structure, with the paper's fitted
parameters, its predictions lie within RMSD .029 of the paper's own RULEX
predictions and .052 of the observed data (paper: .048), with the same split of
learners across Dimension-1 and Dimension-3 rules. It also reproduces the SHJ
ordering. The paper leaves some details open (listed in the `rulex.py`
docstring). One of them mattered: scoring a rule's accuracy on trials where it
gives no answer made two-dimension rules unlearnable and broke Type II, so
accuracy is scored only on trials where the rule responds.

**Type and token presentation are both run.** By default each word is shown
once per epoch (type presentation). Stage 15 also shows words in proportion to
their corpus frequency (token) and its logarithm (log-token), with the same
vocabularies, because the lexicon's type and token profiles order the cue
levels differently.

**Convergence is judged on condition means, not single learners.** Both models
train for 80 epochs. ALCOVE was checked to 320 epochs and RULEX to 160; neither
mean moved by more than about .03 after epoch 80. Single learners never settle
exactly: ALCOVE's fixed learning rate and random trial order, and RULEX's
storing and discarding of exceptions, keep them moving. Token-trained ALCOVE
means wobble by about ±.05 without trend (checked to 240 epochs); the
orderings held at every checkpoint.

**Models are scored like participants.** People give four yes/no answers per
cue level, so stage 16 scores each simulated learner as a participant
answering four items with its own P(-ler). RULEX is also scored with its
published response-error parameter, because without it RULEX cannot give a
mixed answer at all.

**No hybrid model (ATRIUM).** The original plan named the hybrid
rule-and-exemplar models from the course, especially ATRIUM (Erickson &
Kruschke 1998). In this feature space every nonce item has exact real-word
twins, and the exceptions sit almost entirely among the *-at* words. ATRIUM's
gate would route those to its exemplar module and apply its rule elsewhere, so
on these items it would almost certainly reproduce ALCOVE's predictions. This
is a reasoned prediction, not a tested one. The Tolerance Principle, whose
prediction (a step) differs from both a slope and a flat line, takes the third
place instead.

**No orthography anywhere in the lexical analysis.** Turkish spelling does not
reliably mark palatality or vowel length, so coding from written forms destroys
exactly the cues under investigation. Everything reads TELL's phonemic
transcription. (The survey is necessarily written; see the limitations.)

## Known limitations

**Data**

- TELL is one speaker's idiolect. The stage 05 corpus check is the only external
  validation, and it is spoken-register subtitles.
- Compound detection is a hand list in `03_at_control.py`, not an algorithm.
  Automatic detection was tried and rejected: requiring the final element to be
  an attested nominal wrongly flags Arabic words that merely end in another word
  (*cerahat* contains *rahat*). The hand list is printed in the stage output.
- 273 of the 375 `/at/` items have too little corpus evidence to spot-check.
- The corpus counts word forms, not nouns, so some token frequencies include
  other parts of speech (*fakat* is mostly the conjunction "but", *rahat* mostly
  the adjective "comfortable"). This inflates some regular *-at* tokens; it does
  not change the token order.

**Survey**

- 51 of the 97 participants have studied linguistics or Turkish philology.
  Every result is also broken out by training, and training specifically
  raises *-ler* on /h/ items.
- Written presentation cannot carry vowel length, and Turkish spelling does not
  mark palatal vs velar *k*, so a written *teşakat* is ambiguous between the
  two dorsal classes. An auditory replication would separate these.
- No dialect data: the province question did not make it into the final form.
- One participant was 17 and is included. Every regression test is reported
  with and without them; no conclusion changes, and the cue-level rates move by
  under one percentage point.

**Models**

- No model matches both the human order and the human magnitudes.
  Token-trained ALCOVE gets the order but is too extreme; type training gets
  closer magnitudes but the wrong order.
- The four stems are identical to the models (the features read only the final
  syllable), so for the models the 24 nonce items are four test items.
- Stage 12's "attention off" setting had not settled by epoch 80 and is
  reported as indicative.



## References

Albright, A. & Hayes, B. (2003). Rules vs. analogy in English past tenses.
*Cognition* 90, 119–161.

Bates, D., Mächler, M., Bolker, B. & Walker, S. (2015). Fitting linear
mixed-effects models using lme4. *Journal of Statistical Software* 67(1), 1–48.

Becker, M., Ketrez, N. & Nevins, A. (2011). The surfeit of the stimulus:
analytic biases filter lexical statistics in Turkish laryngeal alternations.
*Language* 87, 84–125.

Clements, G. N. & Sezer, E. (1982). Vowel and consonant disharmony in Turkish.
In H. van der Hulst & N. Smith (eds.), *The Structure of Phonological
Representations, Part II*, 213–255. Dordrecht: Foris.

Efraimidis, P. S. & Spirakis, P. G. (2006). Weighted random sampling with a
reservoir. *Information Processing Letters* 97, 181–185.

Erickson, M. A. & Kruschke, J. K. (1998). Rules and exemplars in category
learning. *Journal of Experimental Psychology: General* 127, 107–140.

Inkelas, S., Küntay, A., Orgun, C. O. & Sprouse, R. (2000). Turkish Electronic
Living Lexicon (TELL). *Turkic Languages* 4, 253–275.

Kruschke, J. K. (1992). ALCOVE: an exemplar-based connectionist model of
category learning. *Psychological Review* 99, 22–44.

Medin, D. L. & Schaffer, M. M. (1978). Context theory of classification
learning. *Psychological Review* 85, 207–238.

Nosofsky, R. M., Gluck, M. A., Palmeri, T. J., McKinley, S. C. & Glauthier, P.
(1994). Comparing models of rule-based classification learning: a replication
and extension of Shepard, Hovland, and Jenkins (1961). *Memory & Cognition* 22,
352–369.

Nosofsky, R. M., Palmeri, T. J. & McKinley, S. C. (1994). Rule-plus-exception
model of classification learning. *Psychological Review* 101, 53–79.

Shepard, R. N., Hovland, C. I. & Jenkins, H. M. (1961). Learning and
memorization of classifications. *Psychological Monographs* 75(13, Whole
No. 517).

Wills, A. J. et al. *catlearn: Formal Psychological Models of Categorization
and Learning*. R package, version 1.1. https://CRAN.R-project.org/package=catlearn

Yang, C. (2016). *The Price of Linguistic Productivity*. MIT Press.
