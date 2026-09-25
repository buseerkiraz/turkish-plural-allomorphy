# Rule vs. Exemplar Learning of Turkish Plural Allomorphy

Course project for "From Neurons to Transformers: Cognitive Principles in
AI Research" (Heidelberg University, SoSe 2026).

## Team
Niyousha Mojoudi, Buse Erkiraz

## Running it

Python 3.7 or newer. No packages need to be installed.
Stage 00 needs network access; every later stage is offline. A full run takes
about 45 minutes once the data is cached, almost all of it stages 09, 11 and 12 (model training, run in parallel on all cores); stages 00–08 and 10 take under a minute, plus roughly a minute the first time
for the 33 MB of downloads.

macOS / Linux:

```bash
bash run_all.sh
```

Windows (Command Prompt or PowerShell):

```
run_all.bat
```

Windows users with Git Bash or WSL can use `run_all.sh` instead.

To run a single stage, or to re-run one after editing it:

```bash
python3 src/03_at_control.py      # stages are independent once 01 has run
```

Stages must be run in order the first time, because 02 onwards read
`output/01_candidates.tsv`. After that any stage can be re-run on its own.

Every stage prints its full findings to the terminal and `run_all` also saves
them to `logs/`. If you only want the numbers quoted in the write-up, read the
logs rather than the TSVs.

## The research question

Turkish selects the plural suffix `-lAr` by the backness of the stem's final
vowel: back → `-lar`, front → `-ler`. A small set of nouns, mostly Arabic and
French loans, have a back final vowel but take `-ler` anyway (*saat → saatler*).

According to RULEX (Nosofsky, Palmeri & McKinley 1994), learners acquire a rule and
memorise the exceptions as a separate list; while ALCOVE (Kruschke 1992) says there is
no rule and no list, only stored exemplars and similarity-based generalisation.
The two agree on known words and diverge on novel ones. This pipeline builds the
dataset and test items needed to make them diverge measurably.

## Stages

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
| 08 | `08_alcove_benchmark.py` | Validates the ALCOVE implementation (`alcove.py`) on Shepard, Hovland & Jenkins (1961) |
| 09 | `09_alcove_turkish.py` | Trains ALCOVE on frequency-weighted Turkish vocabularies and tests it on the nonce items |
| 10 | `10_rulex_benchmark.py` | Validates the RULEX implementation (`rulex.py`) against Nosofsky, Palmeri & McKinley's (1994) published results |
| 11 | `11_rulex_turkish.py` | Trains RULEX on the same vocabularies as stage 09 (`training_data.py`) and tests it on the nonce items |
| 12 | `12_alcove_sensitivity.py` | Reruns the 5,000-word ALCOVE condition with each parameter halved and doubled, and with attention learning off |

## Figures

`figures/make_figures.py` draws the report figures from the pipeline's output
files. It is the only part of the project that needs a third-party package
(matplotlib), so it runs in its own environment and the pipeline stays
standard-library only:

```bash
python3 -m venv .venv && .venv/bin/pip install matplotlib
.venv/bin/python figures/make_figures.py
```

| Figure | Shows |
|---|---|
| `fig1_alcove_benchmark` | ALCOVE on the six SHJ types, with attention learning and with it frozen |
| `fig2_rulex_benchmark` | RULEX against Nosofsky et al.'s (1994) published 5-4 predictions, and its SHJ ordering |
| `fig3_nonce_profiles` | Both models on the nonce items, beside the dictionary rates and the Tolerance Principle, per vocabulary size and lateral condition |
| `fig4_learner_split` | *kunaat* per learner: ALCOVE's graded answers vs RULEX's all-or-none ones |

Each is written as PNG and PDF. Colours follow one rule across figures:
ALCOVE is always blue, RULEX always orange.

## Data sources

**TELL** (Turkish Electronic Living Lexicon), Inkelas, Küntay, Orgun & Sprouse,
UC Berkeley. Phonemic transcriptions of ~30k lexemes elicited from a native
speaker. Download URL is the HTTP tilde path
`http://linguistics.berkeley.edu/~tell/telldata_all.zip`; the obvious HTTPS
`/TELL/` path 404s.

**OpenSubtitles-2018 Turkish frequency list** from `hermitdave/FrequencyWords`,
~2.0m types. Used only in stage 05.

### Licensing of the two datasets

Neither dataset is redistributed by this repository; both are downloaded at
run time by stage 00 and gitignored locally. Nothing here changes if you
never run the pipeline.

- **TELL**: no licence or terms of use are stated on the Berkeley pages this
  project downloads from. If you use TELL, cite Inkelas, Küntay, Orgun &
  Sprouse's TELL papers as an academic courtesy.
- **FrequencyWords**: the generator code is MIT; the word-list content itself
  (what stage 05 downloads) is **CC BY-SA 4.0**, per the repository's own
  README. Attribution: hermitdave/FrequencyWords, OpenSubtitles-2018 Turkish
  list. Since the file is only downloaded and read, never redistributed or
  published in this repository, ShareAlike is not triggered. If you ever
  commit a copy of the raw list or a close derivative of it, that clause
  applies.

## Headline results

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
  Inside that class **34/84 items are exceptions (40.5%)**; outside it **2/211
  (0.9%)**. phi = 0.545, recall 94.4%.
- **The lateral cue is circular and unusable.** TELL writes dark `ɫ` after front
  vowels in 369 of 370 cases, which is phonetically wrong, so its `l`/`ɫ`
  contrast tracks harmony class rather than sound. Lateral-final items remain as
  training vocabulary but are excluded from reported results.
- The dorsal contrast is clean (0 palatal dorsals in 649 front-vowel items), and
  the winning cue does not read the palatality diacritic at all.
- Tolerance Principle, checked in both directions: the /at/ cue class as a whole
  (84 items, 34 `-ler`) licenses **no productive rule**. Its sub-classes differ:
  hiatus licenses a local `-ler` sub-rule (12/17), velar /k/ keeps `-lar` (5/24),
  and /h/ licenses neither (9/22). So the Tolerance Principle predicts a stepped
  profile on the nonce items, with *kunakat* falling with the no-cue items.
- Plural spot-check: **269/280 testable items agree** with TELL's
  accusative-based class (96.1%); 99/102 within the `/at/` class.

## Outputs

| File | Contents |
|---|---|
| `output/01_candidates.tsv` | Every classified nominal with its phonological coding |
| `output/02_phase0_cues.tsv` | Effect size for each candidate cue |
| `output/03_at_class.tsv` | The cleaned 295-item `/at/` class |
| `output/05_plural_check.tsv` | Per-item corpus plural counts |
| `output/06_model_matrix.tsv` | **Handoff file.** Ten binary dimensions per item (D0–D9). Train on `plural_ler` (1 = -ler); `is_exception` is for analysis only |
| `output/07_nonce_items.tsv` | The screened 24-item wug set, coded on the same dimensions |
| `output/08_alcove_shj.tsv` | ALCOVE learning curves on the six SHJ types, with and without attention learning |
| `output/09_alcove_nonce.tsv` | ALCOVE P(-ler) for each simulated learner and nonce cue level |
| `output/10_rulex_benchmarks.tsv` | RULEX on the Medin & Schaffer 5-4 structure and the six SHJ types, beside the published values |
| `output/11_rulex_nonce.tsv` | RULEX P(-ler) per learner and cue level, for the main and three sensitivity parameter settings |
| `output/12_alcove_sensitivity.tsv` | ALCOVE P(-ler) per learner and cue level for each of the ten parameter settings |

## Design decisions encoded in the code

**Testing is on nonce items, not held-out real words.** Albright & Hayes (2003)
ran the held-out design over 4,253 English verbs; both a rule model and an
analogical model returned the regular form for essentially every held-out item.
Real speakers have memorised their exceptions and models have not, so the
comparison is unfair. Stage 07 builds the replacement.

**Features are binary and always defined.** RULEX has no not-applicable value,
so conditional features like "is the lateral clear?" cannot be used. Every
dimension in stage 06 is a property of the final syllable.

**The segment before the final vowel is three dimensions, not one.**
`D2_onset_vowel`, `D3_onset_h` and `D4_onset_dorsal` are separate, mutually
exclusive flags. An earlier version had a single D2 (vowel or /h/) and no
velar-dorsal dimension. That coded the nonce items *kunaat* and *kunahat*
identically, and *kunakat* identically to the no-cue items, so neither model
could have reproduced the 71/41/21 ordering the nonce test is built to detect.
The features are now defined once, in `common.MODEL_DIMENSIONS`. Stages 06 and
07 both call that function, and stage 07 stops the run if any two cue levels
share a feature vector. `AT_CUE_CLASS` (vowel, /h/ or any dorsal) is still the
`/at/`-local analysis finding used in stages 02–04. It is not a model input.

**The models are told a word ends in /at/ (`D9_at_final`).** The predicted
levels (71/41/21%) and the Tolerance Principle verdicts are stated over /at/-final
words only. Without D9, a nonce item like *kunaat* shares its feature vector with
every back-vowel hiatus word, most of which do not end in /at/, and the -ler
rates a model can see for the four cue levels drop to 34/18/6/0.6%. A first
ALCOVE run without D9 tracked exactly those diluted rates. D9 does not leak the
answer: the /at/ rime on its own is only 11.7% precise (stage 03).

**ALCOVE is validated before it touches Turkish.** Stage 08 reproduces the
Shepard, Hovland & Jenkins (1961) difficulty ordering (I < II < III–V < VI) and
Kruschke's (1992) ablation: with attention learning frozen, Type II loses its
advantage over Type IV. Separately, `alcove.py` was run against the reference
implementation `slpALCOVE` in the R package `catlearn` (v1.1) on the same trial
sequences (SHJ Types II and VI, and a random 9-dimension, 3-category problem).
Choice probabilities agreed to within 1e-10 on every trial.

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

**No orthography anywhere.** Turkish spelling does not reliably mark palatality
or vowel length, so coding from written forms destroys exactly the cues under
investigation. Everything reads TELL's phonemic transcription.

## Known limitations

- TELL is one speaker's idiolect. The stage 05 corpus check is the only external
  validation, and it is spoken-register subtitles.
- Compound detection is a hand list in `03_at_control.py`, not an algorithm.
  Automatic detection was tried and rejected: requiring the final element to be
  an attested nominal wrongly flags Arabic words that merely end in another word
  (*cerahat* contains *rahat*). The hand list is printed in the stage output.
- 273 of the 375 `/at/` items have too little corpus evidence to spot-check.
- Stage 07's nonce forms are screened for non-wordhood but have not been rated
  for phonotactic naturalness by native speakers.

## References

Albright, A. & Hayes, B. (2003). Rules vs. analogy in English past tenses.
*Cognition* 90, 119–161.

Becker, M., Ketrez, N. & Nevins, A. (2011). The surfeit of the stimulus:
analytic biases filter lexical statistics in Turkish laryngeal alternations.
*Language* 87, 84–125.

Clements, G. N. & Sezer, E. (1982). Vowel and consonant disharmony in Turkish.

Kruschke, J. K. (1992). ALCOVE: an exemplar-based connectionist model of
category learning. *Psychological Review* 99, 22–44.

Nosofsky, R. M., Palmeri, T. J. & McKinley, S. C. (1994). Rule-plus-exception
model of classification learning. *Psychological Review* 101, 53–79.

Yang, C. (2016). *The Price of Linguistic Productivity*. MIT Press.
