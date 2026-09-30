# Mixed-effects logistic regression of the human wug-test responses.
#
# Are the differences between cue levels in Stage 13 reliable, once we allow
# for some participants and some items simply leaning towards -ler overall?
#
#   response (1 = -ler) ~ cue level + linguistics training
#                         + (1 | participant) + (1 | item)
#
# Needs lme4 (R). From the repo root, after the pipeline has run Stage 13:
#
#   mkdir -p .Rlib
#   Rscript -e "install.packages('lme4', lib='.Rlib', repos='https://cloud.r-project.org')"
#   Rscript stats/mixed_model.R
#
# On R 4.3 the current lme4 binary is built for a newer Matrix and warns of an
# ABI mismatch; install the archived source release instead, which compiles
# against the installed Matrix:
#   Rscript -e "install.packages('https://cran.r-project.org/src/contrib/Archive/lme4/lme4_1.1-35.5.tar.gz', lib='.Rlib', repos=NULL, type='source')"
#
# Writes stats/mixed_model_results.txt (committed, so the results can be read
# without installing lme4).

.libPaths(c(".Rlib", .libPaths()))
suppressPackageStartupMessages(library(lme4))

d <- read.delim("output/13_human_responses.tsv", encoding = "UTF-8")
d$participant <- factor(d$participant)
d$item <- factor(d$item)
d$ling <- factor(d$linguistics, levels = c("N", "Y"))
d$cue <- factor(d$cue_level, levels = c("no cue", "/h/", "velar dorsal", "hiatus"))

# Ages, for the robustness check without under-18s. Stage 13 numbers
# participants by row of data/survey_responses.csv, starting at 1.
raw <- read.csv("data/survey_responses.csv", encoding = "UTF-8", check.names = FALSE)
age <- suppressWarnings(as.integer(raw[[grep("^Ya", names(raw))[1]]]))
d$age <- age[as.integer(as.character(d$participant))]

ctrl <- glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 2e5))
fit <- function(formula, data) glmer(formula, data = data, family = binomial, control = ctrl)

or_table <- function(m, keep) {
  s <- summary(m)$coefficients
  s <- s[rownames(s) %in% keep, , drop = FALSE]
  data.frame(odds_ratio = round(exp(s[, 1]), 2),
             ci_low = round(exp(s[, 1] - 1.96 * s[, 2]), 2),
             ci_high = round(exp(s[, 1] + 1.96 * s[, 2]), 2),
             z = round(s[, 3], 2),
             p = signif(s[, 4], 3),
             row.names = rownames(s))
}

sink("stats/mixed_model_results.txt", split = TRUE)
cat("MIXED-EFFECTS LOGISTIC REGRESSION, human responses (Stage 13)\n")
cat(sprintf("%d responses, %d participants (%d with linguistics training), %d items\n\n",
            nrow(d), nlevels(d$participant),
            length(unique(d$participant[d$ling == "Y"])), nlevels(d$item)))

# 1. Main model. Reference level = no cue.
m <- fit(response_ler ~ cue + ling + (1 | participant) + (1 | item), d)
m0 <- fit(response_ler ~ ling + (1 | participant) + (1 | item), d)
cat("=== 1. MAIN MODEL: response ~ cue + linguistics + (1|participant) + (1|item) ===\n")
cat("Does cue level matter at all? Likelihood-ratio test against a model without it:\n")
print(anova(m0, m))
cat("\nEach cue level against no cue, and linguistics training (odds ratios, 95% CI):\n")
print(or_table(m, c("cue/h/", "cuevelar dorsal", "cuehiatus", "lingY")))
cat("\nRandom-effect SDs (log-odds): participant ",
    round(attr(VarCorr(m)$participant, "stddev"), 2), ", item ",
    round(attr(VarCorr(m)$item, "stddev"), 2), "\n", sep = "")

# 2. The comparisons between cue levels: refit with /h/ as the reference.
d2 <- d
d2$cue <- relevel(d2$cue, ref = "/h/")
m2 <- fit(response_ler ~ cue + ling + (1 | participant) + (1 | item), d2)
cat("\n=== 2. BETWEEN CUE LEVELS (reference /h/) ===\n")
cat("velar dorsal vs /h/ is the type/token reversal: the lexicon by types puts /h/\n")
cat("above /k/, the lexicon by tokens and the humans put /k/ above /h/.\n")
print(or_table(m2, c("cuevelar dorsal", "cuehiatus")))
# hiatus vs velar dorsal as a contrast within the same fit (difference of two
# coefficients, with their covariance), not by refitting with yet another
# reference level: a refit can land on a degenerate Hessian and report a
# collapsed standard error.
b <- fixef(m2); V <- as.matrix(vcov(m2))
est <- b["cuehiatus"] - b["cuevelar dorsal"]
se <- sqrt(V["cuehiatus", "cuehiatus"] + V["cuevelar dorsal", "cuevelar dorsal"]
           - 2 * V["cuehiatus", "cuevelar dorsal"])
cat("\nhiatus vs velar dorsal (contrast within the same model):\n")
print(data.frame(odds_ratio = round(exp(est), 2), ci_low = round(exp(est - 1.96 * se), 2),
                 ci_high = round(exp(est + 1.96 * se), 2), z = round(est / se, 2),
                 p = signif(2 * pnorm(-abs(est / se)), 3), row.names = "hiatus - velar"))

# 3. Does training change the cue effect?
mi <- fit(response_ler ~ cue * ling + (1 | participant) + (1 | item), d)
cat("\n=== 3. DOES LINGUISTICS TRAINING CHANGE THE CUE EFFECT? ===\n")
cat("Likelihood-ratio test, cue x linguistics interaction:\n")
print(anova(m, mi))
print(or_table(mi, grep(":", rownames(summary(mi)$coefficients), value = TRUE)))

# 4. Robustness.
cat("\n=== 4. ROBUSTNESS ===\n")
adult <- d[!is.na(d$age) & d$age >= 18, ]
ma <- fit(response_ler ~ cue + ling + (1 | participant) + (1 | item), adult)
cat(sprintf("a) Without participants under 18 (%d excluded):\n",
            nlevels(d$participant) - length(unique(adult$participant))))
print(or_table(ma, c("cue/h/", "cuevelar dorsal", "cuehiatus", "lingY")))
adult2 <- adult
adult2$cue <- relevel(adult2$cue, ref = "/h/")
ma2 <- fit(response_ler ~ cue + ling + (1 | participant) + (1 | item), adult2)
cat("   velar dorsal vs /h/ without under-18s:\n")
print(or_table(ma2, c("cuevelar dorsal")))

cat("\nb) By-participant random slopes for cue (each person their own cue effect):\n")
ms <- tryCatch(fit(response_ler ~ cue + ling + (1 + cue | participant) + (1 | item), d2),
               error = function(e) NULL)
if (is.null(ms)) {
  cat("   did not fit\n")
} else {
  cat(sprintf("   singular fit: %s (TRUE means the slopes are not identifiable;\n",
              isSingular(ms)))
  cat("   the intercept-only model above is then the one to report)\n")
  print(or_table(ms, c("cuevelar dorsal", "cuehiatus", "cueno cue")))
}
sink()
