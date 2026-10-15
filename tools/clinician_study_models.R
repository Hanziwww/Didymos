# Paper P's clinician study: the primary mixed models (docs/paper/clinician_study_protocol.md, sections 5-8).
# Fixed before the first rating; run once on the locked tables of tools/clinician_study_analysis.py.
#
#   Rscript tools/clinician_study_models.R <analysis dir> [bootstrap draws, default 999]
#
# Reads a.csv, b1.csv, b2.csv and c.csv; writes primary_lme4.json beside them. Each model is fitted on the raters
# with at least 80% of the module answered (complete == 1, primary) and on all raters (sensitivity).
#
# * A: glmer(correct ~ 1 + (1 | rater) + (1 | item), binomial). The pooled accuracy is the population-averaged
#   probability of a correct call: the mean of plogis(b0 + z * sqrt(var_rater + var_item)) over z ~ N(0, 1), by a
#   999-node quantile rule. Its 90% interval is a parametric bootstrap (bootMer, percentile). Equivalence to chance:
#   the interval within 0.40-0.60. Also reported: plogis of the intercept with its 90% Wald interval.
# * B1: lmer(plausibility ~ twin + (1 | rater) + (1 | item) + (1 | item:version)), insulin excluded; Satterthwaite
#   df (lmerTest). A stimulus (an item in one version: one drawn rollout, or the real course) is its own random
#   effect, so the twin effect's error reflects between-stimulus variation, as the power simulation's per-item
#   contrasts did.
#   Non-inferiority: the lower bound of the 95% interval of the twin effect above -0.5.
# * B2: lmer(relevance ~ method + (1 | rater) + (1 | item) + (1 | item:method)), random as the reference (a stimulus:
#   an item with one method's three factors). Editing and knockout against random: two-sided tests, Holm-adjusted;
#   superior when the adjusted p < 0.05 and the effect is positive. Editing against knockout is secondary
#   (unadjusted).
# * C: per outcome, lmer(d ~ 1 + (1 | rater) + (1 | case)), d = (aided - y)^2 - (unaided - y)^2 per estimate. The
#   primary endpoint is ICU death, two-sided at 0.05; the other outcomes are secondary.

suppressPackageStartupMessages({
  library(lme4)
  library(lmerTest)
  library(jsonlite)
})

args <- commandArgs(trailingOnly = TRUE)
dir <- args[1]
nboot <- if (length(args) > 1) as.integer(args[2]) else 999L
SEED <- 20261010L
Z <- qnorm((seq_len(999) - 0.5) / 999)

rd <- function(f) read.csv(file.path(dir, f), stringsAsFactors = FALSE, na.strings = c("", "NA"))
a <- rd("a.csv")
b1 <- rd("b1.csv")
b2 <- rd("b2.csv")
cc <- rd("c.csv")

ctl <- glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 2e5))

marginal <- function(fit) {
  vc <- as.data.frame(VarCorr(fit))
  mean(plogis(fixef(fit)[[1]] + Z * sqrt(sum(vc$vcov))))
}

fit_a <- function(d) {
  fit <- glmer(correct ~ 1 + (1 | rater) + (1 | item), data = d, family = binomial, control = ctl)
  set.seed(SEED)
  bs <- bootMer(fit, marginal, nsim = nboot, type = "parametric", use.u = FALSE)
  t <- bs$t[is.finite(bs$t)]
  ci <- unname(quantile(t, c(0.05, 0.95)))
  w <- confint(fit, parm = "(Intercept)", method = "Wald", level = 0.90)
  vc <- as.data.frame(VarCorr(fit))
  list(raters = length(unique(d$rater)), items = length(unique(d$item)), calls = nrow(d),
       accuracy = marginal(fit), lo90 = ci[1], hi90 = ci[2], boot_draws = length(t),
       equivalent = ci[1] > 0.40 && ci[2] < 0.60,
       conditional_accuracy = plogis(fixef(fit)[[1]]), conditional_lo90 = plogis(w[1]), conditional_hi90 = plogis(w[2]),
       sd_rater = sqrt(vc$vcov[vc$grp == "rater"]), sd_item = sqrt(vc$vcov[vc$grp == "item"]),
       singular = isSingular(fit), converged = is.null(fit@optinfo$conv$lme4$messages))
}

coef_row <- function(fit, term) {
  s <- coef(summary(fit, ddf = "Satterthwaite"))
  est <- s[term, "Estimate"]
  se <- s[term, "Std. Error"]
  df <- s[term, "df"]
  list(est = est, se = se, df = df, lo95 = est - qt(0.975, df) * se, hi95 = est + qt(0.975, df) * se,
       p = s[term, "Pr(>|t|)"])
}

fit_b1 <- function(d) {
  d <- d[d$treatment != "insulin", ]
  fit <- lmer(plausibility ~ twin + (1 | rater) + (1 | item) + (1 | item:version), data = d)
  r <- coef_row(fit, "twin")
  c(list(raters = length(unique(d$rater)), items = length(unique(d$item)), ratings = nrow(d),
         non_inferior = r$lo95 > -0.5, singular = isSingular(fit)), twin_minus_real = list(r))
}

fit_b2 <- function(d) {
  d$method <- relevel(factor(d$method), ref = "random")
  fit <- lmer(relevance ~ method + (1 | rater) + (1 | item) + (1 | item:method), data = d)
  e <- coef_row(fit, "methodedit")
  k <- coef_row(fit, "methodknockout")
  adj <- p.adjust(c(e$p, k$p), method = "holm")
  e$p_holm <- adj[1]
  k$p_holm <- adj[2]
  d$method <- relevel(d$method, ref = "knockout")
  fit2 <- lmer(relevance ~ method + (1 | rater) + (1 | item) + (1 | item:method), data = d)
  list(raters = length(unique(d$rater)), items = length(unique(d$item)), ratings = nrow(d),
       edit_minus_random = e, knockout_minus_random = k, edit_minus_knockout = coef_row(fit2, "methodedit"),
       edit_superior = e$p_holm < 0.05 && e$est > 0, knockout_superior = k$p_holm < 0.05 && k$est > 0,
       singular = isSingular(fit))
}

fit_c <- function(d) {
  out <- list()
  for (o in c("icu_death", "vasopressor", "aki")) {
    x <- d[d$outcome == o, ]
    if (nrow(x) == 0) next
    x$d <- (x$aided - x$label)^2 - (x$unaided - x$label)^2
    fit <- lmer(d ~ 1 + (1 | rater) + (1 | case), data = x)
    out[[o]] <- c(list(raters = length(unique(x$rater)), cases = length(unique(x$case)), estimates = nrow(x),
                       brier_unaided = mean((x$unaided - x$label)^2), brier_aided = mean((x$aided - x$label)^2),
                       singular = isSingular(fit)),
                  aided_minus_unaided = list(coef_row(fit, "(Intercept)")))
  }
  out
}

both <- function(f, d) list(primary = f(d[d$complete == 1, ]), sensitivity_all_raters = f(d))

res <- list(
  A = both(fit_a, a),
  B1 = both(fit_b1, b1),
  B2 = both(fit_b2, b2),
  C = both(fit_c, cc),
  environment = list(R = R.version.string, lme4 = as.character(packageVersion("lme4")),
                     lmerTest = as.character(packageVersion("lmerTest")),
                     Matrix = as.character(packageVersion("Matrix")), bootstrap_draws = nboot, seed = SEED)
)
writeLines(toJSON(res, auto_unbox = TRUE, digits = NA, pretty = TRUE), file.path(dir, "primary_lme4.json"))
p <- res$A$primary
cat(sprintf("A accuracy %.3f [%.3f, %.3f] equivalent %s | B1 twin-real %.2f lo %.2f | C icu_death %.4f p %.3g\n",
            p$accuracy, p$lo90, p$hi90, p$equivalent, res$B1$primary$twin_minus_real$est,
            res$B1$primary$twin_minus_real$lo95, res$C$primary$icu_death$aided_minus_unaided$est,
            res$C$primary$icu_death$aided_minus_unaided$p))
