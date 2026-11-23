# ============================================================
# No-pooling SAM burden model with LLM MIXTURE priors -- Somalia, 74 districts
# (Draft 1)
#
# Variant of no_pooling_LLM_informed_priors.R. That script pools an LLM's 10 elicited runs into ONE Beta per
# district and parameter by moment matching (an approximation of the mixture of the runs). Here Stan receives
# the 10 runs per district intact and the prior is the actual mixture over runs (no approximation):
#   joint mixture    prior(p, gamma, omega | district) = sum_k w_k Beta(p|run k) Beta(gamma|run k) Beta(omega|run k)
#   separate mixture one mixture per parameter, independent a priori
#   fixed weights    w_k = 1/K          estimated weights    one simplex w shared by all districts, Dirichlet(1,..,1)
# Likelihood, state-space recursion, P0 prior, burden, sampler settings, seeds, data and district order are those of
# the existing model. NOTHING in the existing files is modified: this script only adds files.
#
# MODEL IDS (cache/<scheme>/<id>.rds, outputs/parameter_and_burden_summary_<id>.csv, ...)
#   <llm>_mixture_joint_fixedw    PRIMARY: joint mixture, w = 1/K
#   <llm>_mixture_joint_estw      joint mixture, estimated shared weights
#   <llm>_mixture_sep_fixedw      separate mixture per parameter, w = 1/K
#   chatgpt_mixture_check_K1_run01          validation: K = 1 (run 1 only) -> must reproduce chatgpt_run01
#   chatgpt_mixture_check_identical_run01   validation: K copies of run 1       -> must reproduce chatgpt_run01
#
# CONTENTS
#   0. Setup and batch controls
#   1. Helpers shared with the existing script (HMC diagnostics, prior JSON reader, moment-match pooling)
#   2. Data loading and cleaning (same as the existing script)
#   3. Mixture inputs: the K x D arrays handed to Stan
#   4. Pre-checks (no Stan): within-run dependence, mixture vs moment-matched Beta, between-run disagreement
#   5. Fits: Stan interface, cache, per-model outputs and diagnostics
#   6. Comparisons and validation (K = 1, identical components, mixture vs pooled)
#   7. LLO-2 / LLO-3 cross-validation of the mixture models
#
# RUNNING (batches via environment variables, defaults in brackets)
#   MIX_LLMS       comma list of LLMs to FIT     [chatgpt,gemini,claude]
#   MIX_VARIANTS   comma list of variants        [joint_fixedw,joint_estw,sep_fixedw]
#   MIX_SCHEMES    comma list of schemes to fit  [full,llo2]   (llo3 optional)
#   MIX_VALIDATION 1 = also fit the two validation models (chatgpt)  [1]
#   MIX_SMOKE      1 = short chains, ids get the suffix _smoke (wiring test, not a result)  [0]
#   Everything already cached is always used; tables are built from whatever is cached.
# ============================================================

library(rstan)
rstan_options(auto_write = TRUE)  # cache compiled models to disk (.rds next to the .stan
                                   # file) so re-running the script doesn't recompile Stan
                                   # code that hasn't changed -- fewer fresh-DLL loads means
                                   # fewer chances of hitting a Windows Application Control
                                   # block on the freshly-compiled binary.
library(dplyr)
library(ggplot2)
library(bayesplot)
library(loo)
library(tidyr)
library(readr)
library(sf)
library(viridis)
library(scales)
library(stringr)
library(jsonlite)
library(scoringRules)
library(ggrepel)

# ============================================================
# 0. SETUP AND BATCH CONTROLS
# ============================================================
PROJECT_DIR    <- "C:/Users/ecemy/OneDrive - INSEAD/Desktop/Waste prediction/Code/Draft 1/no_pooling_LLM_informed_priors"
HIER_DIR       <- file.path(dirname(PROJECT_DIR), "hierarchical_weakly_informative_priors")
DATA_PATH      <- "C:/Users/ecemy/OneDrive - INSEAD/Desktop/Waste prediction/Datasets/Final_Clean_Dataset.csv"
setwd(PROJECT_DIR)

.env_list <- function(name, default) strsplit(Sys.getenv(name, default), ",", fixed = TRUE)[[1]]
RUN_LLMS       <- .env_list("MIX_LLMS",     "chatgpt,gemini,claude")
RUN_VARIANTS   <- .env_list("MIX_VARIANTS", "joint_fixedw,joint_estw,sep_fixedw")
RUN_SCHEMES    <- .env_list("MIX_SCHEMES",  "full,llo2")
RUN_VALIDATION <- Sys.getenv("MIX_VALIDATION", "1") == "1"
SMOKE          <- Sys.getenv("MIX_SMOKE", "0") == "1"
FORCE_REFIT    <- FALSE
ID_SUFFIX      <- if (SMOKE) "_smoke" else ""

dir.create("outputs", showWarnings = FALSE, recursive = TRUE)
dir.create("plots",   showWarnings = FALSE, recursive = TRUE)
DIAG_DIR <- "outputs/diagnostics"                      # per-model Rhat/ESS and fit statistics (as for the existing models)
MIX_DIR  <- "outputs/mixture_priors"                   # files specific to the mixture models
for (.d in c(DIAG_DIR, MIX_DIR)) dir.create(.d, showWarnings = FALSE, recursive = TRUE)
CACHE_DIR <- "cache"                                   # same cache folder and layout as the existing models
for (.s in c("full", "llo2", "llo3")) dir.create(file.path(CACHE_DIR, .s), showWarnings = FALSE, recursive = TRUE)

HIER_ID <- "hierarchical"                              # read-only comparator, fitted in HIER_DIR

# ---- Sampler settings and seeds: identical to the existing fits ----
MCMC  <- list(chains = 4, warmup = 5000, iter = 10000, cores = 4)
SEEDS <- c(full = 123, llo2 = 456, llo3 = 789)
if (SMOKE) MCMC <- list(chains = 2, warmup = 300, iter = 600, cores = 2)
THIN_EVERY <- 5        # cached draws of p, gamma, omega and district-mean burden are thinned, as for the existing models

# Beta shapes of the fallback prior, used only for a district entirely absent from an LLM's JSON
FALLBACK_PRIOR_BETA <- list(p = c(2, 5), gamma = c(3, 200), omega = c(3, 200))
LLM_ALL <- c("claude", "gemini", "chatgpt")
RHAT_FLAG <- 1.01


# ============================================================
# 1. HELPERS SHARED WITH THE EXISTING SCRIPT
# ============================================================
# ---- Post-fit HMC diagnostic reporting ----
# Rhat/ESS alone cannot certify convergence -- divergent transitions, hitting
# max_treedepth, and low E-BFMI energy fraction are separate failure modes
# that Rhat/ESS can miss entirely. Call once per fit, right after print(fit).
.report_hmc_diagnostics <- function(fit, label) {
  n_div      <- tryCatch(rstan::get_num_divergent(fit),    error = function(e) NA_integer_)
  n_maxtd    <- tryCatch(rstan::get_num_max_treedepth(fit), error = function(e) NA_integer_)
  bfmi       <- tryCatch(rstan::get_bfmi(fit),               error = function(e) NA_real_)
  n_low_bfmi <- sum(bfmi < 0.2, na.rm = TRUE)
  message("HMC diagnostics [", label, "]: ", n_div, " divergent transitions, ",
          n_maxtd, " max-treedepth hits, ", n_low_bfmi, " chain(s) with E-BFMI < 0.2",
          if (any(!is.finite(bfmi))) " (BFMI unavailable for some chains)" else "",
          " -- Rhat/ESS alone do not certify convergence.")
  invisible(list(divergent = n_div, max_treedepth = n_maxtd, low_bfmi_chains = n_low_bfmi))
}

# ---- LLM prior inputs: constants, JSON reader, pooling ----
# (LLM_NAMES / file naming: priors_llm/priors_llm_<llm>_<run>.json)
PRIOR_JSON_DIR <- file.path(getwd(), "priors_llm")
LLM_NAMES      <- c("claude", "gemini", "chatgpt")


POSITIVITY_FLOOR <- 1e-6   # matches Stan's <lower=1e-6> bound on a_p/b_p/...

prior_json_path <- function(llm, run) {
  file.path(PRIOR_JSON_DIR, sprintf("priors_llm_%s_%d.json", llm, as.integer(run)))
}

# Run numbers available on disk for one LLM, ascending.
discover_prior_runs <- function(llm) {
  pat   <- sprintf("^priors_llm_%s_([0-9]+)\\.json$", llm)
  files <- list.files(PRIOR_JSON_DIR, pattern = pat)
  sort(as.integer(sub(pat, "\\1", files)))
}

# ---- Read district x {p, gamma, omega} Beta(a, b) priors from one LLM JSON ----
# A district absent from the JSON entirely falls back to default_prior (a
# data-availability gap); all such districts are reported in one loud summary.
# Malformed entries (missing parameter block, non-numeric / non-positive a or
# b) halt the run rather than being silently replaced. District names in the
# JSON that do not match any district in the data are reported too, since a
# spelling mismatch would otherwise show up as a silent default fallback.
# The prior actually used (a, b, mean, sd, 5/50/95% quantiles) is written to
# `summary_csv` for every district x parameter so it is auditable on disk.
read_priors_from_json <- function(json_path, default_prior, summary_csv = NULL) {
  raw       <- fromJSON(json_path, simplifyDataFrame = FALSE)
  districts <- raw$districts
  if (is.null(districts) || length(districts) == 0) {
    stop("read_priors_from_json: ", basename(json_path), " has no 'districts' array")
  }
  district_ids <- vapply(districts, function(x) as.character(x$district), character(1))

  dup_names <- unique(district_ids[duplicated(district_ids)])
  if (length(dup_names) > 0) {
    stop("read_priors_from_json: ", basename(json_path), " has ", length(dup_names),
         " ambiguous duplicate district name(s): ", paste(dup_names, collapse = ", "),
         " -- refusing to silently pick one entry over another.")
  }
  unknown_dns <- setdiff(district_ids, district_names)
  if (length(unknown_dns) > 0) {
    warning("read_priors_from_json: ", basename(json_path), " has district name(s) not found in ",
            "the data (ignored): ", paste(unknown_dns, collapse = ", "))
  }
  lookup <- setNames(districts, district_ids)

  PARAMS      <- c("p", "gamma", "omega")
  fitted      <- list()
  missing_dns <- character(0)
  rows        <- list()

  for (dn in district_names) {
    entry <- lookup[[dn]]
    if (is.null(entry)) {
      missing_dns  <- c(missing_dns, dn)
      fitted[[dn]] <- default_prior
      source_dn    <- "default_fallback"
    } else {
      missing_params <- setdiff(PARAMS, names(entry))
      if (length(missing_params) > 0) {
        stop("read_priors_from_json: district '", dn, "' in ", basename(json_path),
             " is missing parameter block(s): ", paste(missing_params, collapse = ", "))
      }
      fitted[[dn]] <- list()
      for (param in PARAMS) {
        ab <- entry[[param]]
        ok <- function(v) is.numeric(v) && length(v) == 1 && is.finite(v) && v >= POSITIVITY_FLOOR
        if (!ok(ab$a) || !ok(ab$b)) {
          stop("read_priors_from_json: district '", dn, "', parameter '", param, "' in ",
               basename(json_path), " needs numeric a and b >= ", POSITIVITY_FLOOR,
               " (got a=", paste(ab$a, collapse = ","), ", b=", paste(ab$b, collapse = ","), ")")
        }
        fitted[[dn]][[param]] <- list(a = as.numeric(ab$a), b = as.numeric(ab$b))
      }
      source_dn <- "json"
    }

    for (param in PARAMS) {
      a <- fitted[[dn]][[param]]$a
      b <- fitted[[dn]][[param]]$b
      rows[[length(rows) + 1]] <- tibble(
        district      = dn,
        parameter     = param,
        source        = source_dn,
        a             = a,
        b             = b,
        concentration = a + b,
        prior_mean    = a / (a + b),
        prior_sd      = sqrt(a * b / ((a + b)^2 * (a + b + 1))),
        prior_q05     = qbeta(0.05, a, b),
        prior_q50     = qbeta(0.50, a, b),
        prior_q95     = qbeta(0.95, a, b)
      )
    }
  }

  if (length(missing_dns) > 0) {
    cat("\n", strrep("*", 70), "\n",
        "*** MISSING DISTRICTS in ", basename(json_path), " (", length(missing_dns),
        " of ", length(district_names), ") -- falling back to default_prior:\n",
        paste0("  - ", missing_dns, collapse = "\n"), "\n",
        strrep("*", 70), "\n\n", sep = "")
  }

  if (!is.null(summary_csv)) {
    summary_df <- bind_rows(rows)
    dir.create(dirname(summary_csv), showWarnings = FALSE, recursive = TRUE)
    write_csv(summary_df, summary_csv)
    message("Saved: ", summary_csv, " (", nrow(summary_df), " district-parameter rows, ",
            length(missing_dns), " districts on default_prior fallback)")
  }

  fitted
}

# Missing-district fallback for the LLM-informed (no-pooling) model. These are
# standalone fallback Beta(a,b) distributions -- numerically the same shapes
# as FALLBACK_PRIOR_BETA (the hierarchical model's GLOBAL hyperpriors on
# p_global/gamma_global/omega_global) -- but they are NOT the hierarchical
# model's marginal district priors, and they are only ever used for a
# district entirely absent from the LLM's JSON (see read_priors_from_json()
# above), never substituted for a valid entry.
default_prior_ip <- list(
  p     = list(a = FALLBACK_PRIOR_BETA$p[1],     b = FALLBACK_PRIOR_BETA$p[2]),
  gamma = list(a = FALLBACK_PRIOR_BETA$gamma[1], b = FALLBACK_PRIOR_BETA$gamma[2]),
  omega = list(a = FALLBACK_PRIOR_BETA$omega[1], b = FALLBACK_PRIOR_BETA$omega[2])
)

extract_prior_vec <- function(fitted_priors, param, hyper) {
  sapply(district_names, function(dn) fitted_priors[[dn]][[param]][[hyper]])
}

# Compile the informed-priors Stan program ONCE and reuse the compiled object
# for every LLM x run (both here and in the CV blocks below) via sampling(),
# rather than calling stan(file = ...) repeatedly -- stan() recompiles from
# source on every call. Fewer fresh DLL builds also means fewer chances of

# ---- Pooled prior per LLM (combines the repeated prompt runs) ----
# Each LLM's runs are treated as repeated draws of the same elicitation. For
# every district x parameter the runs' Betas are combined as an equal-weight
# mixture and moment-matched back to ONE Beta(a, b), so the Stan model is
# unchanged:
#   m = sum_i w_i m_i,   V = sum_i w_i (v_i + m_i^2) - m^2   (law of total variance)
#   k = m (1 - m) / V - 1,   a = m k,   b = (1 - m) k
# The pooled mean is the average of the run means, and the pooled variance
# contains the BETWEEN-run spread on top of the within-run spread, so the pooled
# prior is wider exactly where the runs disagree. (A mixture of Betas can be
# bimodal when runs disagree strongly; the matched Beta is always unimodal and
# is only a summary of it.)
pool_priors_moment_match <- function(prior_list, weights = NULL) {
  n <- length(prior_list)
  w <- if (is.null(weights)) rep(1 / n, n) else weights / sum(weights)
  pooled <- list()
  for (dn in district_names) {
    pooled[[dn]] <- list()
    for (param in c("p", "gamma", "omega")) {
      a   <- vapply(prior_list, function(pl) pl[[dn]][[param]]$a, numeric(1))
      b   <- vapply(prior_list, function(pl) pl[[dn]][[param]]$b, numeric(1))
      m_i <- a / (a + b)
      v_i <- a * b / ((a + b)^2 * (a + b + 1))
      m   <- sum(w * m_i)
      V   <- sum(w * (v_i + m_i^2)) - m^2
      k   <- m * (1 - m) / V - 1
      pooled[[dn]][[param]] <- list(a = max(m * k, POSITIVITY_FLOOR),
                                    b = max((1 - m) * k, POSITIVITY_FLOOR))
    }
  }
  pooled
}

# ---- Tempered prior: the same Beta means with the concentration a+b multiplied by `f` ----
# f < 1 widens every district x parameter prior without moving its mean; f = 1 returns it
# unchanged. a and b stay above the positivity floor (Stan's lower bound).
temper_prior <- function(prior, f) {
  out <- list()
  for (dn in district_names) {
    out[[dn]] <- list()
    for (param in c("p", "gamma", "omega")) {
      ab <- prior[[dn]][[param]]
      out[[dn]][[param]] <- list(a = max(ab$a * f, POSITIVITY_FLOOR), b = max(ab$b * f, POSITIVITY_FLOOR))
    }
  }
  out
}
tempered_id <- function(llm, f) sprintf("%s_pooled_t%03d", llm, round(100 * f))

# ============================================================
# 2. DATA LOADING AND CLEANING
# ============================================================
# ----------------------------- Data work -----------------------------
df <- read.csv(DATA_PATH)

df_filtered <- df[!( (df$Year == 2016 & df$Month == 1) |
                       (df$Year == 2016 & df$Month == 7) |
                       (df$Year == 2017 & df$Month == 1)), ]

# Fill missing SAM in Rab Dhuure
rab_dhuure_data <- df_filtered[df_filtered$District == "Rab Dhuure", ]
avg_sam <- mean(rab_dhuure_data$SAM, na.rm = TRUE)
rab_dhuure_data$SAM[is.na(rab_dhuure_data$SAM)] <- avg_sam

df_filtered <- df_filtered %>%
  filter(District != "Rab Dhuure") %>%
  bind_rows(rab_dhuure_data)

df_filtered <- df_filtered %>%
  arrange(District, Year, Month)

df_filtered$district_id <- as.numeric(as.factor(df_filtered$District))
district_names <- levels(as.factor(df_filtered$District))

df_filtered <- df_filtered %>%
  rename(Pop_U5 = Population.Under.5)

# Build model frame
df_model <- df_filtered %>%
  filter(!is.na(District)) %>%
  arrange(District, Year, Month) %>%
  mutate(
    district_id = as.integer(factor(District, levels = unique(District))),
    Timepoint   = paste(Year, Month, sep = "-")
  )

district_names <- levels(factor(df_model$District, levels = unique(df_model$District)))

population_ref <- df_model %>%
  group_by(District) %>%
  summarise(Pop_U5_ref = mean(ifelse(Pop_U5 > 0, Pop_U5, NA), na.rm = TRUE), .groups = "drop")
pop_u5_ref <- population_ref$Pop_U5_ref[match(district_names, population_ref$District)]

district_starts <- which(!duplicated(df_model$district_id))

df_model <- tibble::as_tibble(df_model)
N <- nrow(df_model); D <- length(district_names)
stopifnot(length(unique(df_model$district_id)) == D)


# ============================================================
# 3. MIXTURE INPUTS
# For every LLM the K = 10 runs are read once (read_priors_from_json: same reader, same 1e-6 positivity floor as
# the single-Beta model). A model uses a subset / copy of them as K x D matrices.
# ============================================================
RUN_PRIORS <- list()      # RUN_PRIORS[[llm]][[run]] : district -> {p, gamma, omega} -> {a, b}
for (llm in LLM_ALL) {
  runs <- discover_prior_runs(llm)
  RUN_PRIORS[[llm]] <- setNames(lapply(runs, function(r) read_priors_from_json(prior_json_path(llm, r), default_prior_ip)), as.character(runs))
}

# K x D matrix of one hyperparameter over a list of per-run priors
.prior_matrix <- function(run_list, param, hyper) {
  K <- length(run_list)
  v <- vapply(district_names, function(dn) vapply(run_list, function(pl) pl[[dn]][[param]][[hyper]], numeric(1)), numeric(K))
  matrix(v, nrow = K, ncol = D, dimnames = list(NULL, district_names))
}

# ---- Model specifications ----
VARIANTS <- list(
  joint_fixedw = list(joint = 1L, estimate_w = 0L),
  joint_estw   = list(joint = 1L, estimate_w = 1L),
  sep_fixedw   = list(joint = 0L, estimate_w = 0L)
)
mixture_id <- function(llm, variant) paste0(llm, "_mixture_", variant, ID_SUFFIX)

MIX_SPECS <- list()
for (llm in LLM_ALL) for (v in names(VARIANTS)) {
  id <- mixture_id(llm, v)
  MIX_SPECS[[id]] <- c(list(id = id, llm = llm, variant = v, kind = "mixture", runs = seq_along(RUN_PRIORS[[llm]])), VARIANTS[[v]])
}
MIX_SPECS[[paste0("chatgpt_mixture_check_K1_run01", ID_SUFFIX)]] <-
  list(id = paste0("chatgpt_mixture_check_K1_run01", ID_SUFFIX), llm = "chatgpt", variant = "check_K1", kind = "check_K1", runs = 1L, joint = 1L, estimate_w = 0L)
MIX_SPECS[[paste0("chatgpt_mixture_check_identical_run01", ID_SUFFIX)]] <-
  list(id = paste0("chatgpt_mixture_check_identical_run01", ID_SUFFIX), llm = "chatgpt", variant = "check_identical", kind = "check_identical",
       runs = rep(1L, length(RUN_PRIORS[["chatgpt"]])), joint = 1L, estimate_w = 0L)

# per-spec list of per-run priors (a subset, or copies of run 1 for the identical-components check)
spec_run_priors <- function(spec) RUN_PRIORS[[spec$llm]][as.character(spec$runs)]

# The moment-matched Beta of the SAME runs (the quantity the existing <llm>_pooled model uses), in the standard
# district -> parameter -> (a, b) format: stored as `prior` in the cached fits so downstream code finds a Beta there.
MIX_MM_PRIOR <- lapply(MIX_SPECS, function(s) pool_priors_moment_match(unname(spec_run_priors(s))))

# Ids to fit in this session (cached models are always usable)
ids_to_fit <- function() {
  ids <- character(0)
  for (llm in intersect(RUN_LLMS, LLM_ALL)) for (v in intersect(RUN_VARIANTS, names(VARIANTS))) ids <- c(ids, mixture_id(llm, v))
  if (RUN_VALIDATION && "chatgpt" %in% RUN_LLMS) ids <- c(ids, paste0(c("chatgpt_mixture_check_K1_run01", "chatgpt_mixture_check_identical_run01"), ID_SUFFIX))
  ids
}
models_to_fit <- ids_to_fit
model_label <- function(model_id) model_id



# ============================================================
# 4. PRE-CHECKS (no Stan; all LLMs; written to outputs/mixture_priors/)
#  1. within-run dependence: correlation across districts of a run's p, gamma, omega prior means
#  2. overlap coefficient between the true K-component mixture density and its moment-matched Beta
#  3. between-run disagreement per district and parameter (relative SD of the run means), used to focus the
#     convergence checks on the districts where the runs disagree most
# ============================================================
.beta_mean <- function(a, b) a / (a + b)

precheck_dependence <- function() {
  rows <- list()
  for (llm in LLM_ALL) for (k in names(RUN_PRIORS[[llm]])) {
    pl <- RUN_PRIORS[[llm]][[k]]
    m  <- sapply(c("p", "gamma", "omega"), function(par) sapply(district_names, function(dn) .beta_mean(pl[[dn]][[par]]$a, pl[[dn]][[par]]$b)))
    rows[[length(rows) + 1]] <- tibble(
      LLM = llm, run = as.integer(k),
      cor_p_gamma = cor(m[, "p"], m[, "gamma"]), cor_gamma_omega = cor(m[, "gamma"], m[, "omega"]), cor_p_omega = cor(m[, "p"], m[, "omega"]),
      spearman_p_gamma = cor(m[, "p"], m[, "gamma"], method = "spearman"), spearman_gamma_omega = cor(m[, "gamma"], m[, "omega"], method = "spearman"),
      spearman_p_omega = cor(m[, "p"], m[, "omega"], method = "spearman"))
  }
  out <- bind_rows(rows)
  write_csv(out, file.path(MIX_DIR, "precheck_within_run_dependence.csv"))
  smry <- out %>% group_by(LLM) %>% summarise(across(starts_with(c("cor_", "spearman_")), list(median = median, min = min, max = max)), .groups = "drop")
  write_csv(smry, file.path(MIX_DIR, "precheck_within_run_dependence_summary.csv"))
  message("Saved: ", MIX_DIR, "/precheck_within_run_dependence.csv (+ _summary)")
  out
}

# Overlap coefficient  int min(f_mix, f_mm) dx  between the equal-weight mixture of Beta(a_k, b_k) and one Beta(a, b).
# Integrated on a grid made of the quantiles of every component, so the narrow high-density regions are resolved.
.overlap_beta_mixture <- function(a_k, b_k, a_mm, b_mm, n_q = 400) {
  pr <- c(1e-6, seq(5e-4, 1 - 5e-4, length.out = n_q), 1 - 1e-6)
  x  <- sort(unique(c(unlist(Map(function(a, b) qbeta(pr, a, b), a_k, b_k)), qbeta(pr, a_mm, b_mm))))
  x  <- x[x > 0 & x < 1]
  f_mix <- Reduce(`+`, Map(function(a, b) dbeta(x, a, b), a_k, b_k)) / length(a_k)
  f_mm  <- dbeta(x, a_mm, b_mm)
  fmin  <- pmin(f_mix, f_mm)
  mass  <- function(f) sum(diff(x) * (head(f, -1) + tail(f, -1)) / 2)
  c(overlap = mass(fmin),                                   # masses below should be ~1 (accuracy of the integration)
    mass_mix = mass(f_mix), mass_mm = mass(f_mm))
}

precheck_overlap <- function() {
  rows <- list()
  for (llm in LLM_ALL) {
    rp <- unname(RUN_PRIORS[[llm]]); mm <- pool_priors_moment_match(rp)
    for (dn in district_names) for (par in c("p", "gamma", "omega")) {
      a_k <- vapply(rp, function(pl) pl[[dn]][[par]]$a, numeric(1)); b_k <- vapply(rp, function(pl) pl[[dn]][[par]]$b, numeric(1))
      ov  <- .overlap_beta_mixture(a_k, b_k, mm[[dn]][[par]]$a, mm[[dn]][[par]]$b)
      m_k <- .beta_mean(a_k, b_k)
      rows[[length(rows) + 1]] <- tibble(
        LLM = llm, District = dn, Parameter = par, overlap = ov[["overlap"]], total_variation = 1 - ov[["overlap"]],
        mass_mix = ov[["mass_mix"]], mass_mm = ov[["mass_mm"]],
        mm_a = mm[[dn]][[par]]$a, mm_b = mm[[dn]][[par]]$b, mm_mean = .beta_mean(mm[[dn]][[par]]$a, mm[[dn]][[par]]$b),
        n_components_a_below_1 = sum(a_k < 1), min_component_a = min(a_k),
        between_run_sd_of_means = sd(m_k), between_run_rel_sd = sd(m_k) / mean(m_k))
    }
  }
  out <- bind_rows(rows)
  write_csv(out, file.path(MIX_DIR, "precheck_mixture_vs_moment_match_overlap.csv"))
  smry <- out %>% group_by(LLM, Parameter) %>%
    summarise(median_overlap = median(overlap), min_overlap = min(overlap), share_below_0.90 = mean(overlap < 0.90),
              share_below_0.95 = mean(overlap < 0.95), .groups = "drop")
  write_csv(smry, file.path(MIX_DIR, "precheck_mixture_vs_moment_match_overlap_summary.csv"))
  message("Saved: ", MIX_DIR, "/precheck_mixture_vs_moment_match_overlap.csv (+ _summary)")
  out
}

# Between-run disagreement per district and parameter; the districts with the largest relative SD of the run means
# for p or gamma (top 10 per LLM) are flagged `high_disagreement`.
precheck_disagreement <- function(overlap_df) {
  d <- overlap_df %>% select(LLM, District, Parameter, between_run_sd_of_means, between_run_rel_sd) %>%
    group_by(LLM, Parameter) %>% mutate(rank_rel_sd = rank(-between_run_rel_sd, ties.method = "first")) %>% ungroup()
  flag <- d %>% filter(Parameter %in% c("p", "gamma"), rank_rel_sd <= 10) %>% distinct(LLM, District) %>% mutate(high_disagreement = TRUE)
  d <- d %>% left_join(flag, by = c("LLM", "District")) %>% mutate(high_disagreement = !is.na(high_disagreement))
  write_csv(d, file.path(MIX_DIR, "precheck_between_run_disagreement.csv"))
  message("Saved: ", MIX_DIR, "/precheck_between_run_disagreement.csv")
  d
}

PRECHECKS_DONE <- file.exists(file.path(MIX_DIR, "precheck_between_run_disagreement.csv")) &&
                  file.exists(file.path(MIX_DIR, "precheck_mixture_vs_moment_match_overlap.csv"))
if (!PRECHECKS_DONE) {
  precheck_dependence(); overlap_df <- precheck_overlap(); disagree_df <- precheck_disagreement(overlap_df)
} else {
  disagree_df <- read_csv(file.path(MIX_DIR, "precheck_between_run_disagreement.csv"), show_col_types = FALSE)
  message("Pre-checks already in ", MIX_DIR, " (delete them to recompute).")
}


# ============================================================
# 5. FITS: Stan interface, cache, per-model outputs and diagnostics
# ============================================================
# Cache layout and ids as for the existing models. The hierarchical model (comparator) is read from HIER_DIR.
cache_path  <- function(scheme, model_id) {
  if (model_id == HIER_ID) file.path(HIER_DIR, "cache", scheme, paste0(model_id, ".rds"))
  else                     file.path(CACHE_DIR, scheme, paste0(model_id, ".rds"))
}
cache_has   <- function(scheme, model_id) file.exists(cache_path(scheme, model_id))
cache_read  <- function(scheme, model_id) if (cache_has(scheme, model_id)) readRDS(cache_path(scheme, model_id)) else NULL
cache_write <- function(obj, scheme, model_id) {
  stopifnot(model_id != HIER_ID)
  p <- cache_path(scheme, model_id); tmp <- paste0(p, ".tmp"); saveRDS(obj, tmp); file.rename(tmp, p); invisible(p)
}

# Compiled on first use only. Windows Application Control sometimes blocks a freshly compiled DLL at random;
# a retry compiles a new one.
.stan_model_cache <- new.env()
get_stan_model <- function(model_id) {
  for (attempt in 1:4) {
    if (!is.null(.stan_model_cache[["mixture"]])) break
    .stan_model_cache[["mixture"]] <- tryCatch(stan_model(file = "no_pooling_LLM_mixture_priors_monthly.stan"), error = function(e) {
      message("Stan compile/load failed (attempt ", attempt, "): ", substr(conditionMessage(e), 1, 150)); NULL })
  }
  if (is.null(.stan_model_cache[["mixture"]])) stop("Could not compile the mixture Stan model after 4 attempts")
  .stan_model_cache[["mixture"]]
}

.stan_data <- function(model_id, df_tr, starts) {
  spec <- MIX_SPECS[[model_id]]; rp <- unname(spec_run_priors(spec))
  c(list(N = nrow(df_tr), D = D, district_id = df_tr$district_id, y = df_tr$SAM,
         pop_u5_ref = as.vector(pop_u5_ref), start_index = starts, K = length(rp),
         a_p = .prior_matrix(rp, "p", "a"), b_p = .prior_matrix(rp, "p", "b"),
         a_gamma = .prior_matrix(rp, "gamma", "a"), b_gamma = .prior_matrix(rp, "gamma", "b"),
         a_omega = .prior_matrix(rp, "omega", "a"), b_omega = .prior_matrix(rp, "omega", "b"),
         estimate_w = spec$estimate_w, joint = spec$joint))
}

# Rhat, bulk and tail ESS of every district parameter (and of w when it is estimated), from the post-warmup chains
.mcmc_diagnostics <- function(fit, estimate_w) {
  pars <- c("p", "gamma", "omega", "P0", if (estimate_w == 1) "w")
  arr  <- as.array(fit, pars = pars)                           # iterations x chains x parameters
  nm   <- dimnames(arr)$parameters
  tibble(Parameter = nm,
         Rhat      = vapply(seq_along(nm), function(j) rstan::Rhat(arr[, , j]),      numeric(1)),
         Bulk_ESS  = vapply(seq_along(nm), function(j) rstan::ess_bulk(arr[, , j]),  numeric(1)),
         Tail_ESS  = vapply(seq_along(nm), function(j) rstan::ess_tail(arr[, , j]),  numeric(1)))
}

# The ONLY place that calls Stan: sampler settings, HMC diagnostics, draws, Rhat/ESS, trace array, MCMC diagnostics.
.run_stan <- function(model_id, data, seed, label) {
  t0  <- Sys.time()
  fit <- sampling(get_stan_model(model_id), data = data,
                  chains = MCMC$chains, warmup = MCMC$warmup, iter = MCMC$iter,
                  cores = MCMC$cores, seed = seed)
  mins <- round(as.numeric(difftime(Sys.time(), t0, units = "mins")), 2)
  message("Model fit time [", label, "]: ", mins, " minutes")
  hmc <- .report_hmc_diagnostics(fit, label)
  ps  <- summary(fit)$summary
  list(
    post        = rstan::extract(fit),
    rhat_df     = data.frame(Parameter = rownames(ps), Rhat = ps[, "Rhat"], n_eff = ps[, "n_eff"]),
    hmc         = hmc,
    fit_minutes = mins,
    trace_array = as.array(fit, pars = c("p[1]", "gamma[1]", "omega[1]", "beta0[1]")),
    diag        = .mcmc_diagnostics(fit, data$estimate_w)
  )
}

.summarise_full_fit <- function(model_id, st) {
  post   <- st$post
  is_base <- FALSE     # (the kappa columns exist only in the hierarchical model)

  # --- burden per district x timepoint (median and 95% interval) ---
  b <- as.matrix(post$burden)                                  # draws x N
  burden_summary <- tibble(
    District      = district_names[df_model$district_id],
    Timepoint     = df_model$Timepoint,
    burden_median = apply(b, 2, median),
    burden_lower  = apply(b, 2, quantile, 0.025),
    burden_upper  = apply(b, 2, quantile, 0.975)
  ) %>% arrange(District, Timepoint)

  # E[burden] per district: posterior mean averaged over the district's timepoints
  burden_expected <- tibble(
    District        = district_names,
    burden_expected = as.numeric(tapply(colMeans(b), df_model$district_id, mean))
  )
  # draws of the district-mean burden (for run-to-run comparison and plots)
  burden_dm <- sweep(t(rowsum(t(b), df_model$district_id)), 2,
                     as.numeric(table(df_model$district_id)), "/")

  # --- per-district medians (sd_pred varies by observation; median across a district's obs) ---
  sd_pred_median_by_obs <- apply(as.matrix(post$sd_pred), 2, median)
  alpha_equiv <- tapply(sd_pred_median_by_obs, df_model$district_id, median)

  param_medians <- tibble(
    District    = district_names,
    beta0       = apply(post$beta0,           2, median),
    beta1       = apply(post$beta1,           2, median),
    p           = apply(post$p,               2, median),
    gamma       = apply(post$gamma,           2, median),
    gamma_count = apply(post$gamma_count_out, 2, median),
    omega       = apply(post$omega,           2, median),   # fraction
    omega_count = apply(post$omega_count_out, 2, median),   # in counts
    alpha_equiv = as.numeric(alpha_equiv)
  )
  if (is_base) {
    param_medians <- param_medians %>% mutate(
      kappa_p = median(post$kappa_p), kappa_gamma = median(post$kappa_gamma),
      kappa_omega = median(post$kappa_omega))
  }

  wide <- function(col, prefix) {
    burden_summary %>% select(District, Timepoint, all_of(col)) %>%
      pivot_wider(names_from = Timepoint, values_from = all_of(col), names_prefix = prefix)
  }
  final_summary <- param_medians %>%
    left_join(wide("burden_median", "burden_"),    by = "District") %>%
    left_join(wide("burden_lower",  "burden_lo_"), by = "District") %>%
    left_join(wide("burden_upper",  "burden_hi_"), by = "District")

  # --- posterior mean / median / 95% CI per district; duration = 1/p draw by draw (months) ---
  q <- function(x, pr) apply(x, 2, quantile, pr)
  dur <- 1 / post$p
  param_ci <- tibble(
    Model        = model_label(model_id),
    District     = district_names,
    p_mean       = colMeans(post$p),     p_median     = apply(post$p, 2, median),
    p_lower      = q(post$p, 0.025),     p_upper      = q(post$p, 0.975),
    dur_mean     = colMeans(dur),        dur_median   = apply(dur, 2, median),
    dur_lower    = q(dur, 0.025),        dur_upper    = q(dur, 0.975),
    gamma_mean   = colMeans(post$gamma), gamma_median = apply(post$gamma, 2, median),
    gamma_lower  = q(post$gamma, 0.025), gamma_upper  = q(post$gamma, 0.975),
    omega_mean   = colMeans(post$omega), omega_median = apply(post$omega, 2, median),
    omega_lower  = q(post$omega, 0.025), omega_upper  = q(post$omega, 0.975)
  )

  # --- fit statistics. NOTE: PSIS-LOO assumes exchangeable observations, which an
  # autoregressive model violates, so ELPD is reported as an exploratory diagnostic;
  # LLO-2 / LLO-3 (leave-future-out) are the valid predictive checks. ---
  loo_obj <- loo::loo(post$log_lik)
  est <- loo_obj$estimates
  fit_stats <- tibble(
    model_id = model_id, Model = model_label(model_id), scheme = "full",
    n_draws = nrow(post$p), fit_minutes = st$fit_minutes,
    divergences = st$hmc$divergent, max_treedepth_hits = st$hmc$max_treedepth,
    low_bfmi_chains = st$hmc$low_bfmi_chains,
    max_rhat = max(st$rhat_df$Rhat, na.rm = TRUE), min_n_eff = min(st$rhat_df$n_eff, na.rm = TRUE),
    elpd_loo = est["elpd_loo", "Estimate"], se_elpd_loo = est["elpd_loo", "SE"],
    p_loo = est["p_loo", "Estimate"], n_pareto_k_bad = sum(loo::pareto_k_values(loo_obj) > 0.7),
    elpd_note = "exploratory only: LOO assumes exchangeable data (not true for this AR model)"
  )

  idx <- seq(1, nrow(post$p), by = THIN_EVERY)
  list(
    model_id = model_id, prior = MIX_MM_PRIOR[[model_id]],   # moment-matched Beta of the same runs (standard format); the mixture itself is in prior_mixture
    prior_mixture = list(runs = MIX_SPECS[[model_id]]$runs, joint = MIX_SPECS[[model_id]]$joint, estimate_w = MIX_SPECS[[model_id]]$estimate_w),
    param_medians = param_medians, burden_summary = burden_summary,
    burden_expected = burden_expected, final_summary = final_summary, param_ci = param_ci,
    draws_thin = list(p = post$p[idx, , drop = FALSE], gamma = post$gamma[idx, , drop = FALSE],
                      omega = post$omega[idx, , drop = FALSE], burden = burden_dm[idx, , drop = FALSE]),
    rhat_df = st$rhat_df, hmc = st$hmc, fit_minutes = st$fit_minutes,
    trace_array = st$trace_array, fit_stats = fit_stats, loo = loo_obj, diag = st$diag,
    w_draws = if (MIX_SPECS[[model_id]]$estimate_w == 1) post$w[idx, , drop = FALSE] else NULL
  )
}


# Fits (if allowed and not cached) and returns the cached full-data result, else NULL.
fit_full_model <- function(model_id) {
  if (model_id == HIER_ID) return(cache_read("full", model_id))   # read-only: fitted in HIER_DIR
  if (!FORCE_REFIT) {
    cached <- cache_read("full", model_id)
    if (!is.null(cached)) return(cached)
  }
  if (!("full" %in% RUN_SCHEMES) || !(model_id %in% models_to_fit())) return(NULL)
  message("\n==== Fitting full-data model: ", model_label(model_id), " ====")
  st  <- .run_stan(model_id, .stan_data(model_id, df_model, district_starts), SEEDS[["full"]],
                   paste0(model_label(model_id), " (full data)"))
  out <- .summarise_full_fit(model_id, st)
  cache_write(out, "full", model_id)
  out
}

# Per-district convergence table of one fit, with the high-disagreement flag of the pre-checks
diag_by_district <- function(diag_df, llm) {
  d <- diag_df %>% filter(grepl("^(p|gamma|omega|P0)\\[", Parameter)) %>%
    mutate(par = sub("\\[.*$", "", Parameter), di = as.integer(sub("^.*\\[([0-9]+)\\]$", "\\1", Parameter)), District = district_names[di])
  dis <- disagree_df %>% filter(LLM == llm) %>% distinct(District, high_disagreement)
  d %>% group_by(District) %>%
    summarise(max_Rhat = max(Rhat, na.rm = TRUE), min_Bulk_ESS = min(Bulk_ESS, na.rm = TRUE), min_Tail_ESS = min(Tail_ESS, na.rm = TRUE),
              worst_parameter = par[which.max(Rhat)], .groups = "drop") %>%
    left_join(dis, by = "District") %>%
    mutate(flag_rhat = max_Rhat > RHAT_FLAG, high_disagreement = coalesce(high_disagreement, FALSE))
}

write_full_outputs <- function(model_id, full) {
  lab  <- model_label(model_id); spec <- MIX_SPECS[[model_id]]
  out_summary <- paste0("outputs/parameter_and_burden_summary_", lab, ".csv")
  write_csv(full$final_summary, out_summary)
  write_csv(full$rhat_df,       file.path(DIAG_DIR, paste0("rhat_and_ess_", lab, ".csv")))
  write_csv(full$fit_stats,     file.path(DIAG_DIR, paste0("fit_stats_", lab, ".csv")))
  dd <- diag_by_district(full$diag, spec$llm)
  write_csv(dd, file.path(MIX_DIR, paste0("diagnostics_by_district_", lab, ".csv")))
  if (!is.null(full$w_draws)) {
    runs <- spec$runs
    w <- tibble(run = runs, w_mean = colMeans(full$w_draws), w_lower90 = apply(full$w_draws, 2, quantile, 0.05),
                w_upper90 = apply(full$w_draws, 2, quantile, 0.95), prior_mean = 1 / length(runs))
    write_csv(w, file.path(MIX_DIR, paste0("mixture_weights_", lab, ".csv")))
  }
  message("Saved: ", out_summary, "; ", DIAG_DIR, "/rhat_and_ess_", lab, ".csv, fit_stats_", lab, ".csv; ",
          MIX_DIR, "/diagnostics_by_district_", lab, ".csv", if (!is.null(full$w_draws)) ", mixture_weights" else "")
}

# One table: convergence of every cached mixture / validation fit (all schemes)
report_mixture_diagnostics <- function() {
  rows <- list()
  for (id in names(MIX_SPECS)) for (scheme in c("full", "llo2", "llo3")) {
    r <- cache_read(scheme, id); if (is.null(r) || is.null(r$diag)) next
    dd <- diag_by_district(r$diag, MIX_SPECS[[id]]$llm)
    rows[[length(rows) + 1]] <- tibble(
      model_id = id, scheme = scheme, fit_minutes = r$fit_minutes, divergences = r$hmc$divergent, max_treedepth_hits = r$hmc$max_treedepth,
      low_bfmi_chains = r$hmc$low_bfmi_chains, max_Rhat = max(dd$max_Rhat), min_Bulk_ESS = min(dd$min_Bulk_ESS), min_Tail_ESS = min(dd$min_Tail_ESS),
      n_districts_Rhat_above_1.01 = sum(dd$flag_rhat), n_flagged_among_high_disagreement = sum(dd$flag_rhat & dd$high_disagreement),
      n_high_disagreement = sum(dd$high_disagreement), any_divergences = r$hmc$divergent > 0)
  }
  if (length(rows) == 0) return(invisible(NULL))
  out <- bind_rows(rows); write_csv(out, file.path(MIX_DIR, paste0("diagnostics_summary", ID_SUFFIX, ".csv")))
  message("Saved: ", MIX_DIR, "/diagnostics_summary", ID_SUFFIX, ".csv (", nrow(out), " fits)"); invisible(out)
}

# ---- Full-data fits ----
FULL <- list()
for (id in ids_to_fit()) {
  full <- fit_full_model(id)
  if (!is.null(full)) { FULL[[id]] <- full; write_full_outputs(id, full) }
}
# every cached mixture / validation fit (also the ones from earlier sessions)
for (id in names(MIX_SPECS)) if (is.null(FULL[[id]])) { f <- cache_read("full", id); if (!is.null(f)) FULL[[id]] <- f }
report_mixture_diagnostics()



# ============================================================
# 6. COMPARISONS AND VALIDATION
# Per-district posterior means and 90% intervals of p, gamma, omega, gamma/p and the district-mean burden
# (y + 12 * gamma * U5 population, averaged over the district's timepoints) of two fits, from the cached thinned draws.
# ============================================================
.post_stats <- function(full) {
  d <- full$draws_thin
  q <- list(p = d$p, gamma = d$gamma, omega = d$omega, gamma_over_p = d$gamma / d$p, burden = d$burden)
  bind_rows(lapply(names(q), function(nm) {
    x <- q[[nm]]
    tibble(Quantity = nm, District = district_names, mean = colMeans(x), sd = apply(x, 2, sd),
           lower90 = apply(x, 2, quantile, 0.05), upper90 = apply(x, 2, quantile, 0.95))
  }))
}

# A = fit under test, B = reference. Returns the by-district table with differences and, where the bulk ESS is cached
# for both fits (p, gamma, omega), the z-score of the difference of the means in Monte Carlo standard errors.
compare_fits <- function(A, B, label_A, label_B) {
  sa <- .post_stats(A); sb <- .post_stats(B)
  m <- inner_join(sa, sb, by = c("Quantity", "District"), suffix = c("_A", "_B")) %>%
    mutate(abs_diff_mean = abs(mean_A - mean_B), rel_diff_mean = abs_diff_mean / abs(mean_B),
           abs_diff_lower90 = abs(lower90_A - lower90_B), rel_diff_lower90 = abs_diff_lower90 / abs(lower90_B),
           abs_diff_upper90 = abs(upper90_A - upper90_B), rel_diff_upper90 = abs_diff_upper90 / abs(upper90_B),
           A = label_A, B = label_B)
  ess_of <- function(f) {                       # bulk ESS (new fits) or n_eff from the Stan summary (fits of the existing script)
    d <- if (!is.null(f$diag)) f$diag %>% transmute(Parameter, ess = Bulk_ESS) else f$rhat_df %>% transmute(Parameter, ess = n_eff)
    d %>% filter(grepl("^(p|gamma|omega)\\[", Parameter)) %>%
      transmute(Quantity = sub("\\[.*$", "", Parameter), District = district_names[as.integer(sub("^.*\\[([0-9]+)\\]$", "\\1", Parameter))], ess = ess)
  }
  m <- m %>% left_join(ess_of(A) %>% rename(ess_A = ess), by = c("Quantity", "District")) %>%
    left_join(ess_of(B) %>% rename(ess_B = ess), by = c("Quantity", "District")) %>%
    mutate(z_mean = (mean_A - mean_B) / sqrt(sd_A^2 / ess_A + sd_B^2 / ess_B),
           diff_in_sd = (mean_A - mean_B) / sd_B)   # difference of the posterior means in posterior SDs of the reference fit B
  m
}

summarise_comparison <- function(m) {
  m %>% group_by(A, B, Quantity) %>%
    summarise(median_abs_diff_mean = median(abs_diff_mean), max_abs_diff_mean = max(abs_diff_mean),
              median_rel_diff_mean = median(rel_diff_mean), max_rel_diff_mean = max(rel_diff_mean),
              median_rel_diff_lower90 = median(rel_diff_lower90), max_rel_diff_lower90 = max(rel_diff_lower90),
              median_rel_diff_upper90 = median(rel_diff_upper90), max_rel_diff_upper90 = max(rel_diff_upper90),
              median_abs_diff_width90 = median(abs((upper90_A - lower90_A) - (upper90_B - lower90_B))),
              n_districts = n(),
              max_abs_diff_in_sd = max(abs(diff_in_sd), na.rm = TRUE), mean_diff_in_sd = mean(diff_in_sd, na.rm = TRUE),
              max_abs_z = if (all(is.na(z_mean))) NA_real_ else max(abs(z_mean), na.rm = TRUE),
              share_abs_z_below_3 = if (all(is.na(z_mean))) NA_real_ else mean(abs(z_mean) < 3, na.rm = TRUE),
              .groups = "drop")
}

# Districts that differ most (by relative difference of the posterior mean), per quantity
top_districts <- function(m, n = 5) {
  m %>% group_by(A, B, Quantity) %>% slice_max(rel_diff_mean, n = n, with_ties = FALSE) %>%
    select(A, B, Quantity, District, mean_A, mean_B, rel_diff_mean, abs_diff_mean) %>% ungroup()
}

run_comparisons <- function() {
  sums <- list(); tops <- list(); by_d <- list()
  add <- function(A, B, la, lb, file_tag = NULL) {
    if (is.null(A) || is.null(B)) return(invisible(NULL))
    m <- compare_fits(A, B, la, lb)
    sums[[length(sums) + 1]] <<- summarise_comparison(m); tops[[length(tops) + 1]] <<- top_districts(m)
    if (!is.null(file_tag)) write_csv(m, file.path(MIX_DIR, paste0("compare_by_district_", file_tag, ".csv")))
  }
  # validation 1 and 2: the K = 1 and the identical-components models must reproduce the single-Beta run-1 fit
  ref <- cache_read("full", "chatgpt_run01")
  for (cid in c("chatgpt_mixture_check_K1_run01", "chatgpt_mixture_check_identical_run01")) {
    cid <- paste0(cid, ID_SUFFIX); add(cache_read("full", cid), ref, cid, "chatgpt_run01", cid)
  }
  # validation 3: every mixture fit vs the existing moment-matched <llm>_pooled fit
  for (llm in LLM_ALL) for (v in names(VARIANTS)) {
    id <- mixture_id(llm, v); add(cache_read("full", id), cache_read("full", paste0(llm, "_pooled")), id, paste0(llm, "_pooled"), paste0(id, "_vs_pooled"))
  }
  # joint vs separate, estimated vs fixed weights
  for (llm in LLM_ALL) {
    add(cache_read("full", mixture_id(llm, "joint_fixedw")), cache_read("full", mixture_id(llm, "sep_fixedw")), mixture_id(llm, "joint_fixedw"), mixture_id(llm, "sep_fixedw"))
    add(cache_read("full", mixture_id(llm, "joint_estw")),   cache_read("full", mixture_id(llm, "joint_fixedw")), mixture_id(llm, "joint_estw"), mixture_id(llm, "joint_fixedw"))
  }
  if (length(sums) == 0) { message("Comparisons: no pair of cached fits yet."); return(invisible(NULL)) }
  write_csv(bind_rows(sums), file.path(MIX_DIR, paste0("compare_summary", ID_SUFFIX, ".csv")))
  write_csv(bind_rows(tops), file.path(MIX_DIR, paste0("compare_top_districts", ID_SUFFIX, ".csv")))
  message("Saved: ", MIX_DIR, "/compare_summary", ID_SUFFIX, ".csv and compare_top_districts", ID_SUFFIX, ".csv")
  invisible(bind_rows(sums))
}
comparison_summary <- run_comparisons()


MODEL_COLORS  <- character(0)    # report_cv_relative_scores() then uses default colours
MODEL_DISPLAY <- character(0)

.score_cv_preds <- function(model_label_, preds, df_test, points) {
  med <- apply(preds, 2, median)
  idx_map <- df_test %>%
    mutate(Index = row_number()) %>%
    group_by(district_id) %>%
    summarise(District = first(District), idx = list(sort(Index)), .groups = "drop")
  n_dist <- nrow(idx_map)
  bind_rows(lapply(seq_along(points), function(k) {
    idx_k <- vapply(idx_map$idx, function(v) as.integer(v[k]), integer(1))
    tibble(
      Model     = model_label_,
      District  = idx_map$District,
      Point     = points[k],
      Predicted = med[idx_k],
      Observed  = df_test$SAM[idx_k],
      CRPS      = sapply(seq_len(n_dist), function(i)
                    scoringRules::crps_sample(y = df_test$SAM[idx_k[i]], dat = preds[, idx_k[i]]))
    ) %>% mutate(AbsError = abs(Predicted - Observed),
                 SqError  = (Predicted - Observed)^2,
                 MAPE     = ifelse(Observed == 0, NA_real_, 100 * abs(Predicted - Observed) / Observed))
  }))
}

# Benchmarks, defined for any scheme: Naive predicts each held-out point by the
# previous observed value; HistAverage by the mean of ALL earlier observed values.
# Districts left out of every CV METRIC (the models are still fitted on them): Rab Dhuure has imputed SAM
# (the same value at every missing time point), which favours the Naive benchmark.
EXCLUDE_DISTRICTS <- c("Rab Dhuure")
.drop_excluded <- function(x) if (length(EXCLUDE_DISTRICTS) && !is.null(x)) x[!x$District %in% EXCLUDE_DISTRICTS, , drop = FALSE] else x

.benchmark_rows <- function(scheme, which = c("Naive", "HistAverage")) {
  sp <- CV_SPECS[[scheme]]
  n0 <- sp$n_train
  m  <- length(sp$points)
  bind_rows(lapply(seq_len(D), function(d) {
    y   <- df_model$SAM[df_model$district_id == d]
    obs <- y[n0 + seq_len(m)]
    bind_rows(lapply(which, function(w) {
      pred <- if (w == "Naive") y[n0 + seq_len(m) - 1] else sapply(seq_len(m), function(k) mean(y[seq_len(n0 + k - 1)]))
      tibble(Model = w, District = district_names[d], Point = sp$points,
             Predicted = pred, Observed = obs,
             AbsError = abs(pred - obs), SqError = (pred - obs)^2,
             MAPE = ifelse(obs == 0, NA_real_, 100 * abs(pred - obs) / obs), CRPS = NA_real_)
    }))
  })) %>% .drop_excluded()
}

# ---- Relative and population-weighted CV scores ----
# Raw RMSE averages districts whose caseloads differ by orders of magnitude, so
# the districts with the largest SAM counts dominate it. Two additions:
#
# 1) Relative RMSE vs a benchmark, per district d:
#        Rel_d = RMSE_d(model) / RMSE_d(benchmark)
#    where RMSE_d is taken over that district's held-out points (a single
#    point gives |error|). Rel_d is scale-free, so it can be averaged across
#    districts; < 1 means the model beats the benchmark there. Benchmarks:
#    "HistAverage"    = HistAverage (expanding-window historical mean)
#    "Persistence" = Naive (last observed value)
#    Reported across districts as the plain mean (the headline number), the
#    geometric mean (the usual choice for averaging ratios: a district where
#    the model is 2x worse and one where it is 2x better cancel), the median,
#    the population-weighted mean, and the win rate (share of districts with
#    Rel_d < 1). Districts where the benchmark error is exactly 0 have no
#    defined ratio and are dropped from these averages (n_valid reports how
#    many remain); the geometric mean also drops districts where the model
#    error is exactly 0.
#
# 2) Population-weighted RMSE:
#        wRMSE = sqrt( sum_d w_d * SE_d / sum_d w_d ),   w_d = under-5 reference
#    population. It answers "how large is the error for the typical child"
#    rather than "for the typical district", and is relevant when the output
#    drives allocation across people. Caveat: it is dominated by the few large
#    districts (see eff_n_districts = (sum w)^2 / sum w^2, the number of
#    equally weighted districts it is equivalent to), and the errors are still
#    in counts, so the largest districts matter doubly (bigger counts AND
#    bigger weights). Read it next to the unweighted and relative scores
#    rather than instead of them.
# ============================================================

CV_REL_BENCHMARKS <- c(HistAverage = "HistAverage", Persistence = "Naive")

# all_df: long CV results with columns Model, District, Point, SqError.
# Returns one row per Model x Point (plus Point = "All": the district RMSE
# pools all held-out points) and the per-district ratios used behind it.
compute_cv_relative_scores <- function(all_df, pop_by_district, benchmarks = CV_REL_BENCHMARKS) {
  stopifnot(all(unlist(benchmarks) %in% unique(all_df$Model)))
  pop_lookup <- pop_by_district

  long <- bind_rows(
    all_df %>% mutate(Point = as.character(Point)),
    all_df %>% mutate(Point = "All")
  ) %>%
    mutate(Pop = unname(pop_lookup[District]))
  stopifnot(!anyNA(long$Pop))

  dist_rmse <- long %>%
    group_by(Model, Point, District, Pop) %>%
    summarise(RMSE_d = sqrt(mean(SqError, na.rm = TRUE)),
              MAD_d  = median(AbsError, na.rm = TRUE),        # district's median absolute error over its held-out points
              .groups = "drop")

  base <- long %>%
    group_by(Model, Point) %>%
    summarise(
      n_districts      = n_distinct(District),
      RMSE             = sqrt(mean(SqError, na.rm = TRUE)),
      wRMSE_pop        = sqrt(weighted.mean(SqError, Pop, na.rm = TRUE)),
      wMAPE_pop        = weighted.mean(MAPE, Pop, na.rm = TRUE),   # MAPE (%) weighted by under-5 reference population
      MAD              = median(AbsError, na.rm = TRUE),           # median absolute error (robust to a few huge misses)
      bias_pred_over_obs = sum(Predicted, na.rm = TRUE) / sum(Observed, na.rm = TRUE),  # < 1: predicts too low
      eff_n_districts  = (sum(Pop[!is.na(SqError)])^2) / sum(Pop[!is.na(SqError)]^2),
      .groups = "drop"
    )

  by_district <- list()
  out <- base
  for (b in names(benchmarks)) {
    ref <- dist_rmse %>%
      filter(Model == benchmarks[[b]]) %>%
      select(Point, District, RMSE_ref = RMSE_d, MAD_ref = MAD_d)
    rel <- dist_rmse %>%
      left_join(ref, by = c("Point", "District")) %>%
      mutate(Benchmark = b,
             Rel     = ifelse(!is.na(RMSE_ref) & RMSE_ref > 0, RMSE_d / RMSE_ref, NA_real_),
             Rel_MAD = ifelse(!is.na(MAD_ref) & MAD_ref > 0, MAD_d / MAD_ref, NA_real_))   # same ratio, with MAD_d
    by_district[[b]] <- rel

    wref <- base %>% filter(Model == benchmarks[[b]]) %>% select(Point, wRMSE_ref = wRMSE_pop)
    summ <- rel %>%
      group_by(Model, Point) %>%
      summarise(
        n_valid = sum(!is.na(Rel)),
        mean    = mean(Rel, na.rm = TRUE),
        geomean = { r <- Rel[!is.na(Rel) & Rel > 0]; if (length(r)) exp(mean(log(r))) else NA_real_ },
        median  = median(Rel, na.rm = TRUE),
        popw    = { k <- !is.na(Rel); if (any(k)) sum(Pop[k] * Rel[k]) / sum(Pop[k]) else NA_real_ },
        win     = mean(Rel[!is.na(Rel)] < 1),
        .groups = "drop"
      ) %>%
      left_join(base %>% select(Model, Point, wRMSE_pop), by = c("Model", "Point")) %>%
      left_join(wref, by = "Point") %>%
      mutate(wRMSE_ratio = wRMSE_pop / wRMSE_ref) %>%
      select(Model, Point, n_valid, mean, geomean, median, popw, win, wRMSE_ratio)
    names(summ)[-(1:2)] <- c(
      paste0("n_valid_vs", b),
      paste0("RelRMSE_mean_vs",    b),
      paste0("RelRMSE_geomean_vs", b),
      paste0("RelRMSE_median_vs",  b),
      paste0("RelRMSE_popw_vs",    b),
      paste0("WinRate_vs",         b),
      paste0("wRMSE_ratio_vs",     b)
    )
    # relative MAD: district MAD / benchmark's district MAD, averaged over districts (arithmetic, geometric)
    summ_mad <- rel %>%
      group_by(Model, Point) %>%
      summarise(a = mean(Rel_MAD, na.rm = TRUE),
                g = { r <- Rel_MAD[!is.na(Rel_MAD) & Rel_MAD > 0]; if (length(r)) exp(mean(log(r))) else NA_real_ },
                .groups = "drop")
    names(summ_mad)[-(1:2)] <- c(paste0("RelMAD_mean_vs", b), paste0("RelMAD_geomean_vs", b))
    out <- left_join(out, summ, by = c("Model", "Point")) %>% left_join(summ_mad, by = c("Model", "Point"))
  }

  list(summary = out, by_district = bind_rows(by_district))
}

# Writes <tag>_CV_relative_scores.csv / <tag>_CV_relative_by_district.csv and a
# dot plot of per-district relative RMSE (log scale; left of 1 = model wins).
report_cv_relative_scores <- function(all_df, tag, plot_title) {
  pop_named <- setNames(as.numeric(pop_u5_ref), district_names)
  res <- compute_cv_relative_scores(all_df, pop_named)

  out_summary <- file.path("outputs", paste0(tag, "_CV_relative_scores.csv"))
  out_dist    <- file.path("outputs", paste0(tag, "_CV_relative_by_district.csv"))
  write_csv(res$summary,     out_summary)
  write_csv(res$by_district, out_dist)
  message("Saved: ", out_summary, " and ", out_dist)
  print(as.data.frame(res$summary %>% mutate(across(where(is.numeric), ~ round(.x, 3)))))

  plot_df <- res$by_district %>%
    filter(Point == "All", !is.na(Rel), !(Model %in% unlist(CV_REL_BENCHMARKS))) %>%
    mutate(Benchmark = factor(Benchmark, levels = names(CV_REL_BENCHMARKS),
                              labels = paste0("vs ", c("Historical Average", "Persistence"))),
           District  = factor(District, levels = district_names))
  if (nrow(plot_df) > 0) {
    p_rel <- ggplot(plot_df, aes(x = Rel, y = reorder(District, Pop), color = Model)) +
      geom_vline(xintercept = 1, linetype = "dashed", color = "grey30") +
      geom_point(size = 1.1, alpha = 0.85) +
      facet_wrap(~Benchmark, nrow = 1) +
      scale_x_log10() +
      (if (all(unique(plot_df$Model) %in% names(MODEL_COLORS)))
         scale_color_manual(values = MODEL_COLORS, labels = MODEL_DISPLAY, name = NULL)
       else scale_color_discrete(name = NULL)) +
      labs(title = plot_title,
           subtitle = "Per-district RMSE / benchmark RMSE over all held-out points; < 1 (left of dashed line) = model beats benchmark; districts ordered by U5 population",
           x = "Relative RMSE (log scale)", y = NULL) +
      theme_minimal(base_size = 8) +
      theme(legend.position = "bottom", axis.text.y = element_text(size = 5))
    ggsave(file.path("plots", paste0(tag, "_CV_relative_RMSE_by_district.jpg")),
           p_rel, width = 12, height = 14, dpi = 250)
    message("Saved: plots/", tag, "_CV_relative_RMSE_by_district.jpg")
  }
  invisible(res)
}



# ============================================================
# 7. CROSS-VALIDATION (LLO-2 / LLO-3) of the mixture models
# ============================================================
# Predictive draws and scoring are the existing ones (same samplers, same benchmarks, same relative scores).
# ============================================================
# Predictive sampling for the held-out points (used by the LLO-2 and LLO-3 sections)
# ============================================================
# --- Full posterior predictive sampling (recomputes the theoretical variance per draw) ---
.get_llo2_preds_tv <- function(post, df_tr, df_te, df_full, pop_u5_ref_vec) {
  S    <- nrow(post$beta0)
  N_te <- nrow(df_te)

  idx_map <- df_te %>%
    mutate(Index = row_number()) %>%
    group_by(district_id) %>%
    summarise(idx_8 = min(Index), idx_9 = max(Index), .groups = "drop")

  D_cv <- nrow(idx_map)
  ds   <- idx_map$district_id

  # Pre-compute data lookups once outside the s loop (same value for every draw)
  y7_vec <- sapply(ds, function(d) df_tr   %>% filter(district_id == d) %>% pull(SAM) %>% tail(1))
  y8_vec <- sapply(ds, function(d) df_full %>% filter(district_id == d) %>% slice(8)  %>% pull(SAM))

  # Extract posterior slices for relevant districts: S x D_cv matrices
  beta0   <- post$beta0[, ds]
  beta1   <- post$beta1[, ds]
  c1      <- post$c1[, ds]
  c0      <- post$c0_arrival_bernoulli[, ds]
  omega   <- post$omega[, ds]
  spsq    <- post$sum_pi_sq[, ds]

  # omega_c: S x D_cv  (broadcast pop ref over rows)
  omega_c <- sweep(omega, 2, pop_u5_ref_vec[ds], "*")

  # Broadcast y7 / y8 over S rows
  y7_mat <- matrix(y7_vec, nrow = S, ncol = D_cv, byrow = TRUE)
  y8_mat <- matrix(y8_vec, nrow = S, ncol = D_cv, byrow = TRUE)

  # mu and var: S x D_cv matrices — same formula as before, now vectorised
  mu8  <- beta0 + beta1 * y7_mat
  var8 <- pmax(c1 * y7_mat + c0 + omega_c^2 * spsq, 0)
  mu9  <- beta0 + beta1 * y8_mat
  var9 <- pmax(c1 * y8_mat + c0 + omega_c^2 * spsq, 0)

  # Sample all S x D_cv values at once
  pred8 <- matrix(rnorm(S * D_cv, mu8, sqrt(var8)), S, D_cv)
  pred9 <- matrix(rnorm(S * D_cv, mu9, sqrt(var9)), S, D_cv)

  # Assemble into S x N_te output matrix
  y_pred <- matrix(NA_real_, S, N_te)
  for (i in seq_len(D_cv)) {
    y_pred[, idx_map$idx_8[i]] <- pred8[, i]
    y_pred[, idx_map$idx_9[i]] <- pred9[, i]
  }
  y_pred
}

# --- Full posterior predictive sampling for the 7th/8th/9th held-out points ---
.get_llo3_preds_tv <- function(post, df_tr, df_te, df_full, pop_u5_ref_vec) {
  S    <- nrow(post$beta0)
  N_te <- nrow(df_te)

  idx_map <- df_te %>%
    mutate(Index = row_number()) %>%
    group_by(district_id) %>%
    summarise(
      idx_7 = sort(Index)[1],
      idx_8 = sort(Index)[2],
      idx_9 = sort(Index)[3],
      .groups = "drop"
    )

  D_cv <- nrow(idx_map)
  ds   <- idx_map$district_id

  # Each held-out point conditions on the TRUE previous observed value
  # (not a recursive prediction) -- same convention as LL2O-CV.
  y6_vec <- sapply(ds, function(d) df_tr   %>% filter(district_id == d) %>% pull(SAM) %>% tail(1))
  y7_vec <- sapply(ds, function(d) df_full %>% filter(district_id == d) %>% slice(7)  %>% pull(SAM))
  y8_vec <- sapply(ds, function(d) df_full %>% filter(district_id == d) %>% slice(8)  %>% pull(SAM))

  beta0   <- post$beta0[, ds]
  beta1   <- post$beta1[, ds]
  c1      <- post$c1[, ds]
  c0      <- post$c0_arrival_bernoulli[, ds]
  omega   <- post$omega[, ds]
  spsq    <- post$sum_pi_sq[, ds]

  omega_c <- sweep(omega, 2, pop_u5_ref_vec[ds], "*")

  y6_mat <- matrix(y6_vec, nrow = S, ncol = D_cv, byrow = TRUE)
  y7_mat <- matrix(y7_vec, nrow = S, ncol = D_cv, byrow = TRUE)
  y8_mat <- matrix(y8_vec, nrow = S, ncol = D_cv, byrow = TRUE)

  mu7  <- beta0 + beta1 * y6_mat
  var7 <- pmax(c1 * y6_mat + c0 + omega_c^2 * spsq, 0)
  mu8  <- beta0 + beta1 * y7_mat
  var8 <- pmax(c1 * y7_mat + c0 + omega_c^2 * spsq, 0)
  mu9  <- beta0 + beta1 * y8_mat
  var9 <- pmax(c1 * y8_mat + c0 + omega_c^2 * spsq, 0)

  pred7 <- matrix(rnorm(S * D_cv, mu7, sqrt(var7)), S, D_cv)
  pred8 <- matrix(rnorm(S * D_cv, mu8, sqrt(var8)), S, D_cv)
  pred9 <- matrix(rnorm(S * D_cv, mu9, sqrt(var9)), S, D_cv)

  y_pred <- matrix(NA_real_, S, N_te)
  for (i in seq_len(D_cv)) {
    y_pred[, idx_map$idx_7[i]] <- pred7[, i]
    y_pred[, idx_map$idx_8[i]] <- pred8[, i]
    y_pred[, idx_map$idx_9[i]] <- pred9[, i]
  }
  y_pred
}

# ============================================================
# Cross-validation setup (shared by LLO-2 and LLO-3)
# LLO-2: train on the first 7 observations per district, hold out the 8th and 9th.
# LLO-3: train on the first 6, hold out the 7th, 8th and 9th (robustness check: does
#        "LLM priors win at one step ahead" replicate at a different, non-outlier
#        month, with less training data?). Each held-out point is predicted one step
#        ahead from the TRUE observed value at the preceding timepoint (not
#        recursively), so "N-step-ahead" means how many periods past the training cutoff.
# Benchmarks (computed for every scheme): Naive = previous observed value;
# HistAverage = mean of all earlier observed values.
# ============================================================
CV_SPECS <- list(
  llo2 = list(n_train = 7, points = c("8th", "9th"),        preds_fn = .get_llo2_preds_tv,
              rhat_prefix = "llo2_cv",  plot_nrow = 2, rmse_dims = c(14, 16), title = "Leave-Last-2-Out CV"),
  llo3 = list(n_train = 6, points = c("7th", "8th", "9th"), preds_fn = .get_llo3_preds_tv,
              rhat_prefix = "llo3_cv", plot_nrow = 3, rmse_dims = c(14, 20),
              title = "Leave-Last-3-Out CV (robustness check)")
)
for (.nm in names(CV_SPECS)) {
  .sp <- CV_SPECS[[.nm]]
  .sp$train  <- df_model %>% group_by(district_id) %>% slice_head(n = .sp$n_train) %>% ungroup()
  .sp$test   <- df_model %>% group_by(district_id) %>% slice_tail(n = length(.sp$points)) %>% ungroup()
  .sp$starts <- which(!duplicated(.sp$train$district_id))
  CV_SPECS[[.nm]] <- .sp
}

# Observed trajectory per district (used by the district panels)
df_model_t <- df_model %>%
  group_by(District) %>%
  arrange(Year, Month, .by_group = TRUE) %>%
  mutate(t = row_number()) %>%
  ungroup()

# Metrics of ONE model, written as soon as its CV fit exists:
# outputs/<TAG>_CV_per_model/<model>.csv, one row per held-out point plus "All", with
# RMSE, wRMSE (population-weighted), MAE, MAPE, CRPS, the per-district relative RMSE
# vs Historical Average and Persistence, and the fit health (divergences, Rhat, ESS).
write_cv_model_metrics <- function(scheme, r) {
  tag <- toupper(scheme)
  lab <- model_label(r$model_id)
  dir_out <- file.path("outputs", paste0(tag, "_CV_per_model"))
  dir.create(dir_out, showWarnings = FALSE, recursive = TRUE)
  one <- bind_rows(r$scores, .benchmark_rows(scheme))
  rel <- compute_cv_relative_scores(one, setNames(as.numeric(pop_u5_ref), district_names))$summary %>%
    filter(Model == lab)
  extra <- bind_rows(
    r$scores %>% mutate(Point = as.character(Point)),
    r$scores %>% mutate(Point = "All")
  ) %>%
    group_by(Model, Point) %>%
    summarise(MAE = mean(AbsError, na.rm = TRUE), MAPE_mean = mean(MAPE, na.rm = TRUE),
              CRPS_mean = mean(CRPS, na.rm = TRUE), .groups = "drop")
  out <- left_join(rel, extra, by = c("Model", "Point")) %>%
    mutate(scheme = scheme, divergences = r$hmc$divergent, max_treedepth_hits = r$hmc$max_treedepth,
           low_bfmi_chains = r$hmc$low_bfmi_chains,
           max_rhat = max(r$rhat_df$Rhat, na.rm = TRUE), min_n_eff = min(r$rhat_df$n_eff, na.rm = TRUE),
           fit_minutes = r$fit_minutes) %>%
    relocate(scheme, .after = Model)
  write_csv(out, file.path(dir_out, paste0(lab, ".csv")))
  invisible(out)
}


# Fits (if allowed and not cached) and returns the cached CV result, else NULL.
fit_cv_model <- function(scheme, model_id) {
  if (model_id == HIER_ID) return(cache_read(scheme, model_id))   # read-only: fitted in HIER_DIR
  if (!FORCE_REFIT) {
    cached <- cache_read(scheme, model_id)
    if (!is.null(cached)) return(cached)
  }
  if (!(scheme %in% RUN_SCHEMES) || !(model_id %in% models_to_fit())) return(NULL)
  sp <- CV_SPECS[[scheme]]
  message("\n==== Fitting ", toupper(scheme), "-CV model: ", model_label(model_id), " ====")
  st <- .run_stan(model_id, .stan_data(model_id, sp$train, sp$starts), SEEDS[[scheme]],
                  paste0(model_label(model_id), " (", toupper(scheme), "-CV train)"))
  # reproducible predictive draws: the seed depends on the scheme and the model id
  set.seed(SEEDS[[scheme]] + sum(utf8ToInt(model_id)))
  preds  <- sp$preds_fn(st$post, sp$train, sp$test, df_model, pop_u5_ref)
  scores <- .score_cv_preds(model_label(model_id), preds, sp$test, sp$points)
  out <- list(model_id = model_id, scheme = scheme, scores = scores, rhat_df = st$rhat_df,
              hmc = st$hmc, fit_minutes = st$fit_minutes, diag = st$diag)
  cache_write(out, scheme, model_id)
  out
}

# Score tables of the mixture models next to the existing pooled models, the hierarchical model and the benchmarks.
# Files: outputs/<LLO2|LLO3>_mixture_CV_*.csv (new names; the existing LLO2_CV_* tables are not touched) and
# outputs/<LLO2|LLO3>_CV_per_model/<id>.csv for each mixture model.
run_cv_mixture <- function(scheme) {
  sp <- CV_SPECS[[scheme]]; tag <- paste0(toupper(scheme), "_mixture")
  ids <- names(MIX_SPECS)
  res <- list()
  for (id in ids) {
    r <- fit_cv_model(scheme, id)
    if (is.null(r)) next
    r$scores <- .drop_excluded(r$scores)
    res[[id]] <- r
    write_cv_model_metrics(scheme, r)
  }
  if (length(res) == 0) { message(toupper(scheme), ": no cached or scheduled mixture fits -- skipped."); return(invisible(NULL)) }
  comparators <- c(paste0(unique(vapply(res, function(r) MIX_SPECS[[r$model_id]]$llm, character(1))), "_pooled"), HIER_ID)
  for (id in comparators) { r <- cache_read(scheme, id); if (!is.null(r)) { r$scores <- .drop_excluded(r$scores); res[[id]] <- r } }
  all_df <- bind_rows(bind_rows(lapply(res, function(r) r$scores)), .benchmark_rows(scheme))
  write_csv(all_df, sprintf("outputs/%s_CV_all_models_with_CRPS.csv", tag))
  summary_df <- bind_rows(all_df %>% mutate(Point = as.character(Point)), all_df %>% mutate(Point = "All")) %>%
    group_by(Model, Point) %>%
    summarise(RMSE = sqrt(mean(SqError, na.rm = TRUE)), MAE = mean(AbsError, na.rm = TRUE), MAD = median(AbsError, na.rm = TRUE),
              MAPE_mean = mean(MAPE, na.rm = TRUE), CRPS_mean = mean(CRPS, na.rm = TRUE), .groups = "drop") %>%
    arrange(Point, RMSE)
  print(as.data.frame(summary_df %>% filter(Point == "All")))
  write_csv(summary_df, sprintf("outputs/%s_CV_summary_by_model.csv", tag))
  message("Saved: outputs/", tag, "_CV_all_models_with_CRPS.csv and _CV_summary_by_model.csv")
  report_cv_relative_scores(all_df, tag, paste0(sp$title, " (mixture priors): relative RMSE by district"))
  invisible(all_df)
}
for (scheme in intersect(c("llo2", "llo3"), union(RUN_SCHEMES, c("llo2", "llo3")))) run_cv_mixture(scheme)
report_mixture_diagnostics()
message("Done.")
