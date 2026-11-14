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

# ----------------------------- Data work -----------------------------
df <- read.csv("C:/Users/ecemy/OneDrive - INSEAD/Desktop/Waste prediction/Datasets/Final_Clean_Dataset.csv")

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

setwd("C:/Users/ecemy/OneDrive - INSEAD/Desktop/Waste prediction/Code/Hierarchical model")

dir.create("outputs", showWarnings = FALSE, recursive = TRUE)
dir.create("plots",   showWarnings = FALSE, recursive = TRUE)

# ============================================================
# Shared Beta hyperprior configuration
# TV_MONTHLY_HYPERPRIORS: Beta hyperpriors for p_global/gamma_global/omega_global
# in the monthly hierarchical model (sam_theoretical_variance.stan). Used
# identically by the full-data fit and both CV fits below, and as the basis
# for the missing-district fallback (default_prior_ip) and the prior-posterior
# overlay (tv_base_priors) -- single source of truth so the three can't drift
# apart again.
# TV_DAILY_HYPERPRIORS: Beta hyperpriors for the daily model
# (sam_theoretical_variance_daily.stan); an approximate duration-based
# calibration, not an exact rescaling of the monthly values -- see the
# comments at the daily stan_data_tvd block below and the final review.
# ============================================================
TV_MONTHLY_HYPERPRIORS <- list(
  p     = c(2, 5),
  gamma = c(3, 200),
  omega = c(3, 200)
)
TV_DAILY_HYPERPRIORS <- list(
  p     = c(3, 330),
  gamma = c(3, 6000),
  omega = c(3, 6000)
)

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

# ============================================================
# Data analysis: systematic outlier check per district-timepoint
# For each district, checks whether each of its 9 SAM observations is an
# outlier relative to that SAME district's other 8 observations, using
# three complementary measures (only 9 points/district, so this leans on
# small-sample-appropriate methods rather than a single "textbook" test):
#   - pct_jump_from_prev: simple % change from the previous timepoint
#     (t=1 has no previous point, so this is NA there).
#   - z_score_loo: leave-one-out z-score, (x - mean(other 8)) / sd(other 8)
#     -- "leave-one-out" so the point being checked can't inflate/deflate
#     its own reference distribution.
#   - modified_z_mad: the Iglewicz & Hoya (1993) modified z-score using the
#     district's median and MAD (median absolute deviation) instead of
#     mean/SD -- the standard small-sample-robust outlier diagnostic,
#     because a single extreme point distorts an ordinary SD much more
#     than it distorts a MAD. Conventional threshold: |modified_z_mad| > 3.5.
# All three are written per district x timepoint (not just pre-judged and
# collapsed) so you can inspect, re-threshold, or aggregate -- e.g. the
# mean jump per timepoint across all districts -- however you prefer.
# flagged_outlier below uses the Iglewicz & Hoya rule as the default, but
# isn't the only usable criterion in the CSV.
# ============================================================

district_outlier_check <- df_model %>%
  group_by(District) %>%
  arrange(Year, Month, .by_group = TRUE) %>%
  mutate(
    t = row_number(),
    pct_jump_from_prev = 100 * (SAM - lag(SAM)) / lag(SAM),
    z_score_loo = sapply(seq_along(SAM), function(i) {
      others <- SAM[-i]
      (SAM[i] - mean(others)) / sd(others)
    }),
    modified_z_mad = {
      med     <- median(SAM)
      mad_val <- mad(SAM)  # default constant = 1.4826 already gives the properly-scaled MAD
      if (mad_val == 0) rep(NA_real_, length(SAM)) else (SAM - med) / mad_val
    },
    flagged_outlier = abs(modified_z_mad) > 3.5
  ) %>%
  ungroup() %>%
  select(District, Timepoint, t, SAM, pct_jump_from_prev, z_score_loo, modified_z_mad, flagged_outlier)

write_csv(district_outlier_check, "outputs/district_timepoint_outlier_diagnostics.csv")
message("Saved: outputs/district_timepoint_outlier_diagnostics.csv (",
        sum(district_outlier_check$flagged_outlier, na.rm = TRUE), " district-timepoints flagged)")

# Summary across districts per timepoint -- e.g. "what's the mean jump at t=9 across all 74 districts"
outlier_summary_by_timepoint <- district_outlier_check %>%
  group_by(t, Timepoint) %>%
  summarise(
    n_districts        = n(),
    mean_pct_jump      = mean(pct_jump_from_prev, na.rm = TRUE),
    median_pct_jump    = median(pct_jump_from_prev, na.rm = TRUE),
    mean_abs_z_loo     = mean(abs(z_score_loo), na.rm = TRUE),
    n_flagged_outliers = sum(flagged_outlier, na.rm = TRUE),
    .groups = "drop"
  )
write_csv(outlier_summary_by_timepoint, "outputs/timepoint_outlier_summary.csv")
message("Saved: outputs/timepoint_outlier_summary.csv")
print(outlier_summary_by_timepoint)









# ============================================================
# Theoretical variance model (sam_theoretical_variance.stan)
# omega is a fraction in (0,1) with Beta hierarchical prior,
# symmetric to gamma. omega_hyperprior = Beta shape parameters.
# ============================================================
stan_data_tv <- list(
  N            = N,
  D            = D,
  district_id  = df_model$district_id,
  y            = df_model$SAM,
  pop_u5_ref   = as.vector(pop_u5_ref),
  start_index  = district_starts,
  p_hyperprior     = TV_MONTHLY_HYPERPRIORS$p,
  gamma_hyperprior = TV_MONTHLY_HYPERPRIORS$gamma,
  omega_hyperprior = TV_MONTHLY_HYPERPRIORS$omega
)

t0_fit <- Sys.time()
fit_tv <- stan(
  file   = "sam_theoretical_variance.stan",
  data   = stan_data_tv,
  chains = 4,
  warmup = 5000,
  iter   = 10000,
  cores  = 4,
  seed   = 123
)
message("Model fit time [tv_base]: ", round(as.numeric(difftime(Sys.time(), t0_fit, units = "mins")), 2), " minutes")

posterior_tv <- rstan::extract(fit_tv)
print(fit_tv)
.report_hmc_diagnostics(fit_tv, "tv_base (full data)")

log_lik_tv <- rstan::extract(fit_tv, pars = "log_lik")$log_lik
loo_tv     <- loo::loo(log_lik_tv)
print(loo_tv)

# --- Build burden summary ---
burden_draws_tv    <- as.matrix(posterior_tv$burden)
district_labels_tv <- district_names[df_model$district_id]
burden_long_tv     <- as.data.frame(burden_draws_tv)
burden_long_tv$iteration <- seq_len(nrow(burden_long_tv))
burden_summary_tv <- burden_long_tv |>
  tidyr::pivot_longer(-iteration, names_to = "col", values_to = "burden") |>
  mutate(
    i         = as.integer(sub("^V", "", col)),
    District  = district_labels_tv[i],
    Timepoint = df_model$Timepoint[i]
  ) |>
  dplyr::group_by(District, Timepoint) |>
  dplyr::summarise(
    burden_median = median(burden, na.rm = TRUE),
    burden_lower  = quantile(burden, 0.025, na.rm = TRUE),
    burden_upper  = quantile(burden, 0.975, na.rm = TRUE),
    .groups = "drop"
  )

# --- Per-district median parameters ---
# sd_pred[n] varies by observation (depends on P_prev).
# Median across obs per district gives quantity comparable to alpha[d] in base model.
sd_pred_draws     <- as.matrix(posterior_tv$sd_pred)       # iterations x N
sd_pred_median_by_obs <- apply(sd_pred_draws, 2, median)   # length N
alpha_equiv <- tapply(
  sd_pred_median_by_obs,
  df_model$district_id,
  median
)  # length D

param_medians_tv <- tibble(
  District     = district_names,
  beta0        = apply(posterior_tv$beta0,            2, median),
  beta1        = apply(posterior_tv$beta1,            2, median),
  p            = apply(posterior_tv$p,                2, median),
  gamma        = apply(posterior_tv$gamma,            2, median),
  gamma_count  = apply(posterior_tv$gamma_count_out,  2, median),
  omega        = apply(posterior_tv$omega,            2, median),   # fraction
  omega_count  = apply(posterior_tv$omega_count_out,  2, median),   # in counts
  alpha_equiv  = as.numeric(alpha_equiv),
  kappa_p      = median(posterior_tv$kappa_p),
  kappa_gamma  = median(posterior_tv$kappa_gamma),
  kappa_omega  = median(posterior_tv$kappa_omega)                   # new
)

burden_wide_tv <- burden_summary_tv %>%
  select(District, Timepoint, burden_median) %>%
  pivot_wider(names_from = Timepoint, values_from = burden_median, names_prefix = "burden_")

burden_wide_tv_lo <- burden_summary_tv %>%
  select(District, Timepoint, burden_lower) %>%
  pivot_wider(names_from = Timepoint, values_from = burden_lower, names_prefix = "burden_lo_")

burden_wide_tv_hi <- burden_summary_tv %>%
  select(District, Timepoint, burden_upper) %>%
  pivot_wider(names_from = Timepoint, values_from = burden_upper, names_prefix = "burden_hi_")

final_summary_tv <- param_medians_tv %>%
  left_join(burden_wide_tv,    by = "District") %>%
  left_join(burden_wide_tv_lo, by = "District") %>%
  left_join(burden_wide_tv_hi, by = "District")

write_csv(final_summary_tv, "outputs/parameter_and_burden_summary_theoretical_variance.csv")
# ============================================================
# End of theoretical variance model
# ============================================================




# ============================================================
# Theoretical variance model with LLM-informed priors (from JSON)
# Same SIS-derived variance as sam_theoretical_variance.stan, but the
# district-specific priors for p, gamma, omega are Beta(a, b) distributions
# elicited from an LLM and read DIRECTLY from JSON (no percentile -> Beta
# translation step).
#
# Expected JSON structure (one file per LLM x repeated run of the same prompt):
#   { "districts": [
#       { "district": "...",
#         "reasoning": "...",
#         "p":     {"a": ..., "b": ...},
#         "gamma": {"a": ..., "b": ...},
#         "omega": {"a": ..., "b": ...} }, ... ] }
# Any other top-level keys (e.g. "metadata") are ignored.
#
# File naming: priors_llm/priors_llm_<llm>_<run>.json, e.g.
#   priors_llm_claude_1.json, priors_llm_claude_2.json, priors_llm_gemini_1.json
# Every run found on disk for an LLM is fitted in the prior-sensitivity check
# below; SELECTED_RUN picks the one run per LLM that the rest of the script
# (overlay plots, CV, maps, burden comparison) uses.
# ============================================================

PRIOR_JSON_DIR <- file.path(getwd(), "priors_llm")
LLM_NAMES      <- c("claude", "gemini", "chatgpt")

# Which run of each LLM feeds the downstream analysis. Change a value here once
# the sensitivity check below has been looked at; nothing else needs editing.
SELECTED_RUN <- c(claude = 1L, gemini = 1L, chatgpt = 1L)

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
# as TV_MONTHLY_HYPERPRIORS (the hierarchical model's GLOBAL hyperpriors on
# p_global/gamma_global/omega_global) -- but they are NOT the hierarchical
# model's marginal district priors, and they are only ever used for a
# district entirely absent from the LLM's JSON (see read_priors_from_json()
# above), never substituted for a valid entry.
default_prior_ip <- list(
  p     = list(a = TV_MONTHLY_HYPERPRIORS$p[1],     b = TV_MONTHLY_HYPERPRIORS$p[2]),
  gamma = list(a = TV_MONTHLY_HYPERPRIORS$gamma[1], b = TV_MONTHLY_HYPERPRIORS$gamma[2]),
  omega = list(a = TV_MONTHLY_HYPERPRIORS$omega[1], b = TV_MONTHLY_HYPERPRIORS$omega[2])
)

extract_prior_vec <- function(fitted_priors, param, hyper) {
  sapply(district_names, function(dn) fitted_priors[[dn]][[param]][[hyper]])
}

# Compile the informed-priors Stan program ONCE and reuse the compiled object
# for every LLM x run (both here and in the CV blocks below) via sampling(),
# rather than calling stan(file = ...) repeatedly -- stan() recompiles from
# source on every call. Fewer fresh DLL builds also means fewer chances of
# tripping a Windows Application Control block on a freshly-compiled binary.
sam_informed_priors_model <- stan_model(file = "sam_theoretical_variance_informed_priors.stan")

# ============================================================
# LLM prior sensitivity check
# Each LLM is prompted several times with the same prompt (runs 1, 2, ...).
# The question: do the POSTERIORS differ materially from run to run? Every run
# found in PRIOR_JSON_DIR is fitted on the full data with the same sampler
# settings and seed, then compared within LLM (pairwise across runs) on:
#   - posterior median difference, in posterior SDs (std_median_diff)
#   - 1-Wasserstein distance between the posteriors, in posterior SDs (std_W1)
#   - Kolmogorov-Smirnov statistic between the posterior draws
#   - Jaccard overlap of the 95% credible intervals
#   - Spearman correlation across districts of the posterior medians
#   - the same W1 distance between the PRIORS (prior_std_W1), so you can see
#     whether the data wash out prior differences or amplify them
#   - ELPD (PSIS-LOO) per run and loo_compare differences, plus HMC diagnostics
# for p, gamma, omega and the district-mean 12-month burden (the quantity the
# priors ultimately feed). Outputs: outputs/llm_prior_sensitivity/*.csv and
# plots/llm_prior_sensitivity/*.jpg (posterior densities of all runs overlaid).
# Needs >= 2 runs per LLM to compare; with a single run only the fit is done.
# ============================================================

SENS_OUT_DIR  <- "outputs/llm_prior_sensitivity"
SENS_PLOT_DIR <- "plots/llm_prior_sensitivity"
dir.create(SENS_OUT_DIR,  showWarnings = FALSE, recursive = TRUE)
dir.create(SENS_PLOT_DIR, showWarnings = FALSE, recursive = TRUE)

# Heuristic flag: a pair of runs "disagrees" for a district when the posterior
# medians are more than this many posterior SDs apart.
SENS_MEDIAN_DIFF_FLAG <- 0.5

# Fit one (llm, run). Only the run selected for downstream use keeps the full
# stanfit and all draws; the others keep just the draws needed for the
# comparison (p, gamma, omega, burden), to bound memory.
fit_informed_run <- function(llm, run, keep_full) {
  label <- paste0("tv_", llm, " run ", run)
  message("\n==== Fitting TV LLM-informed model: ", llm, ", run ", run, " ====")

  fp <- read_priors_from_json(
    prior_json_path(llm, run), default_prior_ip,
    summary_csv = file.path(SENS_OUT_DIR, sprintf("prior_summary_%s_run%d.csv", llm, run))
  )
  stan_data <- list(
    N           = N,
    D           = D,
    district_id = df_model$district_id,
    y           = df_model$SAM,
    pop_u5_ref  = as.vector(pop_u5_ref),
    start_index = district_starts,
    a_p         = extract_prior_vec(fp, "p",     "a"),
    b_p         = extract_prior_vec(fp, "p",     "b"),
    a_gamma     = extract_prior_vec(fp, "gamma", "a"),
    b_gamma     = extract_prior_vec(fp, "gamma", "b"),
    a_omega     = extract_prior_vec(fp, "omega", "a"),
    b_omega     = extract_prior_vec(fp, "omega", "b")
  )

  t0_fit <- Sys.time()
  fit <- sampling(sam_informed_priors_model, data = stan_data,
                  chains = 4, warmup = 5000, iter = 10000, cores = 4, seed = 123)
  message("Model fit time [", label, "]: ",
          round(as.numeric(difftime(Sys.time(), t0_fit, units = "mins")), 2), " minutes")

  hmc  <- .report_hmc_diagnostics(fit, paste0(label, " (full data)"))
  post <- rstan::extract(fit)

  param_summary <- summary(fit)$summary
  rhat_df <- data.frame(
    Parameter = rownames(param_summary),
    Rhat      = param_summary[, "Rhat"],
    n_eff     = param_summary[, "n_eff"]
  )
  write_csv(rhat_df, file.path(SENS_OUT_DIR, sprintf("rhat_and_ess_%s_run%d.csv", llm, run)))

  loo_obj <- loo::loo(post$log_lik)

  list(
    prior     = fp,
    stan_data = stan_data,
    fit       = if (keep_full) fit else NULL,
    post      = if (keep_full) post else post[c("p", "gamma", "omega", "burden")],
    rhat_df   = rhat_df,
    hmc       = hmc,
    loo       = loo_obj
  )
}

# ---- Fit every available run of every LLM (re-running resumes: fitted runs are skipped) ----
if (!exists("sens_runs")) sens_runs <- list()
sens_run_ids <- list()
for (llm in LLM_NAMES) {
  runs <- discover_prior_runs(llm)
  if (length(runs) == 0) {
    stop("No prior JSON files found for '", llm, "' in ", PRIOR_JSON_DIR,
         " (expected priors_llm_", llm, "_<run>.json)")
  }
  if (!(SELECTED_RUN[[llm]] %in% runs)) {
    stop("SELECTED_RUN for '", llm, "' is ", SELECTED_RUN[[llm]], " but only runs ",
         paste(runs, collapse = ", "), " exist in ", PRIOR_JSON_DIR)
  }
  sens_run_ids[[llm]] <- runs
  if (is.null(sens_runs[[llm]])) sens_runs[[llm]] <- list()
  for (run in runs) {
    key <- as.character(run)
    if (!is.null(sens_runs[[llm]][[key]])) next
    sens_runs[[llm]][[key]] <- fit_informed_run(llm, run, keep_full = (run == SELECTED_RUN[[llm]]))
  }
}

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

pooled_priors_by_llm <- list()
# Works without the full-data fits in memory: falls back to the JSON files.
if (!exists("sens_run_ids")) sens_run_ids <- setNames(lapply(LLM_NAMES, discover_prior_runs), LLM_NAMES)
if (!exists("sens_runs"))    sens_runs    <- list()
for (llm in LLM_NAMES) {
  run_priors <- lapply(as.character(sens_run_ids[[llm]]), function(k) {
    if (!is.null(sens_runs[[llm]][[k]])) sens_runs[[llm]][[k]]$prior
    else read_priors_from_json(prior_json_path(llm, as.integer(k)), default_prior_ip)
  })
  pooled_priors_by_llm[[llm]] <- pool_priors_moment_match(run_priors)

  summ <- bind_rows(lapply(district_names, function(dn) bind_rows(lapply(c("p", "gamma", "omega"), function(param) {
    a <- pooled_priors_by_llm[[llm]][[dn]][[param]]$a
    b <- pooled_priors_by_llm[[llm]][[dn]][[param]]$b
    tibble(district = dn, parameter = param, source = "pooled_runs", a = a, b = b,
           concentration = a + b, prior_mean = a / (a + b),
           prior_sd = sqrt(a * b / ((a + b)^2 * (a + b + 1))),
           prior_q05 = qbeta(0.05, a, b), prior_q50 = qbeta(0.50, a, b), prior_q95 = qbeta(0.95, a, b))
  }))))
  write_csv(summ, file.path(SENS_OUT_DIR, sprintf("prior_summary_%s_pooled.csv", llm)))
  message("Saved: ", file.path(SENS_OUT_DIR, sprintf("prior_summary_%s_pooled.csv", llm)))
}

# ---- Comparison helpers ----
.SENS_PARAMS <- c("p", "gamma", "omega", "burden")
.SENS_U      <- (seq_len(1000) - 0.5) / 1000   # quantile grid for 1-D Wasserstein distances

# S x D matrix of posterior draws for one comparison quantity. "burden" is the
# per-draw mean of the district's 12-month burden over its observed timepoints.
.sens_param_draws <- function(run_obj, param) {
  if (param != "burden") return(run_obj$post[[param]])
  b    <- run_obj$post$burden                       # S x N
  sums <- t(rowsum(t(b), df_model$district_id))     # S x D
  sweep(sums, 2, as.numeric(table(df_model$district_id)), "/")
}

# 1-D Wasserstein-1 distance between two samples, via a common quantile grid.
.w1_from_samples <- function(x, y) mean(abs(quantile(x, .SENS_U, names = FALSE) -
                                            quantile(y, .SENS_U, names = FALSE)))
# 1-D Wasserstein-1 distance between two Beta priors.
.w1_beta <- function(a1, b1, a2, b2) mean(abs(qbeta(.SENS_U, a1, b1) - qbeta(.SENS_U, a2, b2)))

# Jaccard overlap (intersection / union) of two central credible intervals.
.interval_jaccard <- function(x, y, level = 0.95) {
  pr <- c((1 - level) / 2, 1 - (1 - level) / 2)
  qx <- quantile(x, pr, names = FALSE)
  qy <- quantile(y, pr, names = FALSE)
  inter <- max(0, min(qx[2], qy[2]) - max(qx[1], qy[1]))
  union <- max(qx[2], qy[2]) - min(qx[1], qy[1])
  if (union <= 0) NA_real_ else inter / union
}

# District-level pairwise comparison of two draw sets (vectors) for one quantity.
.compare_draws <- function(x, y) {
  sd_pool <- sqrt((var(x) + var(y)) / 2)
  med_x <- median(x); med_y <- median(y)
  w1    <- .w1_from_samples(x, y)
  tibble(
    median_a         = med_x,
    median_b         = med_y,
    abs_median_diff  = abs(med_x - med_y),
    rel_median_diff  = abs(med_x - med_y) / mean(abs(c(med_x, med_y))),
    std_median_diff  = if (sd_pool > 0) abs(med_x - med_y) / sd_pool else NA_real_,
    W1               = w1,
    std_W1           = if (sd_pool > 0) w1 / sd_pool else NA_real_,
    KS               = unname(suppressWarnings(stats::ks.test(x, y, exact = FALSE)$statistic)),
    cri95_jaccard    = .interval_jaccard(x, y, 0.95)
  )
}

# All pairs of runs x districts x quantities for one LLM.
.sens_pairwise_metrics <- function(llm, runs) {
  keys  <- as.character(runs)
  pairs <- utils::combn(keys, 2, simplify = FALSE)
  rows  <- list()
  for (param in .SENS_PARAMS) {
    draws <- setNames(lapply(keys, function(k) .sens_param_draws(sens_runs[[llm]][[k]], param)), keys)
    for (pr in pairs) {
      for (di in seq_len(D)) {
        m <- .compare_draws(draws[[pr[1]]][, di], draws[[pr[2]]][, di])
        prior_std_w1 <- NA_real_
        if (param != "burden") {
          pa <- sens_runs[[llm]][[pr[1]]]$prior[[district_names[di]]][[param]]
          pb <- sens_runs[[llm]][[pr[2]]]$prior[[district_names[di]]][[param]]
          sd_beta <- function(ab) sqrt(ab$a * ab$b / ((ab$a + ab$b)^2 * (ab$a + ab$b + 1)))
          sd_pool <- sqrt((sd_beta(pa)^2 + sd_beta(pb)^2) / 2)
          prior_std_w1 <- .w1_beta(pa$a, pa$b, pb$a, pb$b) / sd_pool
        }
        rows[[length(rows) + 1]] <- bind_cols(
          tibble(LLM = llm, Quantity = param, District = district_names[di],
                 run_a = as.integer(pr[1]), run_b = as.integer(pr[2])),
          m,
          tibble(prior_std_W1 = prior_std_w1)
        )
      }
    }
  }
  bind_rows(rows)
}

# One row per LLM x quantity x run-pair, summarising across districts.
.sens_summarise_pairs <- function(pairwise_df) {
  pairwise_df %>%
    group_by(LLM, Quantity, run_a, run_b) %>%
    summarise(
      n_districts                = n(),
      median_std_median_diff     = median(std_median_diff, na.rm = TRUE),
      max_std_median_diff        = max(std_median_diff,    na.rm = TRUE),
      share_districts_flagged    = mean(std_median_diff > SENS_MEDIAN_DIFF_FLAG, na.rm = TRUE),
      median_std_W1              = median(std_W1, na.rm = TRUE),
      max_std_W1                 = max(std_W1,    na.rm = TRUE),
      mean_KS                    = mean(KS, na.rm = TRUE),
      mean_cri95_jaccard         = mean(cri95_jaccard, na.rm = TRUE),
      median_rel_median_diff     = median(rel_median_diff, na.rm = TRUE),
      spearman_medians           = suppressWarnings(stats::cor(median_a, median_b, method = "spearman")),
      median_prior_std_W1        = median(prior_std_W1, na.rm = TRUE),
      .groups = "drop"
    )
}

# One row per LLM x run: fit health + predictive fit (ELPD, with loo_compare diffs).
.sens_fit_table <- function(llm, runs) {
  keys <- as.character(runs)
  tab <- bind_rows(lapply(keys, function(k) {
    ro <- sens_runs[[llm]][[k]]
    est <- ro$loo$estimates
    tibble(
      LLM = llm, run = as.integer(k), selected_for_downstream = (as.integer(k) == SELECTED_RUN[[llm]]),
      elpd_loo       = est["elpd_loo", "Estimate"],
      se_elpd_loo    = est["elpd_loo", "SE"],
      p_loo          = est["p_loo", "Estimate"],
      n_pareto_k_bad = sum(loo::pareto_k_values(ro$loo) > 0.7),
      divergences    = ro$hmc$divergent,
      max_treedepth_hits = ro$hmc$max_treedepth,
      max_rhat       = max(ro$rhat_df$Rhat, na.rm = TRUE),
      min_n_eff      = min(ro$rhat_df$n_eff, na.rm = TRUE)
    )
  }))
  if (length(keys) >= 2) {
    cmp <- loo::loo_compare(setNames(lapply(keys, function(k) sens_runs[[llm]][[k]]$loo), keys))
    cmp_df <- tibble(run = as.integer(rownames(cmp)),
                     elpd_diff_vs_best = cmp[, "elpd_diff"],
                     se_diff           = cmp[, "se_diff"])
    tab <- left_join(tab, cmp_df, by = "run")
  }
  tab
}

# Prior (dashed) and posterior (solid) densities of every run overlaid, one
# panel per district. "burden" has no prior of its own, so it is posterior-only.
# Each panel's x-range spans the posteriors and the central 98% of every run's
# prior, so the priors are not clipped; where a prior is much wider than the
# posteriors, the posterior peaks dominate the panel's y-axis.
plot_prior_sensitivity <- function(llm, runs, param, x_n = 300) {
  keys <- as.character(runs)
  log_scale <- param %in% c("gamma", "omega")
  has_prior <- param != "burden"
  draws <- setNames(lapply(keys, function(k) .sens_param_draws(sens_runs[[llm]][[k]], param)), keys)

  dens_rows <- list()
  for (di in seq_len(D)) {
    xs <- lapply(keys, function(k) {
      v <- draws[[k]][, di]
      if (log_scale) log(v[v > 0]) else v
    })
    rng <- quantile(unlist(xs), c(0.001, 0.999), names = FALSE)

    prior_ab <- NULL
    if (has_prior) {
      prior_ab <- lapply(keys, function(k) sens_runs[[llm]][[k]]$prior[[district_names[di]]][[param]])
      q_prior  <- unlist(lapply(prior_ab, function(ab) qbeta(c(0.01, 0.99), ab$a, ab$b)))
      if (log_scale) q_prior <- log(q_prior[q_prior > 0])
      rng <- range(rng, q_prior)
    }
    x_grid <- seq(rng[1], rng[2], length.out = x_n)

    for (i in seq_along(keys)) {
      kd <- density(xs[[i]], from = rng[1], to = rng[2], n = x_n)
      dens_rows[[length(dens_rows) + 1]] <- tibble(
        District = district_names[di], Run = keys[i], Type = "Posterior", x = kd$x, density = kd$y
      )
      if (has_prior) {
        ab <- prior_ab[[i]]
        pd <- if (log_scale) dbeta(exp(x_grid), ab$a, ab$b) * exp(x_grid) else dbeta(x_grid, ab$a, ab$b)
        pd[!is.finite(pd)] <- NA
        dens_rows[[length(dens_rows) + 1]] <- tibble(
          District = district_names[di], Run = keys[i], Type = "Prior", x = x_grid, density = pd
        )
      }
    }
  }
  plot_df <- bind_rows(dens_rows) %>%
    filter(!is.na(density)) %>%
    mutate(District = factor(District, levels = district_names),
           Run      = factor(Run, levels = keys),
           Type     = factor(Type, levels = c("Prior", "Posterior")))

  x_lab <- if (log_scale) paste0("log(", param, ")") else
           if (param == "burden") "district-mean 12-month burden" else param

  p_out <- ggplot(plot_df, aes(x = x, y = density, color = Run, linetype = Type,
                               group = interaction(Run, Type))) +
    geom_line(linewidth = 0.5, alpha = 0.9) +
    facet_wrap(~District, scales = "free", ncol = 9) +
    scale_color_viridis_d(end = 0.9, name = "Prompt run") +
    scale_linetype_manual(values = c(Prior = "dashed", Posterior = "solid")) +
    labs(
      title    = paste0("Prior vs posterior of ", x_lab, " across repeated ", llm, " prompt runs"),
      subtitle = paste0("Dashed = prior, solid = posterior, one color per run; panels use their own axes",
                        if (!has_prior) " (burden has no prior of its own)" else ""),
      x = x_lab, y = "Density"
    ) +
    theme_minimal(base_size = 7) +
    theme(
      strip.text       = element_text(size = 5),
      axis.text        = element_text(size = 4),
      axis.title       = element_text(size = 6),
      legend.position  = "bottom",
      panel.grid.minor = element_blank()
    )

  out_path <- file.path(SENS_PLOT_DIR, sprintf("posterior_sensitivity_%s_%s.jpg", llm, param))
  ggsave(out_path, p_out, width = 22, height = 20, dpi = 250)
  message("Saved: ", out_path)
  invisible(p_out)
}

# ---- Fit health / ELPD table (always, even with a single run per LLM) ----
sens_fit_df <- bind_rows(lapply(LLM_NAMES, function(llm) .sens_fit_table(llm, sens_run_ids[[llm]])))
write_csv(sens_fit_df, file.path(SENS_OUT_DIR, "sensitivity_fit_summary.csv"))
message("Saved: ", file.path(SENS_OUT_DIR, "sensitivity_fit_summary.csv"))
print(as.data.frame(sens_fit_df))

# ---- Pairwise posterior comparison across runs (needs >= 2 runs for an LLM) ----
llms_to_compare <- LLM_NAMES[vapply(LLM_NAMES, function(l) length(sens_run_ids[[l]]) >= 2, logical(1))]
if (length(llms_to_compare) == 0) {
  message("Prior sensitivity: only one run per LLM found in ", PRIOR_JSON_DIR,
          " -- add more priors_llm_<llm>_<run>.json files to compare runs.")
} else {
  sens_pairwise_df <- bind_rows(lapply(llms_to_compare, function(llm) {
    message("Comparing posteriors across runs for: ", llm)
    .sens_pairwise_metrics(llm, sens_run_ids[[llm]])
  }))
  write_csv(sens_pairwise_df, file.path(SENS_OUT_DIR, "sensitivity_pairwise_by_district.csv"))

  sens_summary_df <- .sens_summarise_pairs(sens_pairwise_df)
  write_csv(sens_summary_df, file.path(SENS_OUT_DIR, "sensitivity_summary_by_llm.csv"))
  message("Saved: sensitivity_pairwise_by_district.csv, sensitivity_summary_by_llm.csv in ", SENS_OUT_DIR)
  print(as.data.frame(sens_summary_df))

  for (llm in llms_to_compare) {
    for (param in .SENS_PARAMS) plot_prior_sensitivity(llm, sens_run_ids[[llm]], param)
  }

  # Distribution of the district-level disagreement metrics, per LLM and quantity.
  p_sens_summary <- sens_pairwise_df %>%
    select(LLM, Quantity, std_median_diff, std_W1, cri95_jaccard) %>%
    tidyr::pivot_longer(c(std_median_diff, std_W1, cri95_jaccard),
                        names_to = "Metric", values_to = "Value") %>%
    mutate(Metric = factor(Metric,
                           levels = c("std_median_diff", "std_W1", "cri95_jaccard"),
                           labels = c("|Δ median| / posterior SD", "Wasserstein-1 / posterior SD",
                                      "95% CrI Jaccard overlap (1 = identical)")),
           Quantity = factor(Quantity, levels = .SENS_PARAMS)) %>%
    ggplot(aes(x = Quantity, y = Value, fill = LLM)) +
    geom_boxplot(outlier.size = 0.6, linewidth = 0.3) +
    facet_wrap(~Metric, scales = "free_y", nrow = 1) +
    labs(title = "Posterior sensitivity to repeated LLM prompt runs (districts x run pairs)",
         x = NULL, y = NULL) +
    theme_minimal(base_size = 10) +
    theme(legend.position = "bottom")
  ggsave(file.path(SENS_PLOT_DIR, "sensitivity_metric_distributions.jpg"),
         p_sens_summary, width = 14, height = 5, dpi = 250)
  message("Saved: ", file.path(SENS_PLOT_DIR, "sensitivity_metric_distributions.jpg"))
}

# ============================================================
# Selected run per LLM -> the TV LLM-informed models used downstream
# (overlay plots, CV, maps, burden comparison all read these objects).
# ============================================================
fitted_priors_by_llm  <- list()
stan_data_by_llm      <- list()
fit_by_llm            <- list()
posterior_by_llm      <- list()
burden_summary_by_llm <- list()
param_medians_by_llm  <- list()

for (llm in LLM_NAMES) {
  sel <- as.character(SELECTED_RUN[[llm]])
  sr  <- sens_runs[[llm]][[sel]]
  message("\n==== TV LLM-informed model for downstream use: ", llm, " (prior run ", sel, ") ====")

  fitted_priors_by_llm[[llm]] <- sr$prior
  stan_data_by_llm[[llm]]     <- sr$stan_data
  fit_by_llm[[llm]]           <- sr$fit
  posterior_by_llm[[llm]]     <- sr$post
  print(fit_by_llm[[llm]])

  write_csv(sr$rhat_df, paste0("outputs/rhat_and_ess_tv_", llm, ".csv"))
  message("Saved: outputs/rhat_and_ess_tv_", llm, ".csv")

  # --- Burden summary ---
  post <- posterior_by_llm[[llm]]
  burden_draws_llm    <- as.matrix(post$burden)
  district_labels_llm <- district_names[df_model$district_id]
  burden_long_llm     <- as.data.frame(burden_draws_llm)
  burden_long_llm$iteration <- seq_len(nrow(burden_long_llm))
  burden_summary_by_llm[[llm]] <- burden_long_llm |>
    tidyr::pivot_longer(-iteration, names_to = "col", values_to = "burden") |>
    mutate(
      i         = as.integer(sub("^V", "", col)),
      District  = district_labels_llm[i],
      Timepoint = df_model$Timepoint[i]
    ) |>
    dplyr::group_by(District, Timepoint) |>
    dplyr::summarise(
      burden_median = median(burden, na.rm = TRUE),
      burden_lower  = quantile(burden, 0.025, na.rm = TRUE),
      burden_upper  = quantile(burden, 0.975, na.rm = TRUE),
      .groups = "drop"
    )

  sd_pred_draws_llm  <- as.matrix(post$sd_pred)
  sd_pred_median_llm <- apply(sd_pred_draws_llm, 2, median)
  alpha_equiv_llm    <- tapply(sd_pred_median_llm, df_model$district_id, median)

  param_medians_by_llm[[llm]] <- tibble(
    District    = district_names,
    beta0       = apply(post$beta0,           2, median),
    beta1       = apply(post$beta1,           2, median),
    p           = apply(post$p,               2, median),
    gamma       = apply(post$gamma,           2, median),
    gamma_count = apply(post$gamma_count_out, 2, median),
    omega       = apply(post$omega,           2, median),
    omega_count = apply(post$omega_count_out, 2, median),
    alpha_equiv = as.numeric(alpha_equiv_llm)
  )

  burden_wide_llm <- burden_summary_by_llm[[llm]] %>%
    select(District, Timepoint, burden_median) %>%
    pivot_wider(names_from = Timepoint, values_from = burden_median, names_prefix = "burden_")
  burden_wide_llm_lo <- burden_summary_by_llm[[llm]] %>%
    select(District, Timepoint, burden_lower) %>%
    pivot_wider(names_from = Timepoint, values_from = burden_lower, names_prefix = "burden_lo_")
  burden_wide_llm_hi <- burden_summary_by_llm[[llm]] %>%
    select(District, Timepoint, burden_upper) %>%
    pivot_wider(names_from = Timepoint, values_from = burden_upper, names_prefix = "burden_hi_")

  final_summary_llm <- param_medians_by_llm[[llm]] %>%
    left_join(burden_wide_llm,    by = "District") %>%
    left_join(burden_wide_llm_lo, by = "District") %>%
    left_join(burden_wide_llm_hi, by = "District")

  out_path_llm <- paste0("outputs/parameter_and_burden_summary_tv_", llm, ".csv")
  write_csv(final_summary_llm, out_path_llm)
  message("Saved: ", out_path_llm)
}

# Canonical model labels/colors/display-names, reused by every downstream
# comparison section (prior-posterior overlay, CV, maps, burden comparison).
# tv_base is neutral grey; each LLM gets a fixed, distinct color.
TV_MODEL_LABELS  <- c("tv_base", paste0("tv_", LLM_NAMES))
TV_MODEL_COLORS  <- setNames(c("grey40", "#2a78d6", "#eb6834", "#1baf7a"), TV_MODEL_LABELS)
TV_MODEL_DISPLAY <- setNames(
  c("TV (Hierarchical)", paste0("TV (", c("Claude", "Gemini", "ChatGPT"), ")")),
  TV_MODEL_LABELS
)
# ============================================================
# End of TV LLM-informed models (Claude, Gemini, ChatGPT)
# ============================================================

# ============================================================
# Prior-Posterior Overlay: TV Base vs TV Informed Priors
# Both models per district, dashed = prior, solid = posterior
# ============================================================

# TV base model hyperpriors (same for all districts — the GLOBAL-level prior
# on p_global/gamma_global/omega_global, NOT the district-level marginal
# prior on p[d]/gamma[d]/omega[d]; see .tv_base_hierarchical_prior_draws()
# below, which simulates the actual district-level marginal from these).
tv_base_priors <- list(
  p     = c(a = TV_MONTHLY_HYPERPRIORS$p[1],     b = TV_MONTHLY_HYPERPRIORS$p[2]),
  gamma = c(a = TV_MONTHLY_HYPERPRIORS$gamma[1], b = TV_MONTHLY_HYPERPRIORS$gamma[2]),
  omega = c(a = TV_MONTHLY_HYPERPRIORS$omega[1], b = TV_MONTHLY_HYPERPRIORS$omega[2])
)

# ---- Simulate the ACTUAL hierarchical marginal prior for a district param ----
# Beta(2,5)/Beta(3,200) etc. above are hyperpriors on the GLOBAL mean only.
# The district-level marginal prior implied by the hierarchical model
# (sam_theoretical_variance.stan) is: global mean ~ Beta hyperprior,
# concentration kappa ~ Student-t(3,0,10) truncated to (0, Inf) (Stan's
# <lower=1e-10> constraint on kappa_p/kappa_gamma/kappa_omega -- since
# Student-t is symmetric about 0, truncating at 0 is equivalent to folding,
# i.e. abs() of an unconstrained draw), then district param ~ Beta(mean*kappa,
# (1-mean)*kappa) conditional on that (mean, kappa) draw. All districts are
# exchangeable under this specification, so the marginal is identical across
# districts and only needs to be simulated once per parameter. Uses a fixed
# seed and restores the caller's RNG stream on exit so it doesn't perturb
# reproducibility of unrelated random draws elsewhere in the script (e.g. the
# CV posterior-predictive rnorm() calls). Never substitutes posterior
# hyperparameter draws -- draws are entirely prior-side.
.tv_base_hierarchical_prior_draws <- function(param_name, n_draws = 20000, seed = 20260101) {
  old_seed <- if (exists(".Random.seed", envir = .GlobalEnv)) get(".Random.seed", envir = .GlobalEnv) else NULL
  on.exit({
    if (!is.null(old_seed)) assign(".Random.seed", old_seed, envir = .GlobalEnv)
    else if (exists(".Random.seed", envir = .GlobalEnv)) rm(".Random.seed", envir = .GlobalEnv)
  })
  set.seed(seed)

  hp <- tv_base_priors[[param_name]]
  global_mean <- rbeta(n_draws, hp["a"], hp["b"])
  kappa       <- pmax(abs(rt(n_draws, df = 3)) * 10, 1e-10)   # folded/truncated-at-0 Student-t(3,0,10)
  a_d <- pmax(global_mean * kappa, 1e-10)
  b_d <- pmax((1 - global_mean) * kappa, 1e-10)
  rbeta(n_draws, a_d, b_d)
}

# TV informed priors (district-specific), one set per LLM
tv_priors_by_llm <- setNames(lapply(LLM_NAMES, function(llm) {
  fp <- fitted_priors_by_llm[[llm]]
  list(
    a_p     = extract_prior_vec(fp, "p",     "a"),
    b_p     = extract_prior_vec(fp, "p",     "b"),
    a_gamma = extract_prior_vec(fp, "gamma", "a"),
    b_gamma = extract_prior_vec(fp, "gamma", "b"),
    a_omega = extract_prior_vec(fp, "omega", "a"),
    b_omega = extract_prior_vec(fp, "omega", "b")
  )
}), LLM_NAMES)

plot_prior_post_tv <- function(param_name,
                               log_transform = FALSE,
                               x_range       = NULL,
                               x_n           = 300,
                               dens_cap_q    = 0.97,
                               output_dir    = "plots") {
  dir.create(output_dir, showWarnings = FALSE)

  # ---- tv_base marginal prior: simulate from the ACTUAL hierarchical prior ----
  # (not the global-mean hyperprior -- see .tv_base_hierarchical_prior_draws()).
  # Exchangeable across districts, so simulated once per parameter and reused
  # for every district panel below.
  tv_base_draws <- .tv_base_hierarchical_prior_draws(param_name)

  # ---- x-range: union of posterior draws AND prior extents (tv_base sim +   ----
  # ---- LLM district Beta quantiles), so the dashed prior curves are never   ----
  # ---- silently clipped to a posterior-only display window.                ----
  post_samp <- c(
    as.vector(posterior_tv[[param_name]]),
    unlist(lapply(LLM_NAMES, function(llm) as.vector(posterior_by_llm[[llm]][[param_name]])))
  )
  llm_prior_extents <- unlist(lapply(LLM_NAMES, function(llm) {
    pr    <- tv_priors_by_llm[[llm]]
    a_vec <- switch(param_name, p = pr$a_p, gamma = pr$a_gamma, omega = pr$a_omega)
    b_vec <- switch(param_name, p = pr$b_p, gamma = pr$b_gamma, omega = pr$b_omega)
    c(qbeta(0.002, a_vec, b_vec), qbeta(0.998, a_vec, b_vec))
  }))
  all_samp <- c(post_samp, tv_base_draws, llm_prior_extents)
  if (log_transform) all_samp <- log(all_samp[all_samp > 0])
  all_samp <- all_samp[is.finite(all_samp)]

  if (is.null(x_range)) {
    x_range <- quantile(all_samp, c(0.002, 0.998), na.rm = TRUE)
  }
  if (!log_transform) x_range[1] <- max(x_range[1], 1e-4)

  x_grid <- seq(x_range[1], x_range[2], length.out = x_n)

  # tv_base prior as a KDE over the simulated hierarchical-prior draws
  # (same treatment as the posterior curves, since it is no longer a single
  # analytic Beta density but a mixture over the kappa/global-mean draws).
  tv_base_prior_x <- if (log_transform) log(tv_base_draws[tv_base_draws > 0]) else tv_base_draws
  tv_base_prior_x <- tv_base_prior_x[is.finite(tv_base_prior_x)]
  tv_base_kd      <- density(tv_base_prior_x, from = x_range[1], to = x_range[2], n = x_n)
  tv_base_prior_tbl <- tibble(x = tv_base_kd$x, density = tv_base_kd$y)

  row_list <- list()

  for (di in seq_len(length(district_names))) {
    dn <- district_names[di]

    for (m in TV_MODEL_LABELS) {

      # ---- Prior density ----
      if (m == "tv_base") {
        row_list[[paste(dn, m, "Prior", sep = "||")]] <- tv_base_prior_tbl %>%
          mutate(Model = m, Type = "Prior", District = dn)
        post_obj <- posterior_tv
      } else {
        llm <- sub("^tv_", "", m)
        pr  <- tv_priors_by_llm[[llm]]
        a_prior <- switch(param_name,
                          p     = pr$a_p[di],
                          gamma = pr$a_gamma[di],
                          omega = pr$a_omega[di])
        b_prior <- switch(param_name,
                          p     = pr$b_p[di],
                          gamma = pr$b_gamma[di],
                          omega = pr$b_omega[di])
        post_obj <- posterior_by_llm[[llm]]

        if (log_transform) {
          x_orig     <- exp(x_grid)
          prior_dens <- dbeta(x_orig, a_prior, b_prior) * x_orig
        } else {
          prior_dens <- dbeta(x_grid, a_prior, b_prior)
        }
        prior_dens[!is.finite(prior_dens)] <- NA

        row_list[[paste(dn, m, "Prior", sep = "||")]] <- tibble(
          x = x_grid, density = prior_dens,
          Model = m, Type = "Prior", District = dn
        )
      }

      # ---- Posterior density (KDE) ----
      samp <- post_obj[[param_name]][, di]
      if (log_transform) samp <- log(samp[samp > 0])
      samp <- samp[is.finite(samp)]
      if (length(samp) >= 20) {
        kd <- density(samp, from = x_range[1], to = x_range[2], n = x_n)
        row_list[[paste(dn, m, "Posterior", sep = "||")]] <- tibble(
          x = kd$x, density = kd$y,
          Model = m, Type = "Posterior", District = dn
        )
      }
    }
  }

  plot_df <- bind_rows(row_list) %>%
    filter(!is.na(density)) %>%
    mutate(
      Model    = factor(Model,    levels = TV_MODEL_LABELS),
      Type     = factor(Type,     levels = c("Prior", "Posterior")),
      District = factor(District, levels = district_names)
    )

  dens_cap <- quantile(plot_df$density, dens_cap_q, na.rm = TRUE)
  plot_df  <- plot_df %>% mutate(density = pmin(density, dens_cap))

  x_lab <- if (log_transform) paste0("log(", param_name, ")") else param_name

  p_out <- ggplot(
    plot_df,
    aes(x        = x,
        y        = density,
        color    = Model,
        linetype = Type,
        group    = interaction(Model, Type))
  ) +
    geom_line(linewidth = 0.5, alpha = 0.9) +
    facet_wrap(~District, scales = "free_y", ncol = 9) +
    scale_color_manual(values = TV_MODEL_COLORS, labels = TV_MODEL_DISPLAY) +
    scale_linetype_manual(values = c(Prior = "dashed", Posterior = "solid")) +
    labs(
      title    = paste0("Prior vs Posterior: ", param_name,
                        if (log_transform) "  [log scale]" else "",
                        "  —  TV Base vs LLM-Informed (Claude/Gemini/ChatGPT)"),
      subtitle = paste0("Dashed = prior  |  Solid = posterior  |  tv_base prior is the simulated ",
                        "district-level marginal (not the p_global/gamma_global/omega_global hyperprior)  |  ",
                        "density capped at the ", round(dens_cap_q * 100), "th percentile across all curves for display"),
      x        = x_lab,
      y        = "Density"
    ) +
    theme_minimal(base_size = 7) +
    theme(
      strip.text       = element_text(size = 5),
      axis.text        = element_text(size = 4),
      axis.title       = element_text(size = 6),
      legend.position  = "bottom",
      legend.text      = element_text(size = 7),
      panel.grid.minor = element_blank()
    )

  suffix   <- if (log_transform) paste0(param_name, "_logscale") else param_name
  out_path <- file.path(output_dir,
                        paste0("prior_posterior_", suffix, "_tv_base_vs_informed.jpg"))
  ggsave(out_path, p_out, width = 22, height = 20, dpi = 250)
  message("Saved: ", out_path)
  invisible(p_out)
}

# ---- Generate plots ----
plot_prior_post_tv("p",     log_transform = FALSE, x_range = c(1e-4, 1))
plot_prior_post_tv("gamma", log_transform = TRUE)
plot_prior_post_tv("omega", log_transform = TRUE)




# Model list used by PPC and parameter CI sections
tv_models <- c(
  list(tv = posterior_tv),
  setNames(lapply(LLM_NAMES, function(llm) posterior_by_llm[[llm]]), paste0("tv_", LLM_NAMES))
)
tv_colors <- setNames(TV_MODEL_COLORS, c("tv", paste0("tv_", LLM_NAMES)))

# --- Parameter CI comparison: TV vs TV + informed priors ---
# Units: p is the unitless monthly departure probability; dur = 1/p (draw by
# draw, not 1/median(p)) is duration in MONTHS; gamma and omega are the
# monthly incidence rate / SD-of-arrivals rate, each as a FRACTION of the
# district's reference U5 population. *_lower/*_upper are the 2.5%/97.5%
# posterior quantiles (a 95% credible interval), per district, per model.
param_ci_tv_rows <- lapply(names(tv_models), function(m) {
  post      <- tv_models[[m]]
  p_draws   <- post$p
  dur_draws <- 1 / p_draws
  g_draws   <- post$gamma
  o_draws   <- post$omega
  tibble(
    Model        = m,
    District     = district_names,
    p_median     = apply(p_draws,   2, median),
    p_lower      = apply(p_draws,   2, quantile, 0.025),
    p_upper      = apply(p_draws,   2, quantile, 0.975),
    dur_median   = apply(dur_draws, 2, median),
    dur_lower    = apply(dur_draws, 2, quantile, 0.025),
    dur_upper    = apply(dur_draws, 2, quantile, 0.975),
    gamma_median = apply(g_draws,   2, median),
    gamma_lower  = apply(g_draws,   2, quantile, 0.025),
    gamma_upper  = apply(g_draws,   2, quantile, 0.975),
    omega_median = apply(o_draws,   2, median),
    omega_lower  = apply(o_draws,   2, quantile, 0.025),
    omega_upper  = apply(o_draws,   2, quantile, 0.975)
  )
})
param_ci_tv <- bind_rows(param_ci_tv_rows)
write_csv(param_ci_tv, "outputs/parameter_CI_tv_vs_tv_informed.csv")
message("Saved: outputs/parameter_CI_tv_vs_tv_informed.csv")

# ------------------------------------------------------------------------
# ---- Leave-Last-2-Out Cross-Validation: TV base vs TV LLM-informed ----
# ------------------------------------------------------------------------
df_train_cv_tv        <- df_model %>% group_by(district_id) %>% slice_head(n = 7) %>% ungroup()
df_test_cv_tv         <- df_model %>% group_by(district_id) %>% slice_tail(n = 2) %>% ungroup()
district_starts_cv_tv <- which(!duplicated(df_train_cv_tv$district_id))

message("Fitting TV base model on training split for LL2O-CV...")
t0_fit <- Sys.time()
fit_cv_tv_base <- stan(
  "sam_theoretical_variance.stan",
  data = list(
    N            = nrow(df_train_cv_tv),
    D            = D,
    district_id  = df_train_cv_tv$district_id,
    y            = df_train_cv_tv$SAM,
    pop_u5_ref   = as.vector(pop_u5_ref),
    start_index  = district_starts_cv_tv,
    p_hyperprior     = TV_MONTHLY_HYPERPRIORS$p,
    gamma_hyperprior = TV_MONTHLY_HYPERPRIORS$gamma,
    omega_hyperprior = TV_MONTHLY_HYPERPRIORS$omega
  ),
  chains = 4, warmup = 5000, iter = 10000, cores = 4, seed = 456
)
message("CV model fit time [tv_base]: ", round(as.numeric(difftime(Sys.time(), t0_fit, units = "mins")), 2), " minutes")
post_cv_tv_base <- rstan::extract(fit_cv_tv_base)
print(fit_cv_tv_base)
.report_hmc_diagnostics(fit_cv_tv_base, "tv_base (LL2O-CV train)")
param_summary_cv_tv_base <- summary(fit_cv_tv_base)$summary
rhat_df_cv_tv_base <- data.frame(
  Parameter = rownames(param_summary_cv_tv_base),
  Rhat      = param_summary_cv_tv_base[, "Rhat"],
  n_eff     = param_summary_cv_tv_base[, "n_eff"]
)
write_csv(rhat_df_cv_tv_base, "outputs/rhat_and_ess_llto_cv_tv_base.csv")
message("Saved: outputs/rhat_and_ess_llto_cv_tv_base.csv")

if (!exists("fit_cv_by_llm"))  fit_cv_by_llm  <- list()
if (!exists("post_cv_by_llm")) post_cv_by_llm <- list()
if (!exists("post_cv_runs"))   post_cv_runs   <- list()

# LL2O-CV fits for EVERY prior run of each LLM (priors_llm_<llm>_<run>.json),
# not only SELECTED_RUN, so the out-of-sample comparison covers all runs, plus
# the pooled prior per LLM.
# - The selected run keeps the standard objects (fit_cv_by_llm /
#   post_cv_by_llm) that the plots and tables below use, as before.
# - Every run is also stored in post_cv_runs[["<llm>_run<k>"]], used by the
#   all-runs comparison at the end of the script. Non-selected runs keep only
#   the draws the prediction step needs, to bound memory.
# - Re-running resumes: a run already in post_cv_runs is not refitted.
CV_PRED_PARS <- c("beta0", "beta1", "c1", "c0_arrival_bernoulli", "omega", "sum_pi_sq")

# Besides the individual runs, each LLM's POOLED prior (pooled_priors_by_llm,
# see "Pooled prior per LLM" above) is fitted too, stored as "<llm>_pooled".
# Which fits to run. Default: all runs plus "pooled". Set to "pooled" to fit
# ONLY the pooled priors (e.g. when the single-run CV results already exist on
# disk): the all-runs scoring block at the end of the file keeps the saved
# single-run rows from outputs/LLO2_CV_all_runs_with_CRPS.csv.
CV_TAGS_TO_FIT <- c("1", "2", "3", "pooled")

for (llm in LLM_NAMES) {
  for (run in c(as.character(sens_run_ids[[llm]]), "pooled")) {
    key    <- if (run == "pooled") paste0(llm, "_pooled") else paste0(llm, "_run", run)
    is_sel <- (run == as.character(SELECTED_RUN[[llm]]))

    if (!is.null(post_cv_runs[[key]])) {
      if (is_sel) post_cv_by_llm[[llm]] <- post_cv_runs[[key]]
      next
    }
    if (!(run %in% CV_TAGS_TO_FIT)) next

    message("Fitting TV LLM-informed model (", llm, ", ", if (run == "pooled") "pooled prior" else paste("run", run),
            ") on training split for LL2O-CV...")
    fp <- if (run == "pooled") {
      pooled_priors_by_llm[[llm]]
    } else if (!is.null(sens_runs[[llm]][[run]])) {
      sens_runs[[llm]][[run]]$prior
    } else {
      read_priors_from_json(prior_json_path(llm, as.integer(run)), default_prior_ip)
    }

    t0_fit <- Sys.time()
    fit_cv <- sampling(
      sam_informed_priors_model,
      data = list(
        N           = nrow(df_train_cv_tv),
        D           = D,
        district_id = df_train_cv_tv$district_id,
        y           = df_train_cv_tv$SAM,
        pop_u5_ref  = as.vector(pop_u5_ref),
        start_index = district_starts_cv_tv,
        a_p         = extract_prior_vec(fp, "p",     "a"),
        b_p         = extract_prior_vec(fp, "p",     "b"),
        a_gamma     = extract_prior_vec(fp, "gamma", "a"),
        b_gamma     = extract_prior_vec(fp, "gamma", "b"),
        a_omega     = extract_prior_vec(fp, "omega", "a"),
        b_omega     = extract_prior_vec(fp, "omega", "b")
      ),
      chains = 4, warmup = 5000, iter = 10000, cores = 4, seed = 456
    )
    message("CV model fit time [", key, "]: ",
            round(as.numeric(difftime(Sys.time(), t0_fit, units = "mins")), 2), " minutes")
    .report_hmc_diagnostics(fit_cv, paste0("tv_", key, " (LL2O-CV train)"))

    param_summary_cv <- summary(fit_cv)$summary
    rhat_df_cv <- data.frame(
      Parameter = rownames(param_summary_cv),
      Rhat      = param_summary_cv[, "Rhat"],
      n_eff     = param_summary_cv[, "n_eff"]
    )
    write_csv(rhat_df_cv, paste0("outputs/rhat_and_ess_llto_cv_tv_", key, ".csv"))
    message("Saved: outputs/rhat_and_ess_llto_cv_tv_", key, ".csv")

    post_full <- rstan::extract(fit_cv)
    if (is_sel) {
      print(fit_cv)
      fit_cv_by_llm[[llm]]  <- fit_cv
      post_cv_by_llm[[llm]] <- post_full
      post_cv_runs[[key]]   <- post_full
      write_csv(rhat_df_cv, paste0("outputs/rhat_and_ess_llto_cv_tv_", llm, ".csv"))
    } else {
      post_cv_runs[[key]] <- post_full[CV_PRED_PARS]
    }
    rm(fit_cv, post_full)
  }
}

# --- Full posterior predictive sampling (recomputes TV variance per draw) ---
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

tv_cv_labels    <- TV_MODEL_LABELS
tv_cv_colors    <- TV_MODEL_COLORS
all_posts_cv_tv <- c(
  list(tv_base = post_cv_tv_base),
  setNames(lapply(LLM_NAMES, function(llm) post_cv_by_llm[[llm]]), paste0("tv_", LLM_NAMES))
)


# --- Run full posterior predictive with RMSE, MAPE, CRPS ---
.llo2_score_rows <- function(m, post_m) {
  message("Computing LL2O-CV predictions for TV model: ", m)
  preds <- .get_llo2_preds_tv(post_m, df_train_cv_tv, df_test_cv_tv, df_model, pop_u5_ref)
  med   <- apply(preds, 2, median)

  idx_map <- df_test_cv_tv %>%
    mutate(Index = row_number()) %>%
    group_by(district_id) %>%
    summarise(District = first(District), idx_8 = min(Index), idx_9 = max(Index), .groups = "drop")

  n_dist <- nrow(idx_map)

  bind_rows(
    tibble(
      Model     = m,
      District  = idx_map$District,
      Point     = "8th",
      Predicted = med[idx_map$idx_8],
      Observed  = df_test_cv_tv$SAM[idx_map$idx_8],
      CRPS      = sapply(seq_len(n_dist), function(i)
                    scoringRules::crps_sample(
                      y   = df_test_cv_tv$SAM[idx_map$idx_8[i]],
                      dat = preds[, idx_map$idx_8[i]]))
    ) %>% mutate(AbsError = abs(Predicted - Observed),
                 SqError  = (Predicted - Observed)^2,
                 MAPE     = ifelse(Observed == 0, NA_real_, 100 * abs(Predicted - Observed) / Observed)),
    tibble(
      Model     = m,
      District  = idx_map$District,
      Point     = "9th",
      Predicted = med[idx_map$idx_9],
      Observed  = df_test_cv_tv$SAM[idx_map$idx_9],
      CRPS      = sapply(seq_len(n_dist), function(i)
                    scoringRules::crps_sample(
                      y   = df_test_cv_tv$SAM[idx_map$idx_9[i]],
                      dat = preds[, idx_map$idx_9[i]]))
    ) %>% mutate(AbsError = abs(Predicted - Observed),
                 SqError  = (Predicted - Observed)^2,
                 MAPE     = ifelse(Observed == 0, NA_real_, 100 * abs(Predicted - Observed) / Observed))
  )
}
llo2_tv_rows <- lapply(tv_cv_labels, function(m) .llo2_score_rows(m, all_posts_cv_tv[[m]]))

llo2_tv_df <- bind_rows(llo2_tv_rows)

# --- Naive benchmark: P_t = P_{t-6} (last observed value) ---
naive_llo2_rows <- lapply(unique(df_test_cv_tv$district_id), function(d) {
  y7     <- df_train_cv_tv %>% filter(district_id == d) %>% pull(SAM) %>% tail(1)
  y8_act <- df_model      %>% filter(district_id == d) %>% slice(8)  %>% pull(SAM)
  y9_act <- df_model      %>% filter(district_id == d) %>% slice(9)  %>% pull(SAM)
  dist_name <- district_names[d]
  tibble(
    Model     = "Naive",
    District  = dist_name,
    Point     = c("8th", "9th"),
    Predicted = c(y7, y8_act),
    Observed  = c(y8_act, y9_act),
    AbsError  = abs(c(y7, y8_act) - c(y8_act, y9_act)),
    SqError   = (c(y7, y8_act) - c(y8_act, y9_act))^2,
    MAPE      = ifelse(c(y8_act, y9_act) == 0, NA_real_,
                        100 * abs(c(y7, y8_act) - c(y8_act, y9_act)) / c(y8_act, y9_act)),
    CRPS      = NA_real_
  )
})
naive_llo2_df <- bind_rows(naive_llo2_rows)

# --- Combine all methods ---
llo2_all_df <- bind_rows(llo2_tv_df, naive_llo2_df)
write_csv(llo2_all_df, "outputs/LLO2_CV_all_models_with_CRPS.csv")
message("Saved: outputs/LLO2_CV_all_models_with_CRPS.csv")

# --- Summary statistics by model and prediction point ---
llo2_summary <- llo2_all_df %>%
  group_by(Model, Point) %>%
  summarise(
    RMSE      = sqrt(mean(SqError, na.rm = TRUE)),  # aggregate RMSE = sqrt(mean squared error), NOT mean(per-row abs error)
    MAE       = mean(AbsError, na.rm = TRUE),        # what the old "RMSE_mean" column actually computed
    MAPE_mean = mean(MAPE, na.rm = TRUE),
    CRPS_mean = mean(CRPS, na.rm = TRUE),
    .groups   = "drop"
  )
print(llo2_summary)
write_csv(llo2_summary, "outputs/LLO2_CV_summary_by_model.csv")
message("Saved: outputs/LLO2_CV_summary_by_model.csv")

# --- Bar plot: RMSE by district for all models (faceted by prediction point) ---
p_llo2_rmse <- ggplot(
  llo2_all_df,
  aes(x = reorder(District, -AbsError), y = AbsError, fill = Model)
) +
  geom_bar(stat = "identity", position = "dodge") +
  facet_wrap(~Point, nrow = 2, scales = "free_x") +
  coord_flip() +
  scale_fill_manual(
    values = c(tv_cv_colors, Naive = "grey50"),
    labels = c(TV_MODEL_DISPLAY, Naive = "Naive (P_t = P_{t-6})")
  ) +
  labs(title = "Leave-Last-2-Out CV: Absolute Error by District — All Models vs Naive",
       x = "District", y = "Absolute Error") +
  theme_minimal() +
  theme(strip.text = element_text(size = 10), axis.text.y = element_text(size = 7))
ggsave("plots/LLO2_CV_RMSE_all_models.jpg", p_llo2_rmse, width = 14, height = 16, dpi = 300)
message("Saved: plots/LLO2_CV_RMSE_all_models.jpg")

# --- Bar plot: CRPS by district for Bayesian models only (no naive) ---
p_llo2_crps <- ggplot(
  llo2_tv_df,
  aes(x = reorder(District, -CRPS), y = CRPS, fill = Model)
) +
  geom_bar(stat = "identity", position = "dodge") +
  facet_wrap(~Point, nrow = 2, scales = "free_x") +
  coord_flip() +
  scale_fill_manual(
    values = tv_cv_colors,
    labels = TV_MODEL_DISPLAY
  ) +
  labs(title = "Leave-Last-2-Out CV: CRPS by District — Bayesian Models",
       x = "District", y = "CRPS") +
  theme_minimal() +
  theme(strip.text = element_text(size = 10), axis.text.y = element_text(size = 7))
ggsave("plots/LLO2_CV_CRPS_bayesian_models.jpg", p_llo2_crps, width = 14, height = 16, dpi = 300)
message("Saved: plots/LLO2_CV_CRPS_bayesian_models.jpg")

# ---- District panel: observed SAM trajectory vs. each model's LL2O-CV ----
# ---- prediction at the 8th/9th held-out points, one small panel per   ----
# ---- district, so you can see visually where each model's prediction ----
# ---- landed relative to the actual trajectory (and relative to any   ----
# ---- outlier point flagged above in the outlier diagnostics).        ----
df_model_t <- df_model %>%
  group_by(District) %>%
  arrange(Year, Month, .by_group = TRUE) %>%
  mutate(t = row_number()) %>%
  ungroup()

cv_preds_for_plot <- llo2_tv_df %>%
  mutate(t = ifelse(Point == "8th", 8, 9))

p_cv_district_panel <- ggplot() +
  geom_line(data = df_model_t, aes(x = t, y = SAM), color = "black", linewidth = 0.4) +
  geom_point(data = df_model_t, aes(x = t, y = SAM), color = "black", size = 0.6) +
  geom_point(data = cv_preds_for_plot, aes(x = t, y = Predicted, color = Model), size = 1.3, alpha = 0.85) +
  facet_wrap(~District, scales = "free_y", ncol = 9) +
  scale_color_manual(values = TV_MODEL_COLORS, labels = TV_MODEL_DISPLAY, name = NULL) +
  scale_x_continuous(breaks = 1:9) +
  labs(
    title    = "Observed SAM trajectory vs. LL2O-CV predictions (8th/9th held-out points), by district",
    subtitle = "Black line/points = observed  |  colored points = each model's predicted value at that held-out point",
    x        = "Time point (1-9)",
    y        = "SAM"
  ) +
  theme_minimal(base_size = 7) +
  theme(
    strip.text       = element_text(size = 5),
    axis.text        = element_text(size = 4),
    axis.title       = element_text(size = 6),
    legend.position  = "bottom",
    legend.text      = element_text(size = 7),
    panel.grid.minor = element_blank()
  )

ggsave("plots/llo2_cv_district_panel_observed_vs_predicted.jpg", p_cv_district_panel, width = 22, height = 20, dpi = 250)
message("Saved: plots/llo2_cv_district_panel_observed_vs_predicted.jpg")

# ============================================================
# Leave-Last-3-Out Cross-Validation (robustness check)
# Same idea as the LL2O-CV above, but trains on only the first 6
# observations per district (instead of 7) and holds out three points
# (7th, 8th, 9th) instead of two. This lets us check two things the
# LL2O comparison alone can't:
#   - does "LLM priors win at one-step-ahead" replicate at a DIFFERENT,
#     non-outlier month (7th = July 2020; the outlier diagnostics above
#     show this month has a mean jump of only -12% and mean_abs_z_loo of
#     0.79 -- an unremarkable month), using a model that saw even less
#     training data (6 months instead of 7)?
#   - does the "informed priors lose at two-step-ahead" finding from
#     LL2O-CV still hold when the 8th observation is now predicted by a
#     model that never saw month 7 -- or was that reversal specific to
#     the July-2021 (9th observation) shock landing on that comparison?
# Predictions follow the same convention as LL2O-CV: each held-out point
# is predicted one step ahead from the TRUE observed value at the
# preceding timepoint (not recursively from the model's own prior
# prediction), so "N-step-ahead" here means "how many periods past the
# training cutoff," not compounding forecast uncertainty.
# ============================================================

df_train_cv3        <- df_model %>% group_by(district_id) %>% slice_head(n = 6) %>% ungroup()
df_test_cv3         <- df_model %>% group_by(district_id) %>% slice_tail(n = 3) %>% ungroup()
district_starts_cv3 <- which(!duplicated(df_train_cv3$district_id))

message("Fitting TV base model on 6-month training split for LL3O-CV...")
t0_fit <- Sys.time()
fit_cv3_tv_base <- stan(
  "sam_theoretical_variance.stan",
  data = list(
    N            = nrow(df_train_cv3),
    D            = D,
    district_id  = df_train_cv3$district_id,
    y            = df_train_cv3$SAM,
    pop_u5_ref   = as.vector(pop_u5_ref),
    start_index  = district_starts_cv3,
    p_hyperprior     = TV_MONTHLY_HYPERPRIORS$p,
    gamma_hyperprior = TV_MONTHLY_HYPERPRIORS$gamma,
    omega_hyperprior = TV_MONTHLY_HYPERPRIORS$omega
  ),
  chains = 4, warmup = 5000, iter = 10000, cores = 4, seed = 789
)
message("CV3 model fit time [tv_base]: ", round(as.numeric(difftime(Sys.time(), t0_fit, units = "mins")), 2), " minutes")
post_cv3_tv_base <- rstan::extract(fit_cv3_tv_base)
print(fit_cv3_tv_base)
.report_hmc_diagnostics(fit_cv3_tv_base, "tv_base (LL3O-CV train)")
param_summary_cv3_tv_base <- summary(fit_cv3_tv_base)$summary
rhat_df_cv3_tv_base <- data.frame(
  Parameter = rownames(param_summary_cv3_tv_base),
  Rhat      = param_summary_cv3_tv_base[, "Rhat"],
  n_eff     = param_summary_cv3_tv_base[, "n_eff"]
)
write_csv(rhat_df_cv3_tv_base, "outputs/rhat_and_ess_llto3_cv_tv_base.csv")
message("Saved: outputs/rhat_and_ess_llto3_cv_tv_base.csv")

fit_cv3_by_llm  <- list()
post_cv3_by_llm <- list()

# Same reasoning as the main fitting section and the LL2O-CV section above:
# each LLM's CV3 fit is its own standalone block, not a for-loop, so a
# failure on one doesn't require re-running the others. Reuses the
# already-compiled sam_informed_priors_model -- no new Stan compile here.

# ==== Claude ====
llm <- "claude"
message("Fitting TV LLM-informed STEADY-STATE model (", llm, ") on 6-month training split for LL3O-CV...")
fp <- fitted_priors_by_llm[[llm]]
t0_fit <- Sys.time()
fit_cv3_by_llm[[llm]] <- sampling(
  sam_informed_priors_model,
  data = list(
    N           = nrow(df_train_cv3),
    D           = D,
    district_id = df_train_cv3$district_id,
    y           = df_train_cv3$SAM,
    pop_u5_ref  = as.vector(pop_u5_ref),
    start_index = district_starts_cv3,
    a_p         = extract_prior_vec(fp, "p",     "a"),
    b_p         = extract_prior_vec(fp, "p",     "b"),
    a_gamma     = extract_prior_vec(fp, "gamma", "a"),
    b_gamma     = extract_prior_vec(fp, "gamma", "b"),
    a_omega     = extract_prior_vec(fp, "omega", "a"),
    b_omega     = extract_prior_vec(fp, "omega", "b")
  ),
  chains = 4, warmup = 5000, iter = 10000, cores = 4, seed = 789
)
message("CV3 model fit time [", llm, "]: ", round(as.numeric(difftime(Sys.time(), t0_fit, units = "mins")), 2), " minutes")
post_cv3_by_llm[[llm]] <- rstan::extract(fit_cv3_by_llm[[llm]])
print(fit_cv3_by_llm[[llm]])
.report_hmc_diagnostics(fit_cv3_by_llm[[llm]], paste0("tv_", llm, " (LL3O-CV train)"))

param_summary_cv3_llm <- summary(fit_cv3_by_llm[[llm]])$summary
rhat_df_cv3_llm <- data.frame(
  Parameter = rownames(param_summary_cv3_llm),
  Rhat      = param_summary_cv3_llm[, "Rhat"],
  n_eff     = param_summary_cv3_llm[, "n_eff"]
)
write_csv(rhat_df_cv3_llm, paste0("outputs/rhat_and_ess_llto3_cv_tv_", llm, ".csv"))
message("Saved: outputs/rhat_and_ess_llto3_cv_tv_", llm, ".csv")

# ==== Gemini ====
llm <- "gemini"
message("Fitting TV LLM-informed STEADY-STATE model (", llm, ") on 6-month training split for LL3O-CV...")
fp <- fitted_priors_by_llm[[llm]]
t0_fit <- Sys.time()
fit_cv3_by_llm[[llm]] <- sampling(
  sam_informed_priors_model,
  data = list(
    N           = nrow(df_train_cv3),
    D           = D,
    district_id = df_train_cv3$district_id,
    y           = df_train_cv3$SAM,
    pop_u5_ref  = as.vector(pop_u5_ref),
    start_index = district_starts_cv3,
    a_p         = extract_prior_vec(fp, "p",     "a"),
    b_p         = extract_prior_vec(fp, "p",     "b"),
    a_gamma     = extract_prior_vec(fp, "gamma", "a"),
    b_gamma     = extract_prior_vec(fp, "gamma", "b"),
    a_omega     = extract_prior_vec(fp, "omega", "a"),
    b_omega     = extract_prior_vec(fp, "omega", "b")
  ),
  chains = 4, warmup = 5000, iter = 10000, cores = 4, seed = 789
)
message("CV3 model fit time [", llm, "]: ", round(as.numeric(difftime(Sys.time(), t0_fit, units = "mins")), 2), " minutes")
post_cv3_by_llm[[llm]] <- rstan::extract(fit_cv3_by_llm[[llm]])
print(fit_cv3_by_llm[[llm]])
.report_hmc_diagnostics(fit_cv3_by_llm[[llm]], paste0("tv_", llm, " (LL3O-CV train)"))

param_summary_cv3_llm <- summary(fit_cv3_by_llm[[llm]])$summary
rhat_df_cv3_llm <- data.frame(
  Parameter = rownames(param_summary_cv3_llm),
  Rhat      = param_summary_cv3_llm[, "Rhat"],
  n_eff     = param_summary_cv3_llm[, "n_eff"]
)
write_csv(rhat_df_cv3_llm, paste0("outputs/rhat_and_ess_llto3_cv_tv_", llm, ".csv"))
message("Saved: outputs/rhat_and_ess_llto3_cv_tv_", llm, ".csv")

# ==== ChatGPT ====
llm <- "chatgpt"
message("Fitting TV LLM-informed STEADY-STATE model (", llm, ") on 6-month training split for LL3O-CV...")
fp <- fitted_priors_by_llm[[llm]]
t0_fit <- Sys.time()
fit_cv3_by_llm[[llm]] <- sampling(
  sam_informed_priors_model,
  data = list(
    N           = nrow(df_train_cv3),
    D           = D,
    district_id = df_train_cv3$district_id,
    y           = df_train_cv3$SAM,
    pop_u5_ref  = as.vector(pop_u5_ref),
    start_index = district_starts_cv3,
    a_p         = extract_prior_vec(fp, "p",     "a"),
    b_p         = extract_prior_vec(fp, "p",     "b"),
    a_gamma     = extract_prior_vec(fp, "gamma", "a"),
    b_gamma     = extract_prior_vec(fp, "gamma", "b"),
    a_omega     = extract_prior_vec(fp, "omega", "a"),
    b_omega     = extract_prior_vec(fp, "omega", "b")
  ),
  chains = 4, warmup = 5000, iter = 10000, cores = 4, seed = 789
)
message("CV3 model fit time [", llm, "]: ", round(as.numeric(difftime(Sys.time(), t0_fit, units = "mins")), 2), " minutes")
post_cv3_by_llm[[llm]] <- rstan::extract(fit_cv3_by_llm[[llm]])
print(fit_cv3_by_llm[[llm]])
.report_hmc_diagnostics(fit_cv3_by_llm[[llm]], paste0("tv_", llm, " (LL3O-CV train)"))

param_summary_cv3_llm <- summary(fit_cv3_by_llm[[llm]])$summary
rhat_df_cv3_llm <- data.frame(
  Parameter = rownames(param_summary_cv3_llm),
  Rhat      = param_summary_cv3_llm[, "Rhat"],
  n_eff     = param_summary_cv3_llm[, "n_eff"]
)
write_csv(rhat_df_cv3_llm, paste0("outputs/rhat_and_ess_llto3_cv_tv_", llm, ".csv"))
message("Saved: outputs/rhat_and_ess_llto3_cv_tv_", llm, ".csv")

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

tv_cv3_labels    <- TV_MODEL_LABELS
tv_cv3_colors    <- TV_MODEL_COLORS
all_posts_cv3_tv <- c(
  list(tv_base = post_cv3_tv_base),
  setNames(lapply(LLM_NAMES, function(llm) post_cv3_by_llm[[llm]]), paste0("tv_", LLM_NAMES))
)

# --- Run full posterior predictive with RMSE, MAPE, CRPS ---
llo3_tv_rows <- lapply(tv_cv3_labels, function(m) {
  message("Computing LL3O-CV predictions for TV model: ", m)
  preds <- .get_llo3_preds_tv(all_posts_cv3_tv[[m]], df_train_cv3, df_test_cv3, df_model, pop_u5_ref)
  med   <- apply(preds, 2, median)

  idx_map <- df_test_cv3 %>%
    mutate(Index = row_number()) %>%
    group_by(district_id) %>%
    summarise(
      District = first(District),
      idx_7 = sort(Index)[1], idx_8 = sort(Index)[2], idx_9 = sort(Index)[3],
      .groups = "drop"
    )

  n_dist <- nrow(idx_map)

  bind_rows(
    tibble(
      Model     = m,
      District  = idx_map$District,
      Point     = "7th",
      Predicted = med[idx_map$idx_7],
      Observed  = df_test_cv3$SAM[idx_map$idx_7],
      CRPS      = sapply(seq_len(n_dist), function(i)
                    scoringRules::crps_sample(
                      y   = df_test_cv3$SAM[idx_map$idx_7[i]],
                      dat = preds[, idx_map$idx_7[i]]))
    ) %>% mutate(AbsError = abs(Predicted - Observed),
                 SqError  = (Predicted - Observed)^2,
                 MAPE     = ifelse(Observed == 0, NA_real_, 100 * abs(Predicted - Observed) / Observed)),
    tibble(
      Model     = m,
      District  = idx_map$District,
      Point     = "8th",
      Predicted = med[idx_map$idx_8],
      Observed  = df_test_cv3$SAM[idx_map$idx_8],
      CRPS      = sapply(seq_len(n_dist), function(i)
                    scoringRules::crps_sample(
                      y   = df_test_cv3$SAM[idx_map$idx_8[i]],
                      dat = preds[, idx_map$idx_8[i]]))
    ) %>% mutate(AbsError = abs(Predicted - Observed),
                 SqError  = (Predicted - Observed)^2,
                 MAPE     = ifelse(Observed == 0, NA_real_, 100 * abs(Predicted - Observed) / Observed)),
    tibble(
      Model     = m,
      District  = idx_map$District,
      Point     = "9th",
      Predicted = med[idx_map$idx_9],
      Observed  = df_test_cv3$SAM[idx_map$idx_9],
      CRPS      = sapply(seq_len(n_dist), function(i)
                    scoringRules::crps_sample(
                      y   = df_test_cv3$SAM[idx_map$idx_9[i]],
                      dat = preds[, idx_map$idx_9[i]]))
    ) %>% mutate(AbsError = abs(Predicted - Observed),
                 SqError  = (Predicted - Observed)^2,
                 MAPE     = ifelse(Observed == 0, NA_real_, 100 * abs(Predicted - Observed) / Observed))
  )
})

llo3_tv_df <- bind_rows(llo3_tv_rows)

# --- Naive benchmark: P_t = P_{t-6} (last observed value) ---
naive_llo3_rows <- lapply(unique(df_test_cv3$district_id), function(d) {
  y6     <- df_train_cv3 %>% filter(district_id == d) %>% pull(SAM) %>% tail(1)
  y7_act <- df_model     %>% filter(district_id == d) %>% slice(7)  %>% pull(SAM)
  y8_act <- df_model     %>% filter(district_id == d) %>% slice(8)  %>% pull(SAM)
  y9_act <- df_model     %>% filter(district_id == d) %>% slice(9)  %>% pull(SAM)
  dist_name <- district_names[d]
  tibble(
    Model     = "Naive",
    District  = dist_name,
    Point     = c("7th", "8th", "9th"),
    Predicted = c(y6, y7_act, y8_act),
    Observed  = c(y7_act, y8_act, y9_act),
    AbsError  = abs(c(y6, y7_act, y8_act) - c(y7_act, y8_act, y9_act)),
    SqError   = (c(y6, y7_act, y8_act) - c(y7_act, y8_act, y9_act))^2,
    MAPE      = ifelse(c(y7_act, y8_act, y9_act) == 0, NA_real_,
                        100 * abs(c(y6, y7_act, y8_act) - c(y7_act, y8_act, y9_act)) / c(y7_act, y8_act, y9_act)),
    CRPS      = NA_real_
  )
})
naive_llo3_df <- bind_rows(naive_llo3_rows)

# --- Combine all methods ---
llo3_all_df <- bind_rows(llo3_tv_df, naive_llo3_df)
write_csv(llo3_all_df, "outputs/LLO3_CV_all_models_with_CRPS.csv")
message("Saved: outputs/LLO3_CV_all_models_with_CRPS.csv")

# --- Summary statistics by model and prediction point ---
llo3_summary <- llo3_all_df %>%
  group_by(Model, Point) %>%
  summarise(
    RMSE      = sqrt(mean(SqError, na.rm = TRUE)),  # aggregate RMSE = sqrt(mean squared error), NOT mean(per-row abs error)
    MAE       = mean(AbsError, na.rm = TRUE),        # what the old "RMSE_mean" column actually computed
    MAPE_mean = mean(MAPE, na.rm = TRUE),
    CRPS_mean = mean(CRPS, na.rm = TRUE),
    .groups   = "drop"
  )
print(llo3_summary)
write_csv(llo3_summary, "outputs/LLO3_CV_summary_by_model.csv")
message("Saved: outputs/LLO3_CV_summary_by_model.csv")

# --- Bar plot: RMSE by district for all models (faceted by prediction point) ---
p_llo3_rmse <- ggplot(
  llo3_all_df,
  aes(x = reorder(District, -AbsError), y = AbsError, fill = Model)
) +
  geom_bar(stat = "identity", position = "dodge") +
  facet_wrap(~Point, nrow = 3, scales = "free_x") +
  coord_flip() +
  scale_fill_manual(
    values = c(tv_cv3_colors, Naive = "grey50"),
    labels = c(TV_MODEL_DISPLAY, Naive = "Naive (P_t = P_{t-6})")
  ) +
  labs(title = "Leave-Last-3-Out CV (robustness check): Absolute Error by District — All Models vs Naive",
       x = "District", y = "Absolute Error") +
  theme_minimal() +
  theme(strip.text = element_text(size = 10), axis.text.y = element_text(size = 7))
ggsave("plots/LLO3_CV_RMSE_all_models.jpg", p_llo3_rmse, width = 14, height = 20, dpi = 300)
message("Saved: plots/LLO3_CV_RMSE_all_models.jpg")

# --- Bar plot: CRPS by district for Bayesian models only (no naive) ---
p_llo3_crps <- ggplot(
  llo3_tv_df,
  aes(x = reorder(District, -CRPS), y = CRPS, fill = Model)
) +
  geom_bar(stat = "identity", position = "dodge") +
  facet_wrap(~Point, nrow = 3, scales = "free_x") +
  coord_flip() +
  scale_fill_manual(
    values = tv_cv3_colors,
    labels = TV_MODEL_DISPLAY
  ) +
  labs(title = "Leave-Last-3-Out CV (robustness check): CRPS by District — Bayesian Models",
       x = "District", y = "CRPS") +
  theme_minimal() +
  theme(strip.text = element_text(size = 10), axis.text.y = element_text(size = 7))
ggsave("plots/LLO3_CV_CRPS_bayesian_models.jpg", p_llo3_crps, width = 14, height = 20, dpi = 300)
message("Saved: plots/LLO3_CV_CRPS_bayesian_models.jpg")

# ---- District panel: observed SAM trajectory vs. each model's LL3O-CV ----
# ---- prediction at the 7th/8th/9th held-out points, one small panel   ----
# ---- per district (same style as the LL2O-CV panel above).            ----
cv3_preds_for_plot <- llo3_tv_df %>%
  mutate(t = case_when(Point == "7th" ~ 7, Point == "8th" ~ 8, TRUE ~ 9))

p_cv3_district_panel <- ggplot() +
  geom_line(data = df_model_t, aes(x = t, y = SAM), color = "black", linewidth = 0.4) +
  geom_point(data = df_model_t, aes(x = t, y = SAM), color = "black", size = 0.6) +
  geom_point(data = cv3_preds_for_plot, aes(x = t, y = Predicted, color = Model), size = 1.3, alpha = 0.85) +
  facet_wrap(~District, scales = "free_y", ncol = 9) +
  scale_color_manual(values = TV_MODEL_COLORS, labels = TV_MODEL_DISPLAY, name = NULL) +
  scale_x_continuous(breaks = 1:9) +
  labs(
    title    = "Observed SAM trajectory vs. LL3O-CV predictions (7th/8th/9th held-out points), by district",
    subtitle = "Black line/points = observed  |  colored points = each model's predicted value at that held-out point (robustness check, 6-month training window)",
    x        = "Time point (1-9)",
    y        = "SAM"
  ) +
  theme_minimal(base_size = 7) +
  theme(
    strip.text       = element_text(size = 5),
    axis.text        = element_text(size = 4),
    axis.title       = element_text(size = 6),
    legend.position  = "bottom",
    legend.text      = element_text(size = 7),
    panel.grid.minor = element_blank()
  )

ggsave("plots/llo3_cv_district_panel_observed_vs_predicted.jpg", p_cv3_district_panel, width = 22, height = 20, dpi = 250)
message("Saved: plots/llo3_cv_district_panel_observed_vs_predicted.jpg")

# ============================================================
# End of Leave-Last-3-Out CV (robustness check)
# ============================================================

# ============================================================
# End of theoretical variance informed priors model
# ============================================================

# ---- Prepare map inputs: use in-memory posteriors or fall back to saved CSVs ----

# Standalone-safe prerequisites: this section (and Visualization below) only
# needs LLM_NAMES, district_names, df_filtered, and df_model -- not any
# fitted posterior -- since everything below already falls back to the saved
# CSVs in outputs/ when a posterior isn't in memory. If you're running just
# the maps section in a fresh session rather than the full pipeline above,
# these guards reconstruct that handful of cheap, non-Stan objects (same
# data load as the top of the script) so you don't have to re-run or
# refit anything. Skipped entirely if you already have a full session.
if (!exists("LLM_NAMES")) LLM_NAMES <- c("claude", "gemini", "chatgpt")

if (!exists("df_model")) {
  message("df_model not in memory — reloading and re-cleaning raw data for maps.")
  df_raw <- read.csv("C:/Users/ecemy/OneDrive - INSEAD/Desktop/Waste prediction/Datasets/Final_Clean_Dataset.csv")

  df_filtered <- df_raw[!( (df_raw$Year == 2016 & df_raw$Month == 1) |
                             (df_raw$Year == 2016 & df_raw$Month == 7) |
                             (df_raw$Year == 2017 & df_raw$Month == 1)), ]

  rab_dhuure_data <- df_filtered[df_filtered$District == "Rab Dhuure", ]
  avg_sam <- mean(rab_dhuure_data$SAM, na.rm = TRUE)
  rab_dhuure_data$SAM[is.na(rab_dhuure_data$SAM)] <- avg_sam

  df_filtered <- df_filtered %>%
    filter(District != "Rab Dhuure") %>%
    bind_rows(rab_dhuure_data) %>%
    arrange(District, Year, Month)

  df_filtered$district_id <- as.numeric(as.factor(df_filtered$District))
  district_names <- levels(as.factor(df_filtered$District))

  df_filtered <- df_filtered %>%
    rename(Pop_U5 = Population.Under.5)

  df_model <- df_filtered %>%
    filter(!is.na(District)) %>%
    arrange(District, Year, Month) %>%
    mutate(
      district_id = as.integer(factor(District, levels = unique(District))),
      Timepoint   = paste(Year, Month, sep = "-")
    )

  district_names <- levels(factor(df_model$District, levels = unique(df_model$District)))
}

.prep_from_post <- function(post, burden_sum_long) {
  list(
    param_medians = tibble(
      District    = district_names,
      p           = apply(post$p, 2, median),
      gamma       = apply(post$gamma, 2, median),
      gamma_count = apply(post$gamma_count_out, 2, median)
    ),
    burden_sum = burden_sum_long,
    burden_expected = post$burden %>%
      as.data.frame() %>%
      setNames(paste0("V", seq_len(ncol(.)))) %>%
      mutate(iteration = row_number()) %>%
      pivot_longer(-iteration, names_to = "idx", values_to = "burden") %>%
      mutate(idx = as.integer(gsub("V", "", idx)),
             District = df_filtered$District[idx]) %>%
      group_by(District) %>%
      summarise(burden_expected = mean(burden), .groups = "drop")
  )
}

.prep_from_csv <- function(csv_path) {
  df <- read_csv(csv_path, show_col_types = FALSE)
  # Saved summary CSVs have three column families per timepoint:
  # burden_<t> (posterior MEDIAN), burden_lo_<t> and burden_hi_<t> (95% CI
  # endpoints). Match burden_<t> only -- starts_with("burden_") would also
  # sweep in burden_lo_/burden_hi_ as if they were additional timepoints'
  # medians, corrupting both burden_sum and burden_expected below.
  burden_cols <- grep("^burden_(?!lo_|hi_)", names(df), perl = TRUE, value = TRUE)
  burden_long <- df %>%
    select(District, all_of(burden_cols)) %>%
    pivot_longer(-District, names_to = "Timepoint", values_to = "burden_median") %>%
    mutate(Timepoint = sub("^burden_", "", Timepoint))
  message("Note: burden_expected from CSV fallback is the across-timepoint MEAN of ",
          "saved posterior MEDIANS (not a true posterior mean/expectation, and not ",
          "necessarily identical to the in-memory .prep_from_post() calculation, which ",
          "averages raw posterior burden draws directly).")
  list(
    param_medians = df %>% select(District, p, gamma, gamma_count),
    burden_sum    = burden_long,
    burden_expected = burden_long %>%
      group_by(District) %>%
      summarise(burden_expected = mean(burden_median, na.rm = TRUE), .groups = "drop")
  )
}

if (exists("posterior_tv") && !is.null(posterior_tv)) {
  map_inputs_tv <- .prep_from_post(posterior_tv, burden_summary_tv)
} else {
  message("posterior_tv not in memory — loading from CSV for maps.")
  map_inputs_tv <- .prep_from_csv("outputs/parameter_and_burden_summary_theoretical_variance.csv")
}

map_inputs_by_llm <- list()
for (llm in LLM_NAMES) {
  if (exists("posterior_by_llm") && !is.null(posterior_by_llm[[llm]])) {
    map_inputs_by_llm[[llm]] <- .prep_from_post(posterior_by_llm[[llm]], burden_summary_by_llm[[llm]])
  } else {
    message("posterior for ", llm, " not in memory — loading from CSV for maps.")
    map_inputs_by_llm[[llm]] <- .prep_from_csv(paste0("outputs/parameter_and_burden_summary_tv_", llm, ".csv"))
  }
}

# K_expected = E[burden] / E[SAM] - 1, computed for tv_base and each LLM-informed model
sam_mean_per_district <- df_filtered %>%
  group_by(District) %>%
  summarise(SAM_mean = mean(SAM, na.rm = TRUE), .groups = "drop")

.compute_k_expected <- function(burden_expected_df) {
  left_join(burden_expected_df, sam_mean_per_district, by = "District") %>%
    mutate(K_expected = burden_expected / SAM_mean - 1) %>%
    filter(is.finite(K_expected))
}

K_expected_summary <- .compute_k_expected(map_inputs_tv$burden_expected)
K_expected_summary_by_llm <- list()
for (llm in LLM_NAMES) {
  K_expected_summary_by_llm[[llm]] <- .compute_k_expected(map_inputs_by_llm[[llm]]$burden_expected)
}

#------------------------------------- Visualization -------------------------------------

# ---- Name harmonization ----
district_name_mapping <- c(
  "Adan Yabaal" = "Aadan", "Afgooye" = "Afgooye", "Afmadow" = "Afmadow",
  "Baardheere" = "Baar-Dheere", "Badhaadhe" = "Badhaadhe", "Baki" = "Baki",
  "Balcad" = "Balcad", "Banadir" = "Mogadisho", "Bandarbeyla" = "Bander-Beyla",
  "Baraawe" = "Baraawe", "Baydhaba" = "Baydhabo", "Belet Weyne" = "Beled Weyn",
  "Belet Xaawo" = "Beled Xaawo", "Berbera" = "Berbera", "Borama" = "Boorama",
  "Bossaso" = "Bosaaso", "Bu'aale" = "Bu'aale", "Bulo Burto" = "Buulo Burdo",
  "Burco" = "Burao", "Burtinle" = "Burtinle", "Buuhoodle" = "Buuhoodle",
  "Buur Hakaba" = "Buur Xakaba", "Cabudwaaq" = "Caabudwaaq", "Cadaado" = "Cadaado",
  "Cadale" = "Cadale", "Caluula" = "Calawla", "Caynabo" = "Caynabo",
  "Ceel Afweyn" = "Ceel-Afwein", "Ceel Barde" = "Ceel Barde", "Ceel Buur" = "Ceel Buur",
  "Ceel Dheere" = "Ceel Dheer", "Ceel Waaq" = "Ceel Waaq", "Ceerigaabo" = "Ceerigaabo",
  "Dhuusamarreeb" = "Dhuusamareeb", "Diinsoor" = "Diinsoor", "Doolow" = "Dolow",
  "Eyl" = "Eyl", "Gaalkacyo" = "Gaalkacayo", "Galdogob" = "Goldogob",
  "Garbahaarey" = "Garbahaaray", "Garoowe" = "Garoowe", "Gebiley" = "Gabiley",
  "Hargeysa" = "Hargeysa", "Hobyo" = "Hobyo", "Iskushuban" = "Iskushuban",
  "Jalalaqsi" = "Jalalaqsi", "Jamaame" = "Jamaame", "Jariiban" = "Jariiban",
  "Jilib" = "Jilib", "Jowhar" = "Jawhar", "Kismaayo" = "Kismaayo",
  "Kurtunwaarey" = "Kuntuwaaray", "Laas Caanood" = "Lascaanod", "Laasqoray" = "Badhan",
  "Lughaye" = "Lughaya", "Luuq" = "Luuk", "Marka" = "Marka", "Owdweyne" = "Oodweyne",
  "Qandala" = "Qandala", "Qansax Dheere" = "Qansax Dheere", "Qardho" = "Qardho",
  "Qoryooley" = "Qoryooley", "Saakow" = "Saakow", "Sablaale" = "Sablale",
  "Sheikh" = "Sheekh", "Taleex" = "Taleex", "Tayeeglow" = "Tiyeeglow",
  "Waajid" = "Wajid", "Wanla Weyn" = "Wanla Weyn", "Xarardheere" = "Xarardheere",
  "Xudun" = "Xudun", "Xudur" = "Xudur", "Zeylac" = "Zeylac",
  "Rab Dhuure" = "Rab Dhuure"
)

# ---- Load and prepare shapefile ----
shapefile_path <- "C:/Users/ecemy/OneDrive - INSEAD/Desktop/Waste prediction/Code/SOM_adm_2015_July/SOM_adm2.shp"

somalia_map <- st_read(shapefile_path) %>%
  rename(Shapefile_District_Name = NAME_2)

# ---- Unified sequential color palette (all comparison maps) ----
# One green ramp (light yellow-green = low -> dark green = high) reused
# across duration/incidence/burden/K maps, so a reader learns the gradient
# once instead of a different legend logic per map. Built with hcl.colors()
# (perceptually uniform HCL space) rather than RGB interpolation between a
# handful of hex stops, so steps stay visually distinct step-to-step even at
# high bin counts (e.g. duration's 19 bins) instead of bunching up in the
# middle of the ramp.
green_palette <- function(n) rev(grDevices::hcl.colors(n, "YlGn"))

# Shared burden bin scheme -- used by both the per-model burden maps
# (.make_tv_model_maps() below) and the standalone benchmark burden map, so
# the two stay visually comparable and share one legend definition.
burden_map_breaks <- c(0, seq(5000, 50000, by = 5000), Inf)
burden_map_labels <- c("0-5k","5k-10k","10k-15k","15k-20k","20k-25k",
                        "25k-30k","30k-35k","35k-40k","40k-45k","45k-50k",">50k")
burden_map_colors <- setNames(green_palette(length(burden_map_labels)), burden_map_labels)

# Half-month boundaries from 1 to 10 months, open-ended last bin. cut() below
# uses right = FALSE, so the last interval is [10, Inf) -- i.e. 10 itself
# falls in the ">=10" bin, not the "9.5-10" bin, hence the ">=" (not ">").
duration_breaks <- c(seq(1, 10, by = 0.5), Inf)
duration_labels <- c(
  paste0(head(duration_breaks, -2), "–",
         duration_breaks[2:(length(duration_breaks) - 1)]),
  ">=10"
)
duration_colors <- setNames(green_palette(length(duration_labels)), duration_labels)

name_map <- tibble::tibble(
  District = names(district_name_mapping),
  Shapefile_District_Name = unname(district_name_mapping)
)

# ---- K Expected Map ----
# right = FALSE (as in the original code -- preserved here, not changed):
# intervals are [a,b), so the last interval is [10, Inf) and K_expected == 10
# itself falls in the last bin -- hence ">=10" (not ">10"), same reasoning as
# the duration map fix above.
k_bin_breaks <- c(0, 2, 3, 4, 5, 6, 7, 8, 9, 10, Inf)
k_bin_labels <- c("0–2", "2–3", "3–4", "4–5", "5–6", "6–7", "7–8", "8–9", "9–10", ">=10")
k_map_colors <- setNames(green_palette(10), k_bin_labels)

k_map_display_names <- c(tv_base = "Base", tv_claude = "Claude", tv_gemini = "Gemini", tv_chatgpt = "ChatGPT")

# Builds and saves one K-expected choropleth. No lat/long axes, ticks, or
# gridlines are drawn -- these maps show a categorical fill per district
# polygon only, not a geographic coordinate system.
.make_k_expected_map <- function(k_expected_df, model_label) {
  k_join <- k_expected_df %>% dplyr::left_join(name_map, by = "District")

  # Quick sanity: any districts in K that didn't map to a shapefile name?
  unmapped <- dplyr::anti_join(k_join, somalia_map, by = "Shapefile_District_Name")
  if (nrow(unmapped)) {
    message("Heads up: these Districts didn't find a shapefile match (", model_label, "):\n",
            paste(unique(unmapped$District), collapse = ", "))
  }

  map_k <- somalia_map %>%
    dplyr::left_join(k_join, by = "Shapefile_District_Name") %>%
    dplyr::filter(!is.na(K_expected)) %>%
    dplyr::mutate(
      K_bin = cut(K_expected, breaks = k_bin_breaks, labels = k_bin_labels,
                 right = FALSE, include.lowest = TRUE),
      # Explicit factor over the FULL label set so unused bins still get a
      # legend key below (same fix as the duration/incidence/burden maps).
      K_bin = factor(K_bin, levels = k_bin_labels)
    )

  display_label <- unname(ifelse(model_label %in% names(k_map_display_names),
                                  k_map_display_names[model_label], model_label))

  p_k <- ggplot(map_k) +
    geom_sf(aes(fill = K_bin), colour = "black", size = 0.1, show.legend = TRUE) +
    scale_fill_manual(
      name     = "K (Expected)",
      values   = k_map_colors,
      limits   = k_bin_labels,
      breaks   = k_bin_labels,
      labels   = k_bin_labels,
      drop     = FALSE,
      na.value = "grey80"
    ) +
    labs(title = paste0("Expected K per District (E[Burden] / E[SAM] - 1) — ", display_label)) +
    theme_minimal() +
    theme(axis.text = element_blank(), axis.ticks = element_blank(),
          axis.title = element_blank(), panel.grid = element_blank())

  if (!dir.exists("plots")) dir.create("plots", recursive = TRUE)
  ggsave(paste0("plots/K_expected_map_", model_label, ".jpg"), plot = p_k, width = 10, height = 8, dpi = 300)
  message("Saved: plots/K_expected_map_", model_label, ".jpg")
}

.make_k_expected_map(K_expected_summary, "tv_base")
for (llm in LLM_NAMES) {
  .make_k_expected_map(K_expected_summary_by_llm[[llm]], paste0("tv_", llm))
}


# ============================================================
# Geographic orientation overlays for the TV model comparison maps
# ============================================================
# Selected district labels, Admin-1 region boundaries, and regional text
# labels are all computed once here (not per-model) and reused identically
# across the base and every LLM-informed map, so label placement stays
# directly comparable across models.

# Individual district-label boxes are reserved for districts the manuscript
# cites with a specific number or as an individually-argued exception --
# NOT for districts that are part of a uniform regional pattern (those are
# covered by the region-level text annotations below instead: Bakool, Bay,
# Lower Shabelle, Somaliland).
districts_to_label <- c(
  "Baraawe",
  "Hargeysa",
  "Jamaame",
  "Kismaayo",
  "Iskushuban"
)

# ---- District label coordinates (computed once, reused everywhere) ----
district_label_points_sf <- somalia_map %>%
  dplyr::inner_join(
    dplyr::filter(name_map, District %in% districts_to_label),
    by = "Shapefile_District_Name"
  )

.matched <- sort(unique(district_label_points_sf$District))
.missing <- setdiff(districts_to_label, .matched)
.dup_counts <- table(district_label_points_sf$District)
.duplicated <- names(.dup_counts[.dup_counts > 1])

if (length(.missing) > 0 || length(.duplicated) > 0) {
  stop(
    "District label coordinate setup failed.\n",
    if (length(.missing) > 0)
      paste0("  Missing (no shapefile match): ", paste(.missing, collapse = ", "), "\n") else "",
    if (length(.duplicated) > 0)
      paste0("  Duplicated matches: ", paste(.duplicated, collapse = ", "), "\n") else ""
  )
}
message("Geographic orientation: all ", length(districts_to_label),
        " requested district labels matched the shapefile exactly once.")

# st_point_on_surface() (not an ordinary centroid) so the label anchor always
# falls inside its district polygon, even for oddly-shaped districts.
district_label_points <- district_label_points_sf %>%
  sf::st_point_on_surface() %>%
  dplyr::mutate(
    X = sf::st_coordinates(.)[, "X"],
    Y = sf::st_coordinates(.)[, "Y"]
  ) %>%
  sf::st_drop_geometry() %>%
  dplyr::select(District, X, Y)

# Fixed nudges for labels that still overlap after ggrepel -- applied
# identically to every model/parameter map, never per-model.
district_label_nudges <- tibble::tribble(
  ~District, ~nudge_x, ~nudge_y
)

# ---- Admin-1 region boundaries (dissolved from district polygons) ----
# Drawn as a map layer in .add_geo_orientation_layers() below, and also used
# to derive region label positions (Bakool, Bay) and the Somaliland
# orientation text.
somalia_region_boundaries <- somalia_map %>%
  dplyr::group_by(NAME_1) %>%
  dplyr::summarise(geometry = sf::st_union(geometry), .groups = "drop")

# ---- Subtle Admin-1 labels: Bakool, Bay ----
# Bakool covers the labeled long-duration cluster (Ceel Barde/Rab Dhuure/
# Xudur/Tayeeglow/Waajid); Bay covers Baydhaba/Diinsoor/Buur Hakaba. Both
# regions are single compact blobs, so st_point_on_surface() on the dissolved
# Admin-1 polygon lands centrally without further adjustment.
region_label_points <- somalia_region_boundaries %>%
  dplyr::filter(NAME_1 %in% c("Bakool", "Bay")) %>%
  sf::st_point_on_surface() %>%
  dplyr::mutate(
    X = sf::st_coordinates(.)[, "X"],
    Y = sf::st_coordinates(.)[, "Y"],
    label = NAME_1
  ) %>%
  sf::st_drop_geometry() %>%
  dplyr::select(label, X, Y)

# ---- "Lower Shabelle" label: centered across its coastal (Baraawe) and
# inland (Afgooye, Wanla Weyn) sub-clusters ----
# st_point_on_surface() on the dissolved Shabeellaha Hoose polygon lands
# right next to Baraawe (its largest sub-area dominates the pick), so
# instead this averages the anchor points of the three cited districts --
# giving one position that reasonably represents all of them rather than
# favoring the coastal end.
lower_shabelle_anchor_districts <- c("Baraawe", "Afgooye", "Wanla Weyn")
lower_shabelle_anchors <- somalia_map %>%
  dplyr::inner_join(
    dplyr::filter(name_map, District %in% lower_shabelle_anchor_districts),
    by = "Shapefile_District_Name"
  ) %>%
  sf::st_point_on_surface() %>%
  dplyr::mutate(
    X = sf::st_coordinates(.)[, "X"],
    Y = sf::st_coordinates(.)[, "Y"]
  ) %>%
  sf::st_drop_geometry()

region_label_points <- dplyr::bind_rows(
  region_label_points,
  data.frame(
    label = "Lower Shabelle",
    X = mean(lower_shabelle_anchors$X),
    Y = mean(lower_shabelle_anchors$Y)
  )
)

# ---- "Somaliland" orientation label (no boundary drawn) ----
# Purpose is geographic orientation only -- it is not a political boundary
# and does not modify the shapefile's official Admin-1 boundaries. Position
# is calculated from the combined geometry of the northwestern Admin-1
# regions that contain the labeled Somaliland districts (Awdal, Woqooyi
# Galbeed, Togdheer), placed near the northern edge of that combined area so
# it sits above the district-label cluster (Hargeysa/Borama/Baki/Owdweyne/
# Buuhoodle) instead of competing with it -- district labels keep priority.
somaliland_orientation_geom <- somalia_region_boundaries %>%
  dplyr::filter(NAME_1 %in% c("Awdal", "Woqooyi Galbeed", "Togdheer")) %>%
  sf::st_union()
somaliland_bbox <- sf::st_bbox(somaliland_orientation_geom)
somaliland_label_point <- data.frame(
  label = "Somaliland",
  X = unname((somaliland_bbox["xmin"] + somaliland_bbox["xmax"]) / 2),
  Y = unname(somaliland_bbox["ymax"] - 0.4)
)

# Adds the shared geographic orientation layers (Admin-1 region boundaries,
# subtle regional text labels, and selected district labels) on top of a
# district-level choropleth. Reused identically by the duration,
# incidence-count, and incidence-rate maps for every TV model so placement
# stays comparable. Region boundaries are drawn visibly thicker than the
# thin district borders (geom_sf linewidth = 0.5 on the fill layer above) so
# the two stay distinguishable at a glance.
.add_geo_orientation_layers <- function(p) {
  district_labels_nudged <- district_label_points %>%
    dplyr::left_join(district_label_nudges, by = "District") %>%
    dplyr::mutate(
      nudge_x = ifelse(is.na(nudge_x), 0, nudge_x),
      nudge_y = ifelse(is.na(nudge_y), 0, nudge_y)
    )

  p +
    geom_sf(
      data = somalia_region_boundaries,
      fill = NA, colour = "grey15", linewidth = 0.9,
      inherit.aes = FALSE
    ) +
    geom_label(
      data = region_label_points,
      aes(x = X, y = Y, label = label),
      inherit.aes = FALSE, size = 3, colour = "grey30",
      fill = scales::alpha("white", 0.65), linewidth = 0,
      label.padding = grid::unit(0.1, "lines")
    ) +
    geom_label(
      data = somaliland_label_point,
      aes(x = X, y = Y, label = label),
      inherit.aes = FALSE, size = 3.6, colour = "grey40", fontface = "italic",
      fill = scales::alpha("white", 0.65), linewidth = 0,
      label.padding = grid::unit(0.1, "lines")
    ) +
    ggrepel::geom_label_repel(
      data = district_labels_nudged,
      aes(x = X, y = Y, label = District),
      inherit.aes = FALSE,
      nudge_x = district_labels_nudged$nudge_x,
      nudge_y = district_labels_nudged$nudge_y,
      size = 2.5,
      seed = 123,
      min.segment.length = 0,
      segment.color = "grey35",
      segment.size = 0.25,
      box.padding = 0.25,
      point.padding = 0.15,
      label.padding = 0.12,
      label.size = 0.15,
      fill = scales::alpha("white", 0.80),
      max.overlaps = Inf
    )
}


# ============================================================
# Model comparison maps: duration, incidence, burden (all 3 TV models)
# ============================================================

# Helper: generate duration, incidence, and burden maps for one model
.make_tv_model_maps <- function(param_medians, burden_sum, model_label,
                                 name_map, somalia_map,
                                 duration_colors, gamma_cap, gamma_rate_cap) {
  dir.create("plots", showWarnings = FALSE)

  # Titles show a nicely-cased model name (e.g. "Base", "ChatGPT") -- the
  # "tv_" prefix stays in model_label itself since it's still used for
  # filenames. Falls back to the stripped label if a new model name shows up.
  model_display_names <- c(base = "Base", claude = "Claude", gemini = "Gemini", chatgpt = "ChatGPT")
  model_key <- sub("^tv_", "", model_label)
  display_label <- unname(ifelse(model_key %in% names(model_display_names),
                                  model_display_names[model_key], model_key))

  # 1. Duration map (1/p)
  dur_data <- param_medians %>%
    mutate(duration = 1 / p) %>%
    select(District, duration) %>%
    dplyr::left_join(name_map, by = "District")

  map_dur <- somalia_map %>%
    dplyr::left_join(dur_data, by = "Shapefile_District_Name") %>%
    dplyr::filter(!is.na(duration)) %>%
    dplyr::mutate(
      duration_bin = cut(duration,
                         breaks = duration_breaks,
                         labels = duration_labels,
                         right = FALSE, include.lowest = TRUE),
      # Explicit factor over the FULL label set (not just levels present in
      # this model's data) so unused bins still get a legend key below.
      duration_bin = factor(duration_bin, levels = duration_labels)
    )
  p_dur <- ggplot(map_dur) +
    geom_sf(aes(fill = duration_bin), colour = "black", linewidth = 0.25, show.legend = TRUE) +
    scale_fill_manual(
      name     = "Duration (months)",
      values   = duration_colors,
      limits   = duration_labels,
      breaks   = duration_labels,
      labels   = duration_labels,
      drop     = FALSE,
      na.value = "grey80"
    ) +
    labs(title = paste0("Duration (1/p) — ", display_label)) +
    theme_minimal() +
    theme(axis.text = element_blank(), axis.ticks = element_blank(),
          axis.title = element_blank(), panel.grid = element_blank())
  p_dur <- .add_geo_orientation_layers(p_dur)
  ggsave(paste0("plots/duration_map_", model_label, ".jpg"), p_dur, width = 10, height = 8, dpi = 300)
  message("Saved: plots/duration_map_", model_label, ".jpg")

  # 2a. Incidence map (case counts) — shared discrete scale; whichever
  # district(s) exceed the shared cap land in a plain open-ended top bin
  # (not naming a specific district, since that isn't always the same one
  # across models).
  n_inc_bins  <- 8
  bin_step    <- gamma_cap / n_inc_bins
  bin_breaks  <- c(seq(0, gamma_cap, by = bin_step), Inf)
  bin_labels  <- c(
    sapply(seq_len(n_inc_bins), function(i)
      paste0(round((i - 1) * bin_step / 1000, 1), "k–",
             round(i * bin_step / 1000, 1), "k")),
    paste0(">", round(gamma_cap / 1000, 1), "k")
  )
  inc_colors  <- c(green_palette(n_inc_bins), "#FFD700")   # gold for the open-ended top bin

  inc_data <- param_medians %>%
    select(District, gamma_count) %>%
    dplyr::left_join(name_map, by = "District") %>%
    dplyr::mutate(
      gamma_bin = cut(gamma_count, breaks = bin_breaks,
                      labels = bin_labels, right = FALSE, include.lowest = TRUE),
      # Explicit factor over the FULL label set so unused bins still get a
      # legend key below (same fix as the duration map above).
      gamma_bin = factor(gamma_bin, levels = bin_labels)
    )

  map_inc <- somalia_map %>%
    dplyr::left_join(inc_data, by = "Shapefile_District_Name") %>%
    dplyr::filter(!is.na(gamma_bin))

  p_inc <- ggplot(map_inc) +
    geom_sf(aes(fill = gamma_bin), colour = "black", linewidth = 0.25, show.legend = TRUE) +
    scale_fill_manual(
      name     = "New cases\nper period",
      values   = setNames(inc_colors, bin_labels),
      limits   = bin_labels,
      breaks   = bin_labels,
      labels   = bin_labels,
      drop     = FALSE,
      na.value = "grey80"
    ) +
    labs(title = paste0("Incidence (new cases per period) — ", display_label)) +
    theme_minimal() +
    theme(axis.text = element_blank(), axis.ticks = element_blank(),
          axis.title = element_blank(), panel.grid = element_blank())
  p_inc <- .add_geo_orientation_layers(p_inc)
  ggsave(paste0("plots/incidence_map_", model_label, ".jpg"), p_inc, width = 10, height = 8, dpi = 300)
  message("Saved: plots/incidence_map_", model_label, ".jpg")

  # 2b. Incidence rate map (gamma itself, as % of the U5 population) — same
  # binning logic as the count map above, just on the rate scale instead of
  # raw counts, so it stays comparable across models the same way.
  n_rate_bins  <- 8
  rate_step    <- gamma_rate_cap / n_rate_bins
  rate_breaks  <- c(seq(0, gamma_rate_cap, by = rate_step), Inf)
  rate_labels  <- c(
    sapply(seq_len(n_rate_bins), function(i)
      paste0(round((i - 1) * rate_step * 100, 2), "%–",
             round(i * rate_step * 100, 2), "%")),
    paste0(">", round(gamma_rate_cap * 100, 2), "%")
  )
  rate_colors  <- c(green_palette(n_rate_bins), "#FFD700")   # gold for the open-ended top bin

  rate_data <- param_medians %>%
    select(District, gamma) %>%
    dplyr::left_join(name_map, by = "District") %>%
    dplyr::mutate(
      gamma_rate_bin = cut(gamma, breaks = rate_breaks,
                           labels = rate_labels, right = FALSE, include.lowest = TRUE),
      gamma_rate_bin = factor(gamma_rate_bin, levels = rate_labels)
    )

  map_rate <- somalia_map %>%
    dplyr::left_join(rate_data, by = "Shapefile_District_Name") %>%
    dplyr::filter(!is.na(gamma_rate_bin))

  p_rate <- ggplot(map_rate) +
    geom_sf(aes(fill = gamma_rate_bin), colour = "black", linewidth = 0.25, show.legend = TRUE) +
    scale_fill_manual(
      name     = "Incidence rate\n(% of U5 pop)",
      values   = setNames(rate_colors, rate_labels),
      limits   = rate_labels,
      breaks   = rate_labels,
      labels   = rate_labels,
      drop     = FALSE,
      na.value = "grey80"
    ) +
    labs(title = paste0("Incidence Rate (% of U5 population) — ", display_label)) +
    theme_minimal() +
    theme(axis.text = element_blank(), axis.ticks = element_blank(),
          axis.title = element_blank(), panel.grid = element_blank())
  p_rate <- .add_geo_orientation_layers(p_rate)
  ggsave(paste0("plots/incidence_rate_map_", model_label, ".jpg"), p_rate, width = 10, height = 8, dpi = 300)
  message("Saved: plots/incidence_rate_map_", model_label, ".jpg")

  # 3. Burden map (faceted by timepoint)
  b_join <- burden_sum %>% dplyr::left_join(name_map, by = "District")
  map_bur <- somalia_map %>%
    dplyr::select(Shapefile_District_Name, geometry) %>%
    dplyr::right_join(b_join, by = "Shapefile_District_Name") %>%
    sf::st_as_sf() %>%
    dplyr::mutate(
      burden_bin = cut(burden_median,
                       breaks = burden_map_breaks,
                       labels = burden_map_labels,
                       right = FALSE, include.lowest = TRUE),
      # Explicit factor over the FULL label set so unused bins still get a
      # legend key below (same fix as the duration/incidence maps above).
      burden_bin = factor(burden_bin, levels = burden_map_labels),
      Timepoint  = factor(Timepoint, levels = sort(unique(Timepoint)))
    )
  p_bur <- ggplot(map_bur) +
    geom_sf(aes(fill = burden_bin), colour = "black", size = 0.1, show.legend = TRUE) +
    facet_wrap(~ Timepoint, ncol = 4) +
    scale_fill_manual(
      name     = "Burden Range",
      values   = burden_map_colors,
      limits   = burden_map_labels,
      breaks   = burden_map_labels,
      labels   = burden_map_labels,
      drop     = FALSE,
      na.value = "grey80"
    ) +
    labs(title = paste0("Burden per District — ", display_label)) +
    theme_minimal() +
    theme(strip.text = element_text(size = 7), axis.text = element_blank(),
          axis.ticks = element_blank(), axis.title = element_blank(),
          panel.grid = element_blank())
  ggsave(paste0("plots/burden_map_", model_label, ".jpg"), p_bur,
         width = 16, height = 10, dpi = 300)
  message("Saved: plots/burden_map_", model_label, ".jpg")
}

# Observed prevalence map (SAM data, model-independent, faceted by timepoint)
prev_data <- df_model %>%
  dplyr::select(District, Timepoint, SAM) %>%
  dplyr::left_join(name_map, by = "District")

map_prev <- somalia_map %>%
  dplyr::select(Shapefile_District_Name, geometry) %>%
  dplyr::right_join(prev_data, by = "Shapefile_District_Name") %>%
  sf::st_as_sf() %>%
  dplyr::mutate(
    SAM_bin   = cut(SAM,
                    breaks = c(0, 500, 1000, 2000, 3000, 5000, 7500, 10000, Inf),
                    labels = c("0-500","500-1k","1k-2k","2k-3k","3k-5k","5k-7.5k","7.5k-10k",">10k"),
                    right = FALSE, include.lowest = TRUE),
    Timepoint = factor(Timepoint, levels = sort(unique(Timepoint)))
  )

p_prevalence <- ggplot(map_prev) +
  geom_sf(aes(fill = SAM_bin), color = "black", size = 0.1) +
  facet_wrap(~ Timepoint, ncol = 4) +
  scale_fill_manual(values = green_palette(8), na.value = "grey80", drop = FALSE) +
  labs(title = "Observed Prevalence (SAM count) by District and Timepoint",
       fill = "SAM Count") +
  theme_minimal() +
  theme(strip.text = element_text(size = 7), axis.text = element_blank(),
        axis.ticks = element_blank(), panel.grid = element_blank())
ggsave("plots/prevalence_map_observed.jpg", p_prevalence, width = 16, height = 10, dpi = 300)
message("Saved: plots/prevalence_map_observed.jpg")

# ---- Benchmark Burden Map (SAM × (1 + K), K = 1.6, faceted by timepoint) ----
# Consistent with K_expected = burden/SAM - 1 used elsewhere: burden = SAM * (1 + K),
# not SAM * K.
benchmark_K <- 1.6
benchmark_burden_data <- df_model %>%
  dplyr::select(District, Timepoint, SAM) %>%
  dplyr::mutate(burden_benchmark = SAM * (1 + benchmark_K)) %>%
  dplyr::left_join(name_map, by = "District")

map_benchmark_burden <- somalia_map %>%
  dplyr::select(Shapefile_District_Name, geometry) %>%
  dplyr::right_join(benchmark_burden_data, by = "Shapefile_District_Name") %>%
  sf::st_as_sf() %>%
  dplyr::mutate(
    burden_bin = cut(
      burden_benchmark,
      breaks = burden_map_breaks,
      labels = burden_map_labels,
      right = FALSE, include.lowest = TRUE
    ),
    # Explicit factor over the FULL label set so unused bins still get a
    # legend key below (same fix as the per-model burden map).
    burden_bin = factor(burden_bin, levels = burden_map_labels),
    Timepoint  = factor(Timepoint, levels = sort(unique(Timepoint)))
  )

p_benchmark_burden <- ggplot(map_benchmark_burden) +
  geom_sf(aes(fill = burden_bin), colour = "black", size = 0.1, show.legend = TRUE) +
  facet_wrap(~ Timepoint, ncol = 4) +
  scale_fill_manual(
    name     = "Burden Range",
    values   = burden_map_colors,
    limits   = burden_map_labels,
    breaks   = burden_map_labels,
    labels   = burden_map_labels,
    drop     = FALSE,
    na.value = "grey80"
  ) +
  labs(title = "Benchmark Burden per District (SAM × (1 + K), K = 1.6)") +
  theme_minimal() +
  theme(strip.text = element_text(size = 7), axis.text = element_blank(),
        axis.ticks = element_blank(), panel.grid = element_blank())
ggsave("plots/burden_map_benchmark.jpg", p_benchmark_burden, width = 16, height = 10, dpi = 300)
message("Saved: plots/burden_map_benchmark.jpg")

# Compute shared incidence scales (counts and rate) across all four models.
# Each model's own single highest district is dropped before pooling --
# whichever district that happens to be, not hardcoded to a specific name --
# so one outlier per model doesn't stretch the shared "regular" bins for
# everyone else. Any district above the resulting cap (there can be more
# than one, and it isn't always the same district across models) lands in
# the open-ended top bin, labelled as a plain range rather than naming a
# specific district.
all_param_medians <- c(
  list(map_inputs_tv$param_medians),
  lapply(LLM_NAMES, function(llm) map_inputs_by_llm[[llm]]$param_medians)
)
.pooled_cap_excl_top <- function(dfs, col) {
  vals <- unlist(lapply(dfs, function(df) {
    v <- df[[col]]
    v[-which.max(v)]
  }))
  max(vals, na.rm = TRUE)
}
gamma_common_cap      <- .pooled_cap_excl_top(all_param_medians, "gamma_count")
gamma_rate_common_cap <- .pooled_cap_excl_top(all_param_medians, "gamma")

# Generate maps for the base model and each of the three LLM-informed models
.make_tv_model_maps(map_inputs_tv$param_medians, map_inputs_tv$burden_sum, "tv_base",
                    name_map, somalia_map, duration_colors, gamma_common_cap, gamma_rate_common_cap)
for (llm in LLM_NAMES) {
  .make_tv_model_maps(map_inputs_by_llm[[llm]]$param_medians, map_inputs_by_llm[[llm]]$burden_sum,
                      paste0("tv_", llm), name_map, somalia_map, duration_colors, gamma_common_cap, gamma_rate_common_cap)
}






# ============================================================
# Burden comparison: tv_base, tv_<llm> for each of Claude/Gemini/ChatGPT, K=1.6
# ============================================================

# Helper: use in-memory burden summary or reconstruct from saved CSV
.load_burden_summary <- function(obj_name, csv_path) {
  if (exists(obj_name) && !is.null(get(obj_name))) {
    return(get(obj_name))
  }
  message(obj_name, " not in memory — loading from ", csv_path)
  df_csv <- read_csv(csv_path, show_col_types = FALSE)
  df_csv %>%
    select(District, starts_with("burden_")) %>%
    pivot_longer(-District, names_to = "col", values_to = "value") %>%
    mutate(
      stat = case_when(
        str_starts(col, "burden_lo_") ~ "burden_lower",
        str_starts(col, "burden_hi_") ~ "burden_upper",
        TRUE                          ~ "burden_median"
      ),
      Timepoint = case_when(
        str_starts(col, "burden_lo_") ~ str_remove(col, "^burden_lo_"),
        str_starts(col, "burden_hi_") ~ str_remove(col, "^burden_hi_"),
        TRUE                          ~ str_remove(col, "^burden_")
      )
    ) %>%
    select(-col) %>%
    pivot_wider(names_from = stat, values_from = value)
}

bs_tv <- .load_burden_summary(
  "burden_summary_tv",
  "outputs/parameter_and_burden_summary_theoretical_variance.csv"
)

bs_by_llm <- list()
for (llm in LLM_NAMES) {
  if (exists("burden_summary_by_llm") && !is.null(burden_summary_by_llm[[llm]])) {
    bs_by_llm[[llm]] <- burden_summary_by_llm[[llm]]
  } else {
    bs_by_llm[[llm]] <- .load_burden_summary(
      paste0("burden_summary_tv_", llm),
      paste0("outputs/parameter_and_burden_summary_tv_", llm, ".csv")
    )
  }
}

# Long-format: one row per (Model, District, Timepoint)
burden_compare_long <- bind_rows(
  bs_tv %>% mutate(Model = "hierarchical"),
  bind_rows(lapply(LLM_NAMES, function(llm) bs_by_llm[[llm]] %>% mutate(Model = paste0("llm_", llm)))),
  df_model %>%
    select(District, Timepoint, SAM) %>%
    mutate(
      Model         = "k1.6",
      burden_median = SAM * (1 + benchmark_K),
      burden_lower  = NA_real_,
      burden_upper  = NA_real_
    )
) %>%
  select(Model, District, Timepoint, burden_median, burden_lower, burden_upper) %>%
  arrange(District, Timepoint, Model)

write_csv(burden_compare_long, "outputs/burden_comparison_all_models.csv")
message("Saved: outputs/burden_comparison_all_models.csv")

# Wide-format: one row per (District, Timepoint), models side by side with differences
burden_compare_wide <- burden_compare_long %>%
  select(District, Timepoint, Model, burden_median) %>%
  pivot_wider(names_from = Model, values_from = burden_median) %>%
  mutate(diff_hierarchical_vs_k16 = hierarchical - k1.6)

for (llm in LLM_NAMES) {
  col <- paste0("llm_", llm)
  burden_compare_wide[[paste0("diff_", col, "_vs_k16")]]        <- burden_compare_wide[[col]] - burden_compare_wide[["k1.6"]]
  burden_compare_wide[[paste0("diff_hierarchical_vs_", col)]]   <- burden_compare_wide[["hierarchical"]] - burden_compare_wide[[col]]
}

burden_compare_wide <- burden_compare_wide %>% arrange(District, Timepoint)

write_csv(burden_compare_wide, "outputs/burden_comparison_wide.csv")
message("Saved: outputs/burden_comparison_wide.csv")

# Summary: mean burden per district per model (averaged across timepoints)
burden_compare_summary <- burden_compare_long %>%
  group_by(Model, District) %>%
  summarise(
    burden_mean  = mean(burden_median, na.rm = TRUE),
    burden_total = sum(burden_median,  na.rm = TRUE),
    .groups = "drop"
  )

write_csv(burden_compare_summary, "outputs/burden_comparison_summary_by_district.csv")
message("Saved: outputs/burden_comparison_summary_by_district.csv")

# Global summary: grand mean and total across all districts and timepoints
burden_compare_global <- burden_compare_long %>%
  group_by(Model) %>%
  summarise(
    burden_grand_mean  = mean(burden_median, na.rm = TRUE),
    burden_grand_total = sum(burden_median,  na.rm = TRUE),
    .groups = "drop"
  )
print(burden_compare_global)
write_csv(burden_compare_global, "outputs/burden_comparison_global_summary.csv")
message("Saved: outputs/burden_comparison_global_summary.csv")

# ============================================================
# End of burden comparison
# ============================================================

# -------------------------------------------------
# ------- chain and model convergence stats -------
# -------------------------------------------------
posterior_array_tv <- as.array(fit_tv)

# Trace plot
trace_plot <- mcmc_trace(posterior_array_tv, pars = c("p[1]", "gamma[1]", "omega[1]", "beta0[1]"))
ggsave("plots/trace_plot_p_gamma_omega_beta0.jpg", trace_plot, width = 12, height = 6, dpi = 300)

# Density overlays: see if chains produce consistent posteriors
density_plot <- mcmc_dens_overlay(posterior_array_tv, pars = c("p[1]", "gamma[1]", "omega[1]", "beta0[1]"))
ggsave("plots/density_overlay_p_gamma_omega_beta0.jpg", density_plot, width = 8, height = 6, dpi = 300)

# R-hat Values (Gelman-Rubin Diagnostic)
param_summary_tv <- summary(fit_tv)$summary
rhat_values <- param_summary_tv[, "Rhat"]
ess_values  <- param_summary_tv[, "n_eff"]

rhat_df_tv <- data.frame(
  Parameter = rownames(param_summary_tv),
  Rhat  = param_summary_tv[, "Rhat"],
  n_eff = param_summary_tv[, "n_eff"]
)
write_csv(rhat_df_tv, "outputs/rhat_and_ess_tv_base.csv")

# --------- Parameter Shrinkage Plot: gamma[d] vs cross-district mean ---------
gamma_district_medians <- apply(posterior_tv$gamma, 2, median)
gamma_global_median    <- mean(gamma_district_medians)

# Create data frame for plotting
gamma_df <- data.frame(
  District = district_names,
  gamma_district = gamma_district_medians,
  gamma_global = gamma_global_median
)

# Plot
p_gamma_shrinkage <- ggplot(gamma_df, aes(x = reorder(District, gamma_district), y = gamma_district)) +
  geom_point(color = "steelblue", size = 2) +
  geom_hline(aes(yintercept = gamma_global), color = "red", linetype = "dashed") +
  coord_flip() +
  labs(
    title = "District-level gamma vs Global Mean (Shrinkage Effect)",
    x = "District",
    y = "gamma"
  ) +
  theme_minimal()

# Save
ggsave("plots/gamma_shrinkage_plot.jpg", plot = p_gamma_shrinkage, width = 10, height = 8, dpi = 300)

# --------- Shrinkage Plot: p[d] vs cross-district mean ---------
p_district_medians <- apply(posterior_tv$p, 2, median)
p_global_median    <- mean(p_district_medians)

p_df <- data.frame(
  District = district_names,
  p_district = p_district_medians,
  p_global = p_global_median
)

p_p_shrinkage <- ggplot(p_df, aes(x = reorder(District, p_district), y = p_district)) +
  geom_point(color = "darkgreen", size = 2) +
  geom_hline(aes(yintercept = p_global), color = "red", linetype = "dashed") +
  coord_flip() +
  labs(
    title = "District-level p vs Global Mean (Shrinkage Effect)",
    x = "District",
    y = "p"
  ) +
  theme_minimal()

ggsave("plots/p_shrinkage_plot.jpg", plot = p_p_shrinkage, width = 10, height = 8, dpi = 300)

# --------- Shrinkage Plot: omega[d] vs cross-district mean ---------
omega_district_medians <- apply(posterior_tv$omega, 2, median)
omega_global_median    <- mean(omega_district_medians)

omega_df <- data.frame(
  District = district_names,
  omega_district = omega_district_medians,
  omega_global = omega_global_median
)

p_omega_shrinkage <- ggplot(omega_df, aes(x = reorder(District, omega_district), y = omega_district)) +
  geom_point(color = "darkorange", size = 2) +
  geom_hline(aes(yintercept = omega_global), color = "red", linetype = "dashed") +
  coord_flip() +
  labs(
    title = "District-level omega vs Global Mean (Shrinkage Effect)",
    x = "District",
    y = "omega"
  ) +
  theme_minimal()

ggsave("plots/omega_shrinkage_plot.jpg", plot = p_omega_shrinkage, width = 10, height = 8, dpi = 300)

# ------- End of chain and model convergence stats -------
# ============================================================
# Theoretical variance model - Daily (sam_theoretical_variance_daily.stan)
# Same data as base model; p and gamma are daily rates, 180-day (6-month)
# arrival window matching the semi-annual observation interval.
# Output: outputs/parameter_and_burden_summary_tv_daily.csv
# ============================================================
stan_data_tvd <- list(
  N            = N,
  D            = D,
  district_id  = df_model$district_id,
  y            = df_model$SAM,
  pop_u5_ref   = as.vector(pop_u5_ref),
  start_index  = district_starts,
  # Beta(3,330): duration median ~4.14 months, central 95% interval ~1.54-17.85
  # months (1/(30*p_global), 30-day months) -- an approximate duration-based
  # calibration against the monthly Beta(2,5) duration prior (median 3.78,
  # 95% CI 1.56-23.11 months), NOT an exact distributional transform of it.
  p_hyperprior     = TV_DAILY_HYPERPRIORS$p,
  # Beta(3,6000): mean ~0.049975% of U5 pop/day, central 95% interval
  # ~0.01031%-0.12032%; x30 gives ~0.3093%-3.6096%, close to but not identical
  # to the monthly Beta(3,200) 95% interval (~0.3073%-3.5306%). Matches the
  # monthly gamma_hyperprior shape numerically (mean-arrival scaling,
  # gamma_month = 30*gamma_day), NOT a distributional rescaling proof.
  gamma_hyperprior = TV_DAILY_HYPERPRIORS$gamma,
  # Beta(3,6000): same shapes as gamma_hyperprior. This is NOT a preservation
  # of the monthly arrival-SD scaling relation (omega_month = sqrt(30)*omega_day);
  # using the same (3,6000) shapes for omega as for gamma does not, by itself,
  # keep the implied monthly-equivalent process variance unchanged -- see the
  # final review's discussion of the daily omega scaling issue.
  omega_hyperprior = TV_DAILY_HYPERPRIORS$omega
)

t0_fit <- Sys.time()
fit_tvd <- stan(
  file   = "sam_theoretical_variance_daily.stan",
  data   = stan_data_tvd,
  chains = 4,
  warmup = 5000,
  iter   = 10000,
  cores  = 4,
  seed   = 123
)
message("Model fit time [tv_daily]: ", round(as.numeric(difftime(Sys.time(), t0_fit, units = "mins")), 2), " minutes")

posterior_tvd <- rstan::extract(fit_tvd)
print(fit_tvd)
.report_hmc_diagnostics(fit_tvd, "tv_daily (full data)")

log_lik_tvd <- rstan::extract(fit_tvd, pars = "log_lik")$log_lik
loo_tvd     <- loo::loo(log_lik_tvd)
print(loo_tvd)

# --- Build burden summary ---
burden_draws_tvd    <- as.matrix(posterior_tvd$burden)
district_labels_tvd <- district_names[df_model$district_id]
burden_long_tvd     <- as.data.frame(burden_draws_tvd)
burden_long_tvd$iteration <- seq_len(nrow(burden_long_tvd))
burden_summary_tvd <- burden_long_tvd |>
  tidyr::pivot_longer(-iteration, names_to = "col", values_to = "burden") |>
  mutate(
    i         = as.integer(sub("^V", "", col)),
    District  = district_labels_tvd[i],
    Timepoint = df_model$Timepoint[i]
  ) |>
  dplyr::group_by(District, Timepoint) |>
  dplyr::summarise(
    burden_median = median(burden, na.rm = TRUE),
    burden_lower  = quantile(burden, 0.025, na.rm = TRUE),
    burden_upper  = quantile(burden, 0.975, na.rm = TRUE),
    .groups = "drop"
  )

# --- Per-district median parameters ---
sd_pred_draws_tvd      <- as.matrix(posterior_tvd$sd_pred)   # iterations x N
sd_pred_median_by_obs_tvd <- apply(sd_pred_draws_tvd, 2, median)

alpha_equiv_tvd <- tapply(
  sd_pred_median_by_obs_tvd,
  df_model$district_id,
  median
)

param_medians_tvd <- tibble(
  District             = district_names,
  beta0                = apply(posterior_tvd$beta0,                2, median),
  beta1                = apply(posterior_tvd$beta1,                2, median),
  p                    = apply(posterior_tvd$p,                    2, median),
  gamma                = apply(posterior_tvd$gamma,                2, median),
  gamma_count          = apply(posterior_tvd$gamma_count_out,      2, median),
  omega                = apply(posterior_tvd$omega,                2, median),   # fraction
  omega_count          = apply(posterior_tvd$omega_count_out,      2, median),   # in counts
  alpha_equiv          = as.numeric(alpha_equiv_tvd),
  mean_duration        = apply(posterior_tvd$mean_duration,        2, median),   # 1/p in days
  mean_duration_months = apply(posterior_tvd$mean_duration_months, 2, median),   # 1/p in months
  kappa_p              = median(posterior_tvd$kappa_p),
  kappa_gamma          = median(posterior_tvd$kappa_gamma),
  kappa_omega          = median(posterior_tvd$kappa_omega)
)

burden_wide_tvd <- burden_summary_tvd %>%
  select(District, Timepoint, burden_median) %>%
  pivot_wider(
    names_from   = Timepoint,
    values_from  = burden_median,
    names_prefix = "burden_"
  )

final_summary_tvd <- param_medians_tvd %>%
  left_join(burden_wide_tvd, by = "District")

write_csv(final_summary_tvd, "outputs/parameter_and_burden_summary_tv_daily.csv")
# ============================================================
# End of theoretical variance daily model
# ============================================================


# ============================================================
# ---- Mean Benchmark for LL2O-CV ----
# Option A: y8 predicted by mean(y1..y7),
#           y9 predicted by mean(y1..y8)  [symmetric: Bayesian also sees true y8]
# Runs standalone — loads llo2_all_df from CSV if not in memory.
# ============================================================

# --- Load LL2O results from memory or CSV ---
if (!exists("llo2_all_df") || is.null(llo2_all_df)) {
  message("llo2_all_df not in memory — loading from CSV.")
  llo2_all_df <- read_csv("outputs/LLO2_CV_all_models_with_CRPS.csv",
                          show_col_types = FALSE)
}

# --- Reconstruct train/test split from df_model (always available) ---
df_train_mb <- df_model %>% group_by(district_id) %>% slice_head(n = 7) %>% ungroup()

# --- Compute mean benchmark predictions per district ---
mean_bench_rows <- lapply(unique(df_train_mb$district_id), function(d) {
  y_train   <- df_train_mb %>% filter(district_id == d) %>% pull(SAM)
  y8_act    <- df_model    %>% filter(district_id == d) %>% slice(8) %>% pull(SAM)
  y9_act    <- df_model    %>% filter(district_id == d) %>% slice(9) %>% pull(SAM)
  dist_name <- district_names[d]

  pred_8 <- mean(y_train)             # mean(y1..y7)
  pred_9 <- mean(c(y_train, y8_act))  # mean(y1..y8) — symmetric with Bayesian seeing y8

  tibble(
    Model     = "MeanBenchmark",
    District  = dist_name,
    Point     = c("8th", "9th"),
    Predicted = c(pred_8, pred_9),
    Observed  = c(y8_act, y9_act),
    AbsError  = abs(c(pred_8, pred_9) - c(y8_act, y9_act)),
    SqError   = (c(pred_8, pred_9) - c(y8_act, y9_act))^2,
    MAPE      = ifelse(c(y8_act, y9_act) == 0, NA_real_,
                        100 * abs(c(pred_8, pred_9) - c(y8_act, y9_act)) / c(y8_act, y9_act)),
    CRPS      = NA_real_
  )
})
mean_bench_df <- bind_rows(mean_bench_rows)

# --- Merge with existing results (drop any prior MeanBenchmark to avoid duplicates) ---
llo2_all_df_ext <- bind_rows(
  llo2_all_df %>% filter(Model != "MeanBenchmark"),
  mean_bench_df
)
write_csv(llo2_all_df_ext, "outputs/LLO2_CV_all_models_with_CRPS.csv")
message("Saved: outputs/LLO2_CV_all_models_with_CRPS.csv (MeanBenchmark added)")

# --- Updated summary table ---
llo2_summary_ext <- llo2_all_df_ext %>%
  group_by(Model, Point) %>%
  summarise(
    RMSE      = sqrt(mean(SqError, na.rm = TRUE)),  # aggregate RMSE = sqrt(mean squared error), NOT mean(per-row abs error)
    MAE       = mean(AbsError, na.rm = TRUE),        # what the old "RMSE_mean" column actually computed
    MAPE_mean = mean(MAPE, na.rm = TRUE),
    CRPS_mean = mean(CRPS, na.rm = TRUE),
    .groups   = "drop"
  )
print(llo2_summary_ext)
write_csv(llo2_summary_ext, "outputs/LLO2_CV_summary_by_model.csv")
message("Saved: outputs/LLO2_CV_summary_by_model.csv (updated)")

# --- RMSE bar plot (all models + both benchmarks) ---
llo2_plot_colors <- c(TV_MODEL_COLORS, Naive = "grey50", MeanBenchmark = "darkgreen")
llo2_plot_labels <- c(TV_MODEL_DISPLAY, Naive = "Naive (P_t = P_{t-6})", MeanBenchmark = "Mean Benchmark")

p_llo2_rmse_ext <- ggplot(
  llo2_all_df_ext,
  aes(x = reorder(District, -AbsError), y = AbsError, fill = Model)
) +
  geom_bar(stat = "identity", position = "dodge") +
  facet_wrap(~Point, nrow = 2, scales = "free_x") +
  coord_flip() +
  scale_fill_manual(values = llo2_plot_colors, labels = llo2_plot_labels) +
  labs(title = "Leave-Last-2-Out CV: Absolute Error by District — All Models",
       x = "District", y = "Absolute Error") +
  theme_minimal() +
  theme(strip.text = element_text(size = 10), axis.text.y = element_text(size = 7))
ggsave("plots/LLO2_CV_RMSE_all_models.jpg", p_llo2_rmse_ext,
       width = 14, height = 16, dpi = 300)
message("Saved: plots/LLO2_CV_RMSE_all_models.jpg")

# ============================================================
# ---- Mean Benchmark for LL3O-CV ----
# Same idea as the LL2O-CV mean benchmark above, extended to three held-out
# points: 7th predicted by mean(y1..y6), 8th by mean(y1..y7), 9th by
# mean(y1..y8) -- each prediction uses the expanding window of all TRUE
# observations up to (not including) the point being predicted, symmetric
# with how the Bayesian/Naive LL3O-CV predictions condition on the true
# previous observation rather than a recursive prediction.
# Runs standalone — loads llo3_all_df from CSV if not in memory.
# ============================================================

# --- Load LL3O results from memory or CSV ---
if (!exists("llo3_all_df") || is.null(llo3_all_df)) {
  message("llo3_all_df not in memory — loading from CSV.")
  llo3_all_df <- read_csv("outputs/LLO3_CV_all_models_with_CRPS.csv",
                          show_col_types = FALSE)
}

# --- Reconstruct train/test split from df_model (always available) ---
df_train_mb3 <- df_model %>% group_by(district_id) %>% slice_head(n = 6) %>% ungroup()

# --- Compute mean benchmark predictions per district ---
mean_bench3_rows <- lapply(unique(df_train_mb3$district_id), function(d) {
  y_train   <- df_train_mb3 %>% filter(district_id == d) %>% pull(SAM)
  y7_act    <- df_model     %>% filter(district_id == d) %>% slice(7) %>% pull(SAM)
  y8_act    <- df_model     %>% filter(district_id == d) %>% slice(8) %>% pull(SAM)
  y9_act    <- df_model     %>% filter(district_id == d) %>% slice(9) %>% pull(SAM)
  dist_name <- district_names[d]

  pred_7 <- mean(y_train)                    # mean(y1..y6)
  pred_8 <- mean(c(y_train, y7_act))          # mean(y1..y7) — sees true y7, like the Bayesian/Naive models
  pred_9 <- mean(c(y_train, y7_act, y8_act))  # mean(y1..y8) — sees true y7 and y8

  tibble(
    Model     = "MeanBenchmark",
    District  = dist_name,
    Point     = c("7th", "8th", "9th"),
    Predicted = c(pred_7, pred_8, pred_9),
    Observed  = c(y7_act, y8_act, y9_act),
    AbsError  = abs(c(pred_7, pred_8, pred_9) - c(y7_act, y8_act, y9_act)),
    SqError   = (c(pred_7, pred_8, pred_9) - c(y7_act, y8_act, y9_act))^2,
    MAPE      = ifelse(c(y7_act, y8_act, y9_act) == 0, NA_real_,
                        100 * abs(c(pred_7, pred_8, pred_9) - c(y7_act, y8_act, y9_act)) / c(y7_act, y8_act, y9_act)),
    CRPS      = NA_real_
  )
})
mean_bench3_df <- bind_rows(mean_bench3_rows)

# --- Merge with existing results (drop any prior MeanBenchmark to avoid duplicates) ---
llo3_all_df_ext <- bind_rows(
  llo3_all_df %>% filter(Model != "MeanBenchmark"),
  mean_bench3_df
)
write_csv(llo3_all_df_ext, "outputs/LLO3_CV_all_models_with_CRPS.csv")
message("Saved: outputs/LLO3_CV_all_models_with_CRPS.csv (MeanBenchmark added)")

# --- Updated summary table ---
llo3_summary_ext <- llo3_all_df_ext %>%
  group_by(Model, Point) %>%
  summarise(
    RMSE      = sqrt(mean(SqError, na.rm = TRUE)),
    MAE       = mean(AbsError, na.rm = TRUE),
    MAPE_mean = mean(MAPE, na.rm = TRUE),
    CRPS_mean = mean(CRPS, na.rm = TRUE),
    .groups   = "drop"
  )
print(llo3_summary_ext)
write_csv(llo3_summary_ext, "outputs/LLO3_CV_summary_by_model.csv")
message("Saved: outputs/LLO3_CV_summary_by_model.csv (updated)")

# --- Absolute-error bar plot (all models + both benchmarks) ---
llo3_plot_colors <- c(TV_MODEL_COLORS, Naive = "grey50", MeanBenchmark = "darkgreen")
llo3_plot_labels <- c(TV_MODEL_DISPLAY, Naive = "Naive (P_t = P_{t-6})", MeanBenchmark = "Mean Benchmark")

p_llo3_rmse_ext <- ggplot(
  llo3_all_df_ext,
  aes(x = reorder(District, -AbsError), y = AbsError, fill = Model)
) +
  geom_bar(stat = "identity", position = "dodge") +
  facet_wrap(~Point, nrow = 3, scales = "free_x") +
  coord_flip() +
  scale_fill_manual(values = llo3_plot_colors, labels = llo3_plot_labels) +
  labs(title = "Leave-Last-3-Out CV (robustness check): Absolute Error by District — All Models",
       x = "District", y = "Absolute Error") +
  theme_minimal() +
  theme(strip.text = element_text(size = 10), axis.text.y = element_text(size = 7))
ggsave("plots/LLO3_CV_RMSE_all_models.jpg", p_llo3_rmse_ext,
       width = 14, height = 20, dpi = 300)
message("Saved: plots/LLO3_CV_RMSE_all_models.jpg")


# ============================================================
# ---- Relative and population-weighted CV scores ----
# Raw RMSE averages districts whose caseloads differ by orders of magnitude, so
# the districts with the largest SAM counts dominate it. Two additions:
#
# 1) Relative RMSE vs a benchmark, per district d:
#        Rel_d = RMSE_d(model) / RMSE_d(benchmark)
#    where RMSE_d is taken over that district's held-out points (a single
#    point gives |error|). Rel_d is scale-free, so it can be averaged across
#    districts; < 1 means the model beats the benchmark there. Benchmarks:
#    "HistMean"    = MeanBenchmark (expanding-window historical mean)
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

CV_REL_BENCHMARKS <- c(HistMean = "MeanBenchmark", Persistence = "Naive")

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
    summarise(RMSE_d = sqrt(mean(SqError, na.rm = TRUE)), .groups = "drop")

  base <- long %>%
    group_by(Model, Point) %>%
    summarise(
      n_districts      = n_distinct(District),
      RMSE             = sqrt(mean(SqError, na.rm = TRUE)),
      wRMSE_pop        = sqrt(weighted.mean(SqError, Pop, na.rm = TRUE)),
      eff_n_districts  = (sum(Pop[!is.na(SqError)])^2) / sum(Pop[!is.na(SqError)]^2),
      .groups = "drop"
    )

  by_district <- list()
  out <- base
  for (b in names(benchmarks)) {
    ref <- dist_rmse %>%
      filter(Model == benchmarks[[b]]) %>%
      select(Point, District, RMSE_ref = RMSE_d)
    rel <- dist_rmse %>%
      left_join(ref, by = c("Point", "District")) %>%
      mutate(Benchmark = b,
             Rel = ifelse(!is.na(RMSE_ref) & RMSE_ref > 0, RMSE_d / RMSE_ref, NA_real_))
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
    out <- left_join(out, summ, by = c("Model", "Point"))
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
                              labels = paste0("vs ", c("Historical Mean", "Persistence"))),
           District  = factor(District, levels = district_names))
  if (nrow(plot_df) > 0) {
    p_rel <- ggplot(plot_df, aes(x = Rel, y = reorder(District, Pop), color = Model)) +
      geom_vline(xintercept = 1, linetype = "dashed", color = "grey30") +
      geom_point(size = 1.1, alpha = 0.85) +
      facet_wrap(~Benchmark, nrow = 1) +
      scale_x_log10() +
      (if (all(unique(plot_df$Model) %in% names(TV_MODEL_COLORS)))
         scale_color_manual(values = TV_MODEL_COLORS, labels = TV_MODEL_DISPLAY, name = NULL)
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

# Run only when the corresponding MeanBenchmark section has produced its table
# (the all-runs LL2O block at the end of the file does not need them).
if (exists("llo2_all_df_ext")) {
  report_cv_relative_scores(llo2_all_df_ext, "LLO2", "Leave-Last-2-Out CV: relative RMSE by district")
} else message("Skipped LLO2 relative scores: llo2_all_df_ext not found (run the LL2O MeanBenchmark section first).")
if (exists("llo3_all_df_ext")) {
  report_cv_relative_scores(llo3_all_df_ext, "LLO3", "Leave-Last-3-Out CV: relative RMSE by district")
} else message("Skipped LLO3 relative scores: llo3_all_df_ext not found (LL3O not run).")


# ============================================================
# ---- LL2O-CV for every LLM prior run (out-of-sample comparison) ----
# Scores each run's LL2O-CV predictions (fitted in the LL2O-CV section above,
# stored in post_cv_runs) with the same predictive code, RMSE / MAPE / CRPS
# definitions as the main LL2O results, next to tv_base and the two
# benchmarks, then adds the relative / population-weighted scores.
# Models are named tv_<llm>_run<k>. Outputs:
#   outputs/LLO2_CV_all_runs_with_CRPS.csv
#   outputs/LLO2_CV_all_runs_summary.csv          (by model; per point and all points)
#   outputs/LLO2_all_runs_CV_relative_scores.csv  (+ by-district file and plot)
# ============================================================
llo2_runs_df <- bind_rows(lapply(names(post_cv_runs), function(k) {
  .llo2_score_rows(paste0("tv_", k), post_cv_runs[[k]])
}))

# Keep previously saved scores for models whose fits are no longer in memory
# (e.g. single runs scored in an earlier session, when only the pooled priors
# were fitted this time). Models scored just now take precedence.
prev_runs_csv <- "outputs/LLO2_CV_all_runs_with_CRPS.csv"
if (file.exists(prev_runs_csv)) {
  prev_runs_df <- read_csv(prev_runs_csv, show_col_types = FALSE) %>%
    filter(grepl("^tv_(claude|gemini|chatgpt)_(run[0-9]+|pooled)$", Model),
           !(Model %in% llo2_runs_df$Model)) %>%
    mutate(Point = as.character(Point))
  if (nrow(prev_runs_df) > 0) {
    message("Keeping saved scores for: ", paste(unique(prev_runs_df$Model), collapse = ", "))
    llo2_runs_df <- bind_rows(llo2_runs_df %>% mutate(Point = as.character(Point)), prev_runs_df)
  }
}

# Benchmarks: reuse llo2_all_df_ext if the MeanBenchmark section above has been
# run; otherwise take tv_base / Naive from llo2_all_df and compute the
# MeanBenchmark here (same definition: y8 <- mean(y1..y7), y9 <- mean(y1..y8)),
# so this block can be run straight after the LL2O-CV section.
bench_src <- if (exists("llo2_all_df_ext")) llo2_all_df_ext else llo2_all_df
if (!("MeanBenchmark" %in% bench_src$Model)) {
  bench_src <- bind_rows(bench_src, bind_rows(lapply(unique(df_train_cv_tv$district_id), function(d) {
    y_train <- df_train_cv_tv %>% filter(district_id == d) %>% pull(SAM)
    y8_act  <- df_model %>% filter(district_id == d) %>% slice(8) %>% pull(SAM)
    y9_act  <- df_model %>% filter(district_id == d) %>% slice(9) %>% pull(SAM)
    pred    <- c(mean(y_train), mean(c(y_train, y8_act)))
    obs     <- c(y8_act, y9_act)
    tibble(Model = "MeanBenchmark", District = district_names[d], Point = c("8th", "9th"),
           Predicted = pred, Observed = obs, AbsError = abs(pred - obs), SqError = (pred - obs)^2,
           MAPE = ifelse(obs == 0, NA_real_, 100 * abs(pred - obs) / obs), CRPS = NA_real_)
  })))
}

llo2_runs_all_df <- bind_rows(
  llo2_runs_df,
  bench_src %>% filter(Model %in% c("tv_base", "Naive", "MeanBenchmark")) %>%
    mutate(Point = as.character(Point))
)
write_csv(llo2_runs_all_df, "outputs/LLO2_CV_all_runs_with_CRPS.csv")
message("Saved: outputs/LLO2_CV_all_runs_with_CRPS.csv")

llo2_runs_summary <- bind_rows(
  llo2_runs_all_df %>% mutate(Point = as.character(Point)),
  llo2_runs_all_df %>% mutate(Point = "All")
) %>%
  group_by(Model, Point) %>%
  summarise(
    RMSE      = sqrt(mean(SqError, na.rm = TRUE)),
    MAE       = mean(AbsError, na.rm = TRUE),
    MAPE_mean = mean(MAPE, na.rm = TRUE),
    CRPS_mean = mean(CRPS, na.rm = TRUE),
    .groups   = "drop"
  ) %>%
  arrange(Point, RMSE)
print(as.data.frame(llo2_runs_summary %>% filter(Point == "All")))
write_csv(llo2_runs_summary, "outputs/LLO2_CV_all_runs_summary.csv")
message("Saved: outputs/LLO2_CV_all_runs_summary.csv")

report_cv_relative_scores(llo2_runs_all_df, "LLO2_all_runs",
                          "Leave-Last-2-Out CV, all LLM prior runs: relative RMSE by district")
