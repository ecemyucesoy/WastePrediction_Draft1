# ============================================================
# Hierarchical SAM burden model with weakly informative priors -- Somalia, 74 districts
# (Draft 1)
#
# Partial pooling across districts: the district parameters p, gamma and omega are drawn
# from a common Beta distribution whose mean (global p, gamma, omega) and concentration
# (kappa) are estimated, with weakly informative hyperpriors. No LLM information is used.
# The no-pooling model with LLM-informed priors lives in the sibling folder
# no_pooling_LLM_informed_priors/ and is compared with this model there.
#
# CONTENTS
#   0. Setup: libraries, paths, settings, helper functions
#   1. Data loading and cleaning
#   2. Outlier analysis
#   3. Hierarchical model (full data): posterior summaries, output files, prior-posterior
#      overlay, fit statistics
#   4. Chain statistics, trace plots, shrinkage plots
#   5. LLO-2 cross-validation: hierarchical model and the Naive / HistAverage benchmarks
#        (RMSE, wRMSE, MAPE, CRPS, per-district relative scores averaged over districts)
#   6. LLO-3 cross-validation: same
#   7. Daily model (hierarchical_weakly_informative_priors_daily.stan)
#   8. Maps and graphs, and the burden comparison with the K = 1.6 benchmark
#
# FILES IN THIS FOLDER
#   hierarchical_weakly_informative_priors_monthly.stan   monthly model (sections 3-6, 8)
#   hierarchical_weakly_informative_priors_daily.stan     daily model (section 7)
#   cache/<scheme>/hierarchical.rds                       cached fits (scheme = full, llo2, llo3)
#   outputs/, plots/                                      tables and figures written by this script
#
# RUNNING
#   Every Stan fit is saved to cache/ and skipped when that file exists, so running the
#   script from the top only refits what is missing. Delete a cache file (or set
#   FORCE_REFIT <- TRUE) to refit. RUN_SCHEMES limits which of "full", "llo2", "llo3"
#   may be fitted in this session; RUN_DAILY switches the daily fit on or off.
#   Model id and label in every table: "hierarchical". Benchmarks: "Naive" (previous
#   observed value) and "HistAverage" (mean of all earlier observed values).
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

library(scoringRules)
library(ggrepel)

# ============================================================
# 0. SETUP
# ============================================================
PROJECT_DIR    <- "C:/Users/ecemy/OneDrive - INSEAD/Desktop/Waste prediction/Code/Draft 1/hierarchical_weakly_informative_priors"
DATA_PATH      <- "C:/Users/ecemy/OneDrive - INSEAD/Desktop/Waste prediction/Datasets/Final_Clean_Dataset.csv"
SHAPEFILE_PATH <- "C:/Users/ecemy/OneDrive - INSEAD/Desktop/Waste prediction/Code/SOM_adm_2015_July/SOM_adm2.shp"
setwd(PROJECT_DIR)

dir.create("outputs", showWarnings = FALSE, recursive = TRUE)
dir.create("plots",   showWarnings = FALSE, recursive = TRUE)

# Per-model Rhat/ESS and fit-statistics files go here, not in outputs/ itself
DIAG_DIR <- "outputs/diagnostics"
dir.create(DIAG_DIR, showWarnings = FALSE, recursive = TRUE)

CACHE_DIR <- "cache"
for (.s in c("full", "llo2", "llo3")) {
  dir.create(file.path(CACHE_DIR, .s), showWarnings = FALSE, recursive = TRUE)
}

# ---- MCMC settings and seeds per scheme ----
MCMC  <- list(chains = 4, warmup = 5000, iter = 10000, cores = 4)
SEEDS <- c(full = 123, llo2 = 456, llo3 = 789)
if (exists("MCMC_OVERRIDE")) MCMC <- MCMC_OVERRIDE   # hook for quick code tests only

# ---- Batch controls (see the header) ----
RUN_MODELS  <- NULL
RUN_SCHEMES <- c("full", "llo2", "llo3")
RUN_DAILY   <- TRUE
FORCE_REFIT <- FALSE

# Cached full-data draws for p, gamma, omega and the district-mean burden are
# thinned (every THIN_EVERY-th draw) before saving. All summary tables, Rhat
# and ESS are computed from the full, unthinned draws before thinning.
THIN_EVERY <- 5

# ============================================================
# Shared Beta hyperprior configuration
# HIER_MONTHLY_HYPERPRIORS: Beta hyperpriors for p_global/gamma_global/omega_global
# in the monthly hierarchical model (hierarchical_weakly_informative_priors_monthly.stan). Used
# identically by the full-data fit and both CV fits below, and as the basis
# for the missing-district fallback (default_prior_ip) and the prior-posterior
# overlay (hier_priors) -- single source of truth so the three can't drift
# apart again.
# HIER_DAILY_HYPERPRIORS: Beta hyperpriors for the daily model
# (hierarchical_weakly_informative_priors_daily.stan); an approximate duration-based
# calibration, not an exact rescaling of the monthly values -- see the
# comments at the daily stan_data_tvd block below and the final review.
# ============================================================
HIER_MONTHLY_HYPERPRIORS <- list(
  p     = c(2, 5),
  gamma = c(3, 200),
  omega = c(3, 200)
)
HIER_DAILY_HYPERPRIORS <- list(
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
# Model id, fit cache and Stan interface
# The only fitted model is "hierarchical" (full data, LLO-2 and LLO-3 fits).
# ============================================================
MODEL_IDS     <- "hierarchical"
model_label   <- function(model_id) model_id    # id and label coincide
MODEL_LABELS  <- "hierarchical"
MODEL_COLORS  <- c(hierarchical = "grey40")
MODEL_DISPLAY <- c(hierarchical = "Hierarchical")

# Model ids this session may FIT (cached models are always usable regardless).
models_to_fit <- function() {
  ids <- MODEL_IDS
  if (!is.null(RUN_MODELS)) ids <- intersect(ids, RUN_MODELS)
  ids
}

cache_path  <- function(scheme, model_id) file.path(CACHE_DIR, scheme, paste0(model_id, ".rds"))
cache_has   <- function(scheme, model_id) file.exists(cache_path(scheme, model_id))
cache_read  <- function(scheme, model_id) {
  if (cache_has(scheme, model_id)) readRDS(cache_path(scheme, model_id)) else NULL
}
cache_write <- function(obj, scheme, model_id) {
  p <- cache_path(scheme, model_id)
  tmp <- paste0(p, ".tmp")
  saveRDS(obj, tmp)
  file.rename(tmp, p)   # a crash mid-write never leaves a corrupt cache file
  invisible(p)
}

# ---- Stan model: compiled on first use only, so a session that just reads the
# cache never needs the compiler (rstan caches the compiled .rds next to the .stan) ----
.stan_model_cache <- new.env()
get_stan_model <- function(model_id) {
  if (is.null(.stan_model_cache[["monthly"]])) {
    .stan_model_cache[["monthly"]] <- stan_model(file = "hierarchical_weakly_informative_priors_monthly.stan")
  }
  .stan_model_cache[["monthly"]]
}

.stan_data <- function(model_id, df_tr, starts) {
  list(N = nrow(df_tr), D = D, district_id = df_tr$district_id, y = df_tr$SAM,
       pop_u5_ref = as.vector(pop_u5_ref), start_index = starts,
       p_hyperprior     = HIER_MONTHLY_HYPERPRIORS$p,
       gamma_hyperprior = HIER_MONTHLY_HYPERPRIORS$gamma,
       omega_hyperprior = HIER_MONTHLY_HYPERPRIORS$omega)
}

# The ONLY place that calls Stan for the cached models: sampler settings, HMC
# diagnostics, draws, Rhat/ESS and the small trace array used in section 5.
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
    trace_array = as.array(fit, pars = c("p[1]", "gamma[1]", "omega[1]", "beta0[1]"))
  )
}

# ============================================================
# Full-data fits: summaries, cached object, output files
# ============================================================
.summarise_full_fit <- function(model_id, st) {
  post   <- st$post
  is_base <- (model_id == "hierarchical")

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
    model_id = model_id, prior = if (is_base) NULL else LLM_PRIORS[[model_id]],
    param_medians = param_medians, burden_summary = burden_summary,
    burden_expected = burden_expected, final_summary = final_summary, param_ci = param_ci,
    draws_thin = list(p = post$p[idx, , drop = FALSE], gamma = post$gamma[idx, , drop = FALSE],
                      omega = post$omega[idx, , drop = FALSE], burden = burden_dm[idx, , drop = FALSE]),
    rhat_df = st$rhat_df, hmc = st$hmc, fit_minutes = st$fit_minutes,
    trace_array = st$trace_array, fit_stats = fit_stats, loo = loo_obj
  )
}

# Fits (if allowed and not cached) and returns the cached full-data result, else NULL.
fit_full_model <- function(model_id) {
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

write_full_outputs <- function(model_id, full) {
  lab <- model_label(model_id)
  out_summary <- paste0("outputs/parameter_and_burden_summary_", lab, ".csv")
  write_csv(full$final_summary, out_summary)
  write_csv(full$rhat_df,       file.path(DIAG_DIR, paste0("rhat_and_ess_", lab, ".csv")))
  write_csv(full$fit_stats,     file.path(DIAG_DIR, paste0("fit_stats_", lab, ".csv")))
  message("Saved: ", out_summary, "; ", DIAG_DIR, "/rhat_and_ess_", lab, ".csv, fit_stats_", lab, ".csv")
}

# ============================================================
# Cross-validation (LLO-2 / LLO-3): fit on the training split, score held-out points
# ============================================================
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

# Fits (if allowed and not cached) and returns the cached CV result, else NULL.
fit_cv_model <- function(scheme, model_id) {
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
              hmc = st$hmc, fit_minutes = st$fit_minutes)
  cache_write(out, scheme, model_id)
  out
}

# Benchmarks, defined for any scheme: Naive predicts each held-out point by the
# previous observed value; HistAverage by the mean of ALL earlier observed values.
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
  }))
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
# 1. DATA LOADING AND CLEANING
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
# 2. OUTLIER ANALYSIS
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
# 3. HIERARCHICAL MODEL (full data) -- hierarchical_weakly_informative_priors_monthly.stan
# Theoretical variance model: omega is a fraction in (0,1) with a Beta hierarchical
# prior, symmetric to gamma. p_hyperprior / gamma_hyperprior / omega_hyperprior are the
# Beta shape parameters of the global means.
#   3.1 posterior mean / median / CI, output files, prior-posterior overlay
#   3.2 fit statistics file (outputs/diagnostics/fit_stats_hierarchical.csv)
# The fit is cached in cache/full/hierarchical.rds and skipped when present.
# ============================================================
FULL <- list()
hier_full <- fit_full_model("hierarchical")
if (!is.null(hier_full)) FULL[["hierarchical"]] <- hier_full

# ---- Prior-posterior overlay ----
# Dashed = prior, solid = posterior, one panel per district.
# Hyperpriors on the GLOBAL means (the same for all districts); the district-level
# marginal prior implied by the hierarchical model is simulated below.
hier_priors <- list(
  p     = c(a = HIER_MONTHLY_HYPERPRIORS$p[1],     b = HIER_MONTHLY_HYPERPRIORS$p[2]),
  gamma = c(a = HIER_MONTHLY_HYPERPRIORS$gamma[1], b = HIER_MONTHLY_HYPERPRIORS$gamma[2]),
  omega = c(a = HIER_MONTHLY_HYPERPRIORS$omega[1], b = HIER_MONTHLY_HYPERPRIORS$omega[2])
)

# ---- Simulate the ACTUAL hierarchical marginal prior for a district parameter ----
# Beta(2,5)/Beta(3,200) etc. above are hyperpriors on the GLOBAL mean only.
# The district-level marginal prior implied by the hierarchical model is: global mean
# ~ Beta hyperprior, concentration kappa ~ Student-t(3,0,10) truncated to (0, Inf)
# (Stan's <lower=1e-10> constraint on kappa_p/kappa_gamma/kappa_omega; the Student-t is
# symmetric about 0, so truncating at 0 is equivalent to abs() of an unconstrained draw),
# then district parameter ~ Beta(mean*kappa, (1-mean)*kappa). All districts are
# exchangeable under this specification, so the marginal is identical across districts
# and only needs to be simulated once per parameter. A fixed seed is used and the
# caller's RNG stream is restored on exit, so reproducibility of unrelated random draws
# elsewhere in the script (e.g. the CV posterior-predictive rnorm() calls) is untouched.
.hier_prior_draws <- function(param_name, n_draws = 20000, seed = 20260101) {
  old_seed <- if (exists(".Random.seed", envir = .GlobalEnv)) get(".Random.seed", envir = .GlobalEnv) else NULL
  on.exit({
    if (!is.null(old_seed)) assign(".Random.seed", old_seed, envir = .GlobalEnv)
    else if (exists(".Random.seed", envir = .GlobalEnv)) rm(".Random.seed", envir = .GlobalEnv)
  })
  set.seed(seed)

  hp <- hier_priors[[param_name]]
  global_mean <- rbeta(n_draws, hp["a"], hp["b"])
  kappa       <- pmax(abs(rt(n_draws, df = 3)) * 10, 1e-10)   # folded/truncated-at-0 Student-t(3,0,10)
  a_d <- pmax(global_mean * kappa, 1e-10)
  b_d <- pmax((1 - global_mean) * kappa, 1e-10)
  rbeta(n_draws, a_d, b_d)
}

plot_prior_post_hierarchical <- function(param_name,
                                         log_transform = FALSE,
                                         x_range       = NULL,
                                         x_n           = 300,
                                         dens_cap_q    = 0.97,
                                         output_dir    = "plots") {
  dir.create(output_dir, showWarnings = FALSE)
  prior_draws <- .hier_prior_draws(param_name)

  # x-range: union of the posterior draws and the simulated prior, so the dashed
  # prior curve is never clipped to a posterior-only display window
  all_samp <- c(as.vector(posterior_hier[[param_name]]), prior_draws)
  if (log_transform) all_samp <- log(all_samp[all_samp > 0])
  all_samp <- all_samp[is.finite(all_samp)]
  if (is.null(x_range)) x_range <- quantile(all_samp, c(0.002, 0.998), na.rm = TRUE)
  if (!log_transform) x_range[1] <- max(x_range[1], 1e-4)
  x_grid <- seq(x_range[1], x_range[2], length.out = x_n)

  # prior as a KDE over the simulated hierarchical-prior draws (same treatment as the posterior)
  prior_x <- if (log_transform) log(prior_draws[prior_draws > 0]) else prior_draws
  prior_x <- prior_x[is.finite(prior_x)]
  prior_kd  <- density(prior_x, from = x_range[1], to = x_range[2], n = x_n)
  prior_tbl <- tibble(x = prior_kd$x, density = prior_kd$y)

  m <- "hierarchical"
  row_list <- list()
  for (di in seq_along(district_names)) {
    dn <- district_names[di]
    row_list[[paste(dn, "Prior")]] <- prior_tbl %>% mutate(Model = m, Type = "Prior", District = dn)
    samp <- posterior_hier[[param_name]][, di]
    if (log_transform) samp <- log(samp[samp > 0])
    samp <- samp[is.finite(samp)]
    if (length(samp) >= 20) {
      kd <- density(samp, from = x_range[1], to = x_range[2], n = x_n)
      row_list[[paste(dn, "Posterior")]] <- tibble(x = kd$x, density = kd$y, Model = m, Type = "Posterior", District = dn)
    }
  }

  plot_df <- bind_rows(row_list) %>%
    filter(!is.na(density)) %>%
    mutate(Type = factor(Type, levels = c("Prior", "Posterior")),
           District = factor(District, levels = district_names))
  dens_cap <- quantile(plot_df$density, dens_cap_q, na.rm = TRUE)
  plot_df  <- plot_df %>% mutate(density = pmin(density, dens_cap))
  x_lab <- if (log_transform) paste0("log(", param_name, ")") else param_name

  p_out <- ggplot(plot_df, aes(x = x, y = density, linetype = Type, group = Type)) +
    geom_line(linewidth = 0.5, alpha = 0.9, color = MODEL_COLORS[["hierarchical"]]) +
    facet_wrap(~District, scales = "free_y", ncol = 9) +
    scale_linetype_manual(values = c(Prior = "dashed", Posterior = "solid")) +
    labs(
      title    = paste0("Prior vs Posterior: ", param_name, if (log_transform) "  [log scale]" else "", "  \u2014  Hierarchical model"),
      subtitle = paste0("Dashed = prior  |  Solid = posterior  |  the prior is the simulated district-level marginal ",
                        "(not the p_global/gamma_global/omega_global hyperprior)  |  ",
                        "density capped at the ", round(dens_cap_q * 100), "th percentile across all curves for display"),
      x = x_lab, y = "Density"
    ) +
    theme_minimal(base_size = 7) +
    theme(strip.text = element_text(size = 5), axis.text = element_text(size = 4),
          axis.title = element_text(size = 6), legend.position = "bottom",
          legend.text = element_text(size = 7), panel.grid.minor = element_blank())

  suffix   <- if (log_transform) paste0(param_name, "_logscale") else param_name
  out_path <- file.path(output_dir, paste0("prior_posterior_", suffix, "_hierarchical.jpg"))
  ggsave(out_path, p_out, width = 22, height = 20, dpi = 250)
  message("Saved: ", out_path)
  invisible(p_out)
}

# ---- 3.1 / 3.2 outputs ----
if (!is.null(hier_full)) {
  write_full_outputs("hierarchical", hier_full)   # parameter_and_burden_summary_hierarchical.csv, diagnostics/rhat_and_ess_hierarchical.csv, diagnostics/fit_stats_hierarchical.csv
  print(as.data.frame(hier_full$fit_stats))

  # objects reused by later sections
  posterior_hier      <- hier_full$draws_thin
  param_medians_hier  <- hier_full$param_medians
  burden_summary_hier <- hier_full$burden_summary

  plot_prior_post_hierarchical("p",     log_transform = FALSE, x_range = c(1e-4, 1))
  plot_prior_post_hierarchical("gamma", log_transform = TRUE)
  plot_prior_post_hierarchical("omega", log_transform = TRUE)
} else {
  message("Hierarchical full-data fit is not cached and is not scheduled (RUN_SCHEMES) -- section 3 outputs skipped.")
}
# ============================================================
# End of the hierarchical model (full data)
# ============================================================


# -------------------------------------------------
# 4. CHAIN STATISTICS, TRACE PLOTS, SHRINKAGE PLOTS
# -------------------------------------------------
# One table row per scheme (full data, LLO-2, LLO-3) from the cache: divergences,
# max-treedepth hits, low-E-BFMI chains, max Rhat, min ESS, fit minutes. The trace array
# holds p[1], gamma[1], omega[1], beta0[1]. The table is refreshed again after the CV
# sections, because the CV fits are only read there.
report_chain_stats <- function() {
  rows <- list()
  for (id in MODEL_IDS) {
    f <- cache_read("full", id)
    if (!is.null(f)) {
      rows[[length(rows) + 1]] <- f$fit_stats %>%
        select(model_id, Model, scheme, fit_minutes, divergences, max_treedepth_hits,
               low_bfmi_chains, max_rhat, min_n_eff)
    }
    for (scheme in c("llo2", "llo3")) {
      r <- cache_read(scheme, id)
      if (!is.null(r)) {
        rows[[length(rows) + 1]] <- tibble(
          model_id = id, Model = model_label(id), scheme = scheme, fit_minutes = r$fit_minutes,
          divergences = r$hmc$divergent, max_treedepth_hits = r$hmc$max_treedepth,
          low_bfmi_chains = r$hmc$low_bfmi_chains,
          max_rhat = max(r$rhat_df$Rhat, na.rm = TRUE), min_n_eff = min(r$rhat_df$n_eff, na.rm = TRUE))
      }
    }
  }
  if (length(rows) == 0) { message("No cached fits yet -- chain statistics table skipped."); return(invisible(NULL)) }
  chain_stats <- bind_rows(rows)
  write_csv(chain_stats, "outputs/chain_stats_all_models.csv")
  message("Saved: outputs/chain_stats_all_models.csv (", nrow(chain_stats), " model x scheme rows)")
  invisible(chain_stats)
}
chain_stats_all <- report_chain_stats()

# ---- Trace and density overlays ----
if (!is.null(hier_full)) {
  ggsave("plots/trace_plot_hierarchical.jpg", mcmc_trace(hier_full$trace_array), width = 12, height = 6, dpi = 300)
  # Density overlays: see if chains produce consistent posteriors
  ggsave("plots/density_overlay_hierarchical.jpg", mcmc_dens_overlay(hier_full$trace_array), width = 8, height = 6, dpi = 300)
}

# --------- Parameter shrinkage plots (base hierarchical model): district value vs cross-district mean ---------
if (!is.null(hier_full)) {
  # --- gamma[d] vs cross-district mean ---
  gamma_district_medians <- hier_full$param_medians$gamma
  gamma_global_median    <- mean(gamma_district_medians)

  gamma_df <- data.frame(
    District = district_names,
    gamma_district = gamma_district_medians,
    gamma_global = gamma_global_median
  )

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

  ggsave("plots/gamma_shrinkage_plot.jpg", plot = p_gamma_shrinkage, width = 10, height = 8, dpi = 300)

  # --- p[d] vs cross-district mean ---
  p_district_medians <- hier_full$param_medians$p
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

  # --- omega[d] vs cross-district mean ---
  omega_district_medians <- hier_full$param_medians$omega
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
}

# ------- End of chain and model convergence stats -------


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

# One CV section: fit (cached) every model, then tables, relative scores and plots.
# Tables cover every cached model (base, each LLM run, each pooled model) plus the
# benchmarks; the bar plots and district panels show the hierarchical model, the pooled models and
# the benchmarks so they stay readable.
run_cv_section <- function(scheme) {
  sp  <- CV_SPECS[[scheme]]
  tag <- toupper(scheme)

  res <- list()
  for (id in MODEL_IDS) {
    r <- fit_cv_model(scheme, id)
    if (is.null(r)) { message("Skipping ", model_label(id), " (", tag, "): fit not cached and not scheduled."); next }
    res[[id]] <- r
    write_csv(r$rhat_df, file.path(DIAG_DIR, paste0("rhat_and_ess_", sp$rhat_prefix, "_", model_label(id), ".csv")))
    write_cv_model_metrics(scheme, r)   # this model's own metrics file, written as soon as it is fitted
  }
  if (length(res) == 0) { message(tag, ": no cached or scheduled fits -- section skipped."); return(invisible(NULL)) }

  bayes_df <- bind_rows(lapply(res, function(r) r$scores))
  all_df   <- bind_rows(bayes_df, .benchmark_rows(scheme))
  write_csv(all_df, sprintf("outputs/%s_CV_all_models_with_CRPS.csv", tag))
  message("Saved: outputs/", tag, "_CV_all_models_with_CRPS.csv (", length(res), " Bayesian models + 2 benchmarks)")

  # --- Summary by model and held-out point (and all points pooled) ---
  summary_df <- bind_rows(
    all_df %>% mutate(Point = as.character(Point)),
    all_df %>% mutate(Point = "All")
  ) %>%
    group_by(Model, Point) %>%
    summarise(
      RMSE      = sqrt(mean(SqError, na.rm = TRUE)),  # aggregate RMSE = sqrt(mean squared error)
      MAE       = mean(AbsError, na.rm = TRUE),
      MAPE_mean = mean(MAPE, na.rm = TRUE),
      CRPS_mean = mean(CRPS, na.rm = TRUE),
      .groups   = "drop"
    ) %>%
    arrange(Point, RMSE)
  print(as.data.frame(summary_df %>% filter(Point == "All")))
  write_csv(summary_df, sprintf("outputs/%s_CV_summary_by_model.csv", tag))
  message("Saved: outputs/", tag, "_CV_summary_by_model.csv")

  # --- Relative (per district, then averaged) and population-weighted scores ---
  report_cv_relative_scores(all_df, tag, paste0(sp$title, ": relative RMSE by district"))

  # --- Plots: base, pooled models and benchmarks ---
  plot_models <- c(MODEL_LABELS, "Naive", "HistAverage")
  plot_colors <- c(MODEL_COLORS, Naive = "grey50", HistAverage = "darkgreen")
  plot_labels <- c(MODEL_DISPLAY, Naive = "Naive (P_t = P_{t-6})", HistAverage = "Hist. Average")
  plot_df  <- all_df   %>% filter(Model %in% plot_models)
  bayes_pl <- bayes_df %>% filter(Model %in% MODEL_LABELS)

  # Bar plot: absolute error by district for all plotted models (faceted by prediction point)
  p_rmse <- ggplot(plot_df, aes(x = reorder(District, -AbsError), y = AbsError, fill = Model)) +
    geom_bar(stat = "identity", position = "dodge") +
    facet_wrap(~Point, nrow = sp$plot_nrow, scales = "free_x") +
    coord_flip() +
    scale_fill_manual(values = plot_colors, labels = plot_labels) +
    labs(title = paste0(sp$title, ": Absolute Error by District — base, pooled LLM models and benchmarks"),
         x = "District", y = "Absolute Error") +
    theme_minimal() +
    theme(strip.text = element_text(size = 10), axis.text.y = element_text(size = 7))
  ggsave(sprintf("plots/%s_CV_RMSE_all_models.jpg", tag), p_rmse,
         width = sp$rmse_dims[1], height = sp$rmse_dims[2], dpi = 300)
  message("Saved: plots/", tag, "_CV_RMSE_all_models.jpg")

  # Bar plot: CRPS by district for the Bayesian models only (no benchmarks)
  p_crps <- ggplot(bayes_pl, aes(x = reorder(District, -CRPS), y = CRPS, fill = Model)) +
    geom_bar(stat = "identity", position = "dodge") +
    facet_wrap(~Point, nrow = sp$plot_nrow, scales = "free_x") +
    coord_flip() +
    scale_fill_manual(values = MODEL_COLORS, labels = MODEL_DISPLAY) +
    labs(title = paste0(sp$title, ": CRPS by District — Bayesian Models"),
         x = "District", y = "CRPS") +
    theme_minimal() +
    theme(strip.text = element_text(size = 10), axis.text.y = element_text(size = 7))
  ggsave(sprintf("plots/%s_CV_CRPS_bayesian_models.jpg", tag), p_crps,
         width = sp$rmse_dims[1], height = sp$rmse_dims[2], dpi = 300)
  message("Saved: plots/", tag, "_CV_CRPS_bayesian_models.jpg")

  # District panel: observed SAM trajectory vs. each model's prediction at the held-out
  # points, one small panel per district
  preds_for_plot <- bayes_pl %>% mutate(t = as.integer(sub("[a-z]+$", "", Point)))
  p_panel <- ggplot() +
    geom_line(data = df_model_t, aes(x = t, y = SAM), color = "black", linewidth = 0.4) +
    geom_point(data = df_model_t, aes(x = t, y = SAM), color = "black", size = 0.6) +
    geom_point(data = preds_for_plot, aes(x = t, y = Predicted, color = Model), size = 1.3, alpha = 0.85) +
    facet_wrap(~District, scales = "free_y", ncol = 9) +
    scale_color_manual(values = MODEL_COLORS, labels = MODEL_DISPLAY, name = NULL) +
    scale_x_continuous(breaks = 1:9) +
    labs(
      title    = paste0("Observed SAM trajectory vs. ", tag, "-CV predictions (",
                        paste(sp$points, collapse = "/"), " held-out points), by district"),
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
  ggsave(sprintf("plots/%s_cv_district_panel_observed_vs_predicted.jpg", tolower(tag)), p_panel,
         width = 22, height = 20, dpi = 250)
  message("Saved: plots/", tolower(tag), "_cv_district_panel_observed_vs_predicted.jpg")

  invisible(all_df)
}

# ============================================================
# 5. LLO-2 CROSS-VALIDATION -- hierarchical model and benchmarks
# Metrics per model and held-out point (and all points pooled): RMSE, MAE, MAPE, CRPS,
# plus population-weighted RMSE (wRMSE) and the RELATIVE scores -- the district's RMSE
# divided by the benchmark's RMSE (historical mean, persistence), averaged over
# districts (mean, geometric mean, median, population-weighted mean, win rate).
# Outputs: LLO2_CV_all_models_with_CRPS.csv, LLO2_CV_summary_by_model.csv,
# LLO2_CV_relative_scores.csv (+ by-district), plots, rhat files per model.
# ============================================================
llo2_all_df <- run_cv_section("llo2")

# ============================================================
# 6. LLO-3 CROSS-VALIDATION -- same models, metrics and outputs (prefix LLO3)
# ============================================================
llo3_all_df <- run_cv_section("llo3")

# Refresh the chain-statistics table now that the CV fits are available
chain_stats_all <- report_chain_stats()


# ============================================================
# 7. DAILY MODEL -- hierarchical_weakly_informative_priors_daily.stan
# Same data as the monthly model; p and gamma are daily rates, 180-day (6-month)
# arrival window matching the semi-annual observation interval.
# The fit is cached in cache/full/hierarchical_daily.rds (skipped when present).
# Output: outputs/parameter_and_burden_summary_hierarchical_daily.csv
# ============================================================
stan_data_daily <- list(
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
  p_hyperprior     = HIER_DAILY_HYPERPRIORS$p,
  # Beta(3,6000): mean ~0.049975% of U5 pop/day, central 95% interval
  # ~0.01031%-0.12032%; x30 gives ~0.3093%-3.6096%, close to but not identical
  # to the monthly Beta(3,200) 95% interval (~0.3073%-3.5306%). Matches the
  # monthly gamma_hyperprior shape numerically (mean-arrival scaling,
  # gamma_month = 30*gamma_day), NOT a distributional rescaling proof.
  gamma_hyperprior = HIER_DAILY_HYPERPRIORS$gamma,
  # Beta(3,6000): same shapes as gamma_hyperprior. This is NOT a preservation
  # of the monthly arrival-SD scaling relation (omega_month = sqrt(30)*omega_day);
  # using the same (3,6000) shapes for omega as for gamma does not, by itself,
  # keep the implied monthly-equivalent process variance unchanged.
  omega_hyperprior = HIER_DAILY_HYPERPRIORS$omega
)

fit_daily_model <- function() {
  t0_fit <- Sys.time()
  fit_daily <- stan(
    file   = "hierarchical_weakly_informative_priors_daily.stan",
    data   = stan_data_daily,
    chains = MCMC$chains, warmup = MCMC$warmup, iter = MCMC$iter, cores = MCMC$cores,
    seed   = SEEDS[["full"]]
  )
  fit_minutes <- round(as.numeric(difftime(Sys.time(), t0_fit, units = "mins")), 2)
  message("Model fit time [hierarchical_daily]: ", fit_minutes, " minutes")

  posterior_daily <- rstan::extract(fit_daily)
  print(fit_daily)
  hmc <- .report_hmc_diagnostics(fit_daily, "hierarchical_daily (full data)")

  loo_daily <- loo::loo(rstan::extract(fit_daily, pars = "log_lik")$log_lik)
  print(loo_daily)

  # --- Build burden summary ---
  burden_draws_daily    <- as.matrix(posterior_daily$burden)
  district_labels_daily <- district_names[df_model$district_id]
  burden_long_daily     <- as.data.frame(burden_draws_daily)
  burden_long_daily$iteration <- seq_len(nrow(burden_long_daily))
  burden_summary_daily <- burden_long_daily |>
    tidyr::pivot_longer(-iteration, names_to = "col", values_to = "burden") |>
    mutate(
      i         = as.integer(sub("^V", "", col)),
      District  = district_labels_daily[i],
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
  sd_pred_median_by_obs_daily <- apply(as.matrix(posterior_daily$sd_pred), 2, median)   # iterations x N -> N
  alpha_equiv_daily <- tapply(sd_pred_median_by_obs_daily, df_model$district_id, median)

  param_medians_daily <- tibble(
    District             = district_names,
    beta0                = apply(posterior_daily$beta0,                2, median),
    beta1                = apply(posterior_daily$beta1,                2, median),
    p                    = apply(posterior_daily$p,                    2, median),
    gamma                = apply(posterior_daily$gamma,                2, median),
    gamma_count          = apply(posterior_daily$gamma_count_out,      2, median),
    omega                = apply(posterior_daily$omega,                2, median),   # fraction
    omega_count          = apply(posterior_daily$omega_count_out,      2, median),   # in counts
    alpha_equiv          = as.numeric(alpha_equiv_daily),
    mean_duration        = apply(posterior_daily$mean_duration,        2, median),   # 1/p in days
    mean_duration_months = apply(posterior_daily$mean_duration_months, 2, median),   # 1/p in months
    kappa_p              = median(posterior_daily$kappa_p),
    kappa_gamma          = median(posterior_daily$kappa_gamma),
    kappa_omega          = median(posterior_daily$kappa_omega)
  )

  burden_wide_daily <- burden_summary_daily %>%
    select(District, Timepoint, burden_median) %>%
    pivot_wider(names_from = Timepoint, values_from = burden_median, names_prefix = "burden_")

  list(final_summary = param_medians_daily %>% left_join(burden_wide_daily, by = "District"),
       burden_summary = burden_summary_daily, loo = loo_daily, hmc = hmc, fit_minutes = fit_minutes)
}

daily_cache <- file.path(CACHE_DIR, "full", "hierarchical_daily.rds")
daily <- if (file.exists(daily_cache) && !FORCE_REFIT) readRDS(daily_cache) else NULL
if (is.null(daily) && RUN_DAILY && "full" %in% RUN_SCHEMES) {
  daily <- fit_daily_model()
  cache_write(daily, "full", "hierarchical_daily")
}
if (!is.null(daily)) {
  write_csv(daily$final_summary, "outputs/parameter_and_burden_summary_hierarchical_daily.csv")
  message("Saved: outputs/parameter_and_burden_summary_hierarchical_daily.csv")
} else {
  message("Daily model: not cached and not scheduled (RUN_DAILY / RUN_SCHEMES) -- skipped.")
}
# ============================================================
# End of the daily model
# ============================================================


# ============================================================
# 8. MAPS AND GRAPHS, BURDEN COMPARISON WITH K = 1.6
# Models mapped: the hierarchical model. Inputs come from the cached full-data fit.
# The shared incidence scales here are computed from this model alone, so bins can differ
# from the maps in no_pooling_LLM_informed_priors/, which use one common scale across the
# hierarchical model and the pooled LLM models.
# ============================================================
maps_ready <- !is.null(hier_full)
if (!maps_ready) message("Section 8 skipped: needs the cached full-data fit of the hierarchical model.")
if (maps_ready) {

# ---- Prepare map inputs from the cached fit ----
.prep_from_full <- function(f) {
  list(
    param_medians   = f$param_medians %>% select(District, p, gamma, gamma_count),
    burden_sum      = f$burden_summary,
    burden_expected = f$burden_expected
  )
}
map_inputs <- .prep_from_full(hier_full)

# K_expected = E[burden] / E[SAM] - 1
sam_mean_per_district <- df_filtered %>%
  group_by(District) %>%
  summarise(SAM_mean = mean(SAM, na.rm = TRUE), .groups = "drop")

.compute_k_expected <- function(burden_expected_df) {
  left_join(burden_expected_df, sam_mean_per_district, by = "District") %>%
    mutate(K_expected = burden_expected / SAM_mean - 1) %>%
    filter(is.finite(K_expected))
}
K_expected_summary <- .compute_k_expected(map_inputs$burden_expected)

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
shapefile_path <- SHAPEFILE_PATH

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
# (.make_model_maps() below) and the standalone benchmark burden map, so
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
# intervals are [a,b), so the last interval is [5, Inf) and K_expected == 5
# itself falls in the last bin -- hence ">=5" (not ">5"), same reasoning as
# the duration map fix above. Bins are 0.5 wide up to 5; K_expected estimates
# stay below 5.3 across models, so no wider range is needed.
k_bin_breaks <- c(seq(0, 5, by = 0.5), Inf)
k_bin_labels <- c(paste0(sprintf("%.1f", seq(0, 4.5, by = 0.5)), "–", sprintf("%.1f", seq(0.5, 5, by = 0.5))), ">=5")
k_map_colors <- setNames(green_palette(length(k_bin_labels)), k_bin_labels)

k_map_display_names <- c(hierarchical = "Hierarchical")

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
.make_k_expected_map(K_expected_summary, "hierarchical")



# ============================================================
# Geographic orientation overlays for the model comparison maps
# ============================================================
# Selected district labels, Admin-1 region boundaries, and regional text
# labels are all computed once here (not per-model) and reused identically
# across the hierarchical and every LLM-informed map, so label placement stays
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
# incidence-count, and incidence-rate maps for every model so placement
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
# Model comparison maps: duration, incidence, burden (hierarchical and pooled LLM models)
# ============================================================

# Helper: generate duration, incidence, and burden maps for one model
.make_model_maps <- function(param_medians, burden_sum, model_label,
                                 name_map, somalia_map,
                                 duration_colors, gamma_cap, gamma_rate_cap) {
  dir.create("plots", showWarnings = FALSE)

  # Titles show a nicely-cased model name (e.g. "Hierarchical", "ChatGPT"); model_label is
  # still used for the file names. Falls back to the label itself if a new model name shows up.
  model_display_names <- c(hierarchical = "Hierarchical")
  model_key <- model_label
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


# Shared incidence scales (counts and rate). The model's own single highest district is
# dropped before taking the maximum, so one outlier does not stretch the "regular" bins;
# any district above the cap lands in the open-ended top bin.
.cap_excl_top <- function(v) max(v[-which.max(v)], na.rm = TRUE)
gamma_common_cap      <- .cap_excl_top(map_inputs$param_medians$gamma_count)
gamma_rate_common_cap <- .cap_excl_top(map_inputs$param_medians$gamma)

.make_model_maps(map_inputs$param_medians, map_inputs$burden_sum, "hierarchical",
                 name_map, somalia_map, duration_colors, gamma_common_cap, gamma_rate_common_cap)

# ============================================================
# Burden comparison: hierarchical model vs the K = 1.6 benchmark
# ============================================================
burden_compare_long <- bind_rows(
  hier_full$burden_summary %>% mutate(Model = "hierarchical"),
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

# Wide-format: one row per (District, Timepoint), models side by side with the difference
burden_compare_wide <- burden_compare_long %>%
  select(District, Timepoint, Model, burden_median) %>%
  pivot_wider(names_from = Model, values_from = burden_median) %>%
  mutate(diff_hierarchical_vs_k16 = hierarchical - k1.6) %>%
  arrange(District, Timepoint)
write_csv(burden_compare_wide, "outputs/burden_comparison_wide.csv")
message("Saved: outputs/burden_comparison_wide.csv")

# Summary: mean burden per district per model (averaged across timepoints)
burden_compare_summary <- burden_compare_long %>%
  group_by(Model, District) %>%
  summarise(burden_mean = mean(burden_median, na.rm = TRUE), burden_total = sum(burden_median, na.rm = TRUE), .groups = "drop")
write_csv(burden_compare_summary, "outputs/burden_comparison_summary_by_district.csv")
message("Saved: outputs/burden_comparison_summary_by_district.csv")

# Global summary: grand mean and total across all districts and timepoints
burden_compare_global <- burden_compare_long %>%
  group_by(Model) %>%
  summarise(burden_grand_mean = mean(burden_median, na.rm = TRUE), burden_grand_total = sum(burden_median, na.rm = TRUE), .groups = "drop")
print(burden_compare_global)
write_csv(burden_compare_global, "outputs/burden_comparison_global_summary.csv")
message("Saved: outputs/burden_comparison_global_summary.csv")
# ============================================================
# End of burden comparison
# ============================================================
}   # end if (maps_ready)
