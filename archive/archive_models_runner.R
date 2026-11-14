# ============================================================
# Archive runner -- the earlier SAM burden models, run through the Draft 1 pipeline
#
# Same data cleaning, leave-last-2/3-out cross-validation, benchmarks (Naive, HistAverage)
# and score tables (RMSE, wRMSE, MAE, MAPE, CRPS, per-district relative scores) as in
#   ../no_pooling_LLM_informed_priors/no_pooling_LLM_informed_priors.R
#   ../hierarchical_weakly_informative_priors/hierarchical_weakly_informative_priors.R
# so an archived model can be compared with them on equal terms.
#
# The archived models are the AR(1) models with a CONSTANT observation noise alpha (the
# theoretical-variance models of Draft 1 replaced this by a state-dependent variance):
#   hierarchical_AR1_constant_noise                          partial pooling of p, gamma, alpha
#   no_pooling_AR1_constant_noise                            independent districts, fixed priors
#   full_pooling_AR1_constant_noise                          one p, gamma, alpha for all districts
#   no_pooling_LLM_informed_priors_AR1_constant_noise        district Beta priors from the tier JSONs
#   hierarchical_district_risk_factors_AR1_constant_noise    district-level risk-factor covariates
#   hierarchical_time_varying_single_covariate               p_t / gamma_t driven by ONE lagged covariate
#   hierarchical_time_varying_lag_horseshoe                  horseshoe prior on the 6 lag coefficients
#   hierarchical_time_varying_lag_dirichlet_shared_weights   Dirichlet lag weights shared by p and gamma
#   hierarchical_time_varying_lag_dirichlet_channel_weights  separate Dirichlet lag weights for p and gamma
# Stan files are in stan_models/ (current array syntax; the original files are untouched in
# stan_models_original_syntax/). Results of the earlier analyses are in risk_factor_results/.
#
# CONTENTS
#   0. Setup and settings
#   1. Data loading and cleaning (same as the Draft 1 scripts)
#   2. Risk-factor covariates: rainfall, NDVI, total alarms and their 1..6-month lags
#   3. Model specifications: Stan file, data builder and predictive function of every model
#   4. Fits, cache, cross-validation scoring
#   5. LLO-2 and LLO-3 tables
#
# RUNNING
#   Fits are cached in cache/<scheme>/<model_id>.rds (scheme = full, llo2, llo3) and skipped
#   when present. RUN_MODELS (NULL = all) and RUN_SCHEMES limit what is fitted in a session.
#   For a quick wiring test use SMOKE_TEST <- TRUE (2 short chains, separate cache folder
#   and output tables, NOT usable as results).
#   Standardisation of the lagged covariates: with all data for the "full" scheme (as in the
#   original analysis); in LLO-2 / LLO-3 with the mean and SD of the TRAINING rows only, so no
#   information from the held-out points enters the covariates.
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
library(zoo)

# ============================================================
# 0. SETUP
# ============================================================
PROJECT_DIR       <- "C:/Users/ecemy/OneDrive - INSEAD/Desktop/Waste prediction/Code/Draft 1/archive"
DATA_PATH         <- "C:/Users/ecemy/OneDrive - INSEAD/Desktop/Waste prediction/Datasets/Final_Clean_Dataset.csv"
RISK_FACTORS_PATH <- "C:/Users/ecemy/OneDrive - INSEAD/Desktop/Waste prediction/Datasets/FSNAU_riskfactors.csv"
OLD_PRIOR_DIR     <- file.path(PROJECT_DIR, "old_priors_json", "old variance model json files")
PRIOR_TIER        <- "informative"      # tier of the old LLM prior JSONs: informative | moderate | weak
setwd(PROJECT_DIR)

# ---- Batch controls (see the header) ----
RUN_MODELS  <- NULL
RUN_SCHEMES <- c("full", "llo2", "llo3")
FORCE_REFIT <- FALSE
SMOKE_TEST  <- FALSE
MCMC  <- list(chains = 4, warmup = 5000, iter = 10000, cores = 4)
SEEDS <- c(full = 123, llo2 = 456, llo3 = 789)
if (SMOKE_TEST) MCMC <- list(chains = 2, warmup = 300, iter = 600, cores = 2)
if (exists("MCMC_OVERRIDE")) MCMC <- MCMC_OVERRIDE

TAG_DIR   <- if (SMOKE_TEST) "smoke_test" else "."
OUT_DIR   <- file.path(TAG_DIR, "outputs")
PLOT_DIR  <- file.path(TAG_DIR, "plots")
CACHE_DIR <- file.path(TAG_DIR, "cache")
for (.d in c(OUT_DIR, PLOT_DIR)) dir.create(.d, showWarnings = FALSE, recursive = TRUE)
for (.s in c("full", "llo2", "llo3")) dir.create(file.path(CACHE_DIR, .s), showWarnings = FALSE, recursive = TRUE)

# Hyperpriors used in the original analyses of these models
HP <- list(
  alpha_models = list(p = c(3, 2), gamma = c(1, 1), alpha = c(log(1000), 1)),
  risk_model   = list(p = c(3, 2), gamma = c(5, 2.5), alpha = c(log(1000), 0.5)),
  horseshoe    = list(tau0 = 0.05, slab_scale = 1.0, slab_df = 4.0),
  dirichlet    = list(concentration = 1.0, alpha_scale = 0.5),
  single_x     = list(beta_p_sd = 0.2, beta_gamma_sd = 0.2)
)
# Which lagged covariate / channel the single-covariate and horseshoe models use by default
SINGLE_X <- list(covariate = "rainfall_imp", lag = 1L, channel = "gamma")   # channel: "gamma" or "p"
HS_CFG   <- list(covariate = "rainfall_imp", channel = "gamma")

MODEL_COLORS  <- NULL     # used by report_cv_relative_scores(); NULL -> default colours
MODEL_DISPLAY <- NULL

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

df_model$row_id <- seq_len(nrow(df_model))   # position of the row in the full data (used to find y[t-1])


# ============================================================
# 2. RISK-FACTOR COVARIATES
# Monthly FSNAU risk factors: rainfall, NDVI, total alarms. Missing values: rainfall / NDVI are
# filled with the district x calendar-month mean, then the district mean, then the national
# calendar-month mean; total alarms by last observation carried forward within a district, then
# the district mean. Lags 1..6 (months) are built per district on the monthly series and joined
# to the SAM rows (one SAM observation every 6 months), so a lag-k covariate at a SAM date
# only uses information from before that date.
# ============================================================
COV_BASES <- c("rainfall_imp", "ndvi_score_imp", "total_alarms_imp")
N_LAGS    <- 6L

build_risk_covariates <- function(df_model) {
  rf <- readr::read_csv(RISK_FACTORS_PATH, show_col_types = FALSE) %>%
    mutate(district = case_when(district == "Ceel Dheer" ~ "Ceel Dheere",
                                district == "Mogadishu"  ~ "Banadir",
                                TRUE ~ district),
           date  = as.Date(date),
           Year  = as.integer(format(date, "%Y")),
           Month = as.integer(format(date, "%m")))

  for (v in c("rainfall", "ndvi_score")) {
    imp <- paste0(v, "_imp")
    rf <- rf %>% group_by(district, Month) %>%
      mutate(!!imp := ifelse(is.na(.data[[v]]), mean(.data[[v]], na.rm = TRUE), .data[[v]])) %>% ungroup()
    rf <- rf %>% group_by(district) %>%
      mutate(!!imp := ifelse(is.na(.data[[imp]]), mean(.data[[v]], na.rm = TRUE), .data[[imp]])) %>% ungroup()
    rf <- rf %>% group_by(Month) %>%
      mutate(!!imp := ifelse(is.na(.data[[imp]]), mean(.data[[v]], na.rm = TRUE), .data[[imp]])) %>% ungroup()
  }
  rf <- rf %>% group_by(district) %>% arrange(date, .by_group = TRUE) %>%
    mutate(total_alarms_imp = zoo::na.locf(`Total alarms`, na.rm = FALSE),
           total_alarms_imp = ifelse(is.na(total_alarms_imp), mean(`Total alarms`, na.rm = TRUE), total_alarms_imp)) %>%
    ungroup()
  stopifnot(!anyNA(rf[COV_BASES]))

  lag_fns <- setNames(lapply(seq_len(N_LAGS), function(k) { force(k); function(.x) dplyr::lag(.x, k) }),
                      paste0("lag", seq_len(N_LAGS)))
  lagged <- rf %>% arrange(district, date) %>% group_by(district) %>%
    mutate(across(all_of(COV_BASES), lag_fns, .names = "{.col}_{fn}")) %>% ungroup() %>%
    select(district, Year, Month, all_of(COV_BASES), matches("_lag[0-9]+$"))

  out <- df_model %>%
    left_join(lagged, by = c("District" = "district", "Year", "Month")) %>%
    arrange(district_id, Year, Month)
  stopifnot(nrow(out) == nrow(df_model), identical(out$row_id, df_model$row_id))
  lag_cols <- unlist(lapply(COV_BASES, function(b) paste0(b, "_lag", seq_len(N_LAGS))))
  stopifnot(all(is.finite(as.matrix(out[lag_cols]))))
  out
}
df_model_rf <- build_risk_covariates(df_model)
message("Risk-factor covariates joined: ", sum(grepl("_lag[0-9]+$", names(df_model_rf))), " lag columns, ",
        sum(!complete.cases(df_model_rf[COV_BASES])), " SAM rows without a same-month covariate value")

# Standardisation context: mean and SD of every covariate column over a set of rows (the training
# rows in CV, all rows for the full-data fit). Held-out rows are standardised with the SAME numbers.
make_cov_ctx <- function(rows_tr) {
  cols <- c(COV_BASES, unlist(lapply(COV_BASES, function(b) paste0(b, "_lag", seq_len(N_LAGS)))))
  stats <- lapply(setNames(cols, cols), function(cn) c(mean = mean(rows_tr[[cn]]), sd = sd(rows_tr[[cn]])))
  stopifnot(all(vapply(stats, function(s) is.finite(s[["sd"]]) && s[["sd"]] > 0, logical(1))))
  stats
}
z_col <- function(rows, col, stats) (rows[[col]] - stats[[col]][["mean"]]) / stats[[col]][["sd"]]
lag_matrix <- function(rows, base, stats) {
  sapply(seq_len(N_LAGS), function(l) z_col(rows, paste0(base, "_lag", l), stats))   # N x L
}

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

  out_summary <- file.path(OUT_DIR, paste0(tag, "_CV_relative_scores.csv"))
  out_dist    <- file.path(OUT_DIR, paste0(tag, "_CV_relative_by_district.csv"))
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
      scale_color_discrete(name = NULL) +
      labs(title = plot_title,
           subtitle = "Per-district RMSE / benchmark RMSE over all held-out points; < 1 (left of dashed line) = model beats benchmark; districts ordered by U5 population",
           x = "Relative RMSE (log scale)", y = NULL) +
      theme_minimal(base_size = 8) +
      theme(legend.position = "bottom", axis.text.y = element_text(size = 5))
    ggsave(file.path(PLOT_DIR, paste0(tag, "_CV_relative_RMSE_by_district.jpg")),
           p_rel, width = 12, height = 14, dpi = 250)
    message("Saved: ", PLOT_DIR, "/", tag, "_CV_relative_RMSE_by_district.jpg")
  }
  invisible(res)
}


# ============================================================
# 3. MODEL SPECIFICATIONS
# A spec holds: the Stan file, data_fn(rows_tr, starts, cov_stats) -> Stan data, and
# pred_fn(post, rows_te, cov_stats) -> S x N_te matrix of one-step-ahead predictive draws. Each held-out
# point is predicted from the TRUE previous observation (same convention as the Draft 1 scripts):
#   y_t ~ Normal(beta0_t + beta1_t * y_{t-1}, alpha_d)
# with beta1 = (1-p)^6 and beta0 = gamma * U5pop * sum_{i=1..6} (1-p)^(6-i).
# ============================================================
pop_ref_vec <- as.numeric(pop_u5_ref)
.sum_terms <- function(p) { s <- 0; for (i in 1:6) s <- s + (1 - p)^(6 - i); s }

# y[t-1] of every held-out row (the previous row of the same district in the full data)
.y_prev <- function(rows_te) df_model$SAM[rows_te$row_id - 1]

# Predictive draws for models whose beta0, beta1 and alpha are district parameters (no covariate time variation)
.pred_district_params <- function(post, rows_te, cov_stats = NULL) {
  S <- nrow(post$beta0); ds <- rows_te$district_id
  b0 <- post$beta0[, ds, drop = FALSE]; b1 <- post$beta1[, ds, drop = FALSE]; al <- post$alpha[, ds, drop = FALSE]
  yp <- matrix(.y_prev(rows_te), S, nrow(rows_te), byrow = TRUE)
  matrix(rnorm(S * nrow(rows_te), b0 + b1 * yp, al), S, nrow(rows_te))
}
# One p, gamma, alpha for everybody (full pooling): scalars per draw
.pred_scalar_params <- function(post, rows_te, cov_stats = NULL) {
  S <- length(post$beta0); n <- nrow(rows_te)
  yp <- matrix(.y_prev(rows_te), S, n, byrow = TRUE)
  # rstan returns scalar parameters as 1-d arrays; plain vectors recycle down the columns of the S x n matrix
  matrix(rnorm(S * n, as.vector(post$beta0) + as.vector(post$beta1) * yp, as.vector(post$alpha)), S, n)
}
# Time-varying p_t, gamma_t: logit(p_t) = logit(p_d) + xb_p, logit(gamma_t) = logit(gamma_d) + xb_g
.pred_time_varying <- function(post, rows_te, xb_fn) {
  S <- nrow(post$p); n <- nrow(rows_te); ds <- rows_te$district_id
  xb <- xb_fn(post, rows_te)                                                  # list(xb_p, xb_g): S x n
  p_t <- plogis(qlogis(post$p[, ds, drop = FALSE])         + xb$xb_p)
  g_t <- plogis(qlogis(post$gamma[, ds, drop = FALSE])     + xb$xb_g)
  b1  <- (1 - p_t)^6
  b0  <- g_t * matrix(pop_ref_vec[ds], S, n, byrow = TRUE) * .sum_terms(p_t)
  yp  <- matrix(.y_prev(rows_te), S, n, byrow = TRUE)
  matrix(rnorm(S * n, b0 + b1 * yp, post$alpha[, ds, drop = FALSE]), S, n)
}
# Stan-data pieces shared by the time-varying models
.tv_base_data <- function(rows_tr, starts) {
  list(N = nrow(rows_tr), D = D, district_id = rows_tr$district_id, y = rows_tr$SAM,
       pop_u5_ref = pop_ref_vec, start_index = starts,
       p_hyperprior = HP$alpha_models$p, gamma_hyperprior = HP$alpha_models$gamma,
       alpha_hyperprior = HP$alpha_models$alpha)
}
.use_channel <- function(channel) list(p = as.integer(channel == "p"), g = as.integer(channel == "gamma"))

# ---- Old LLM prior JSON (tier files): Beta(a,b) for p and gamma, lognormal(mu, sigma) for alpha ----
read_old_llm_priors <- function(tier = PRIOR_TIER) {
  raw <- jsonlite::fromJSON(file.path(OLD_PRIOR_DIR, paste0("priors_", tier, ".json")), simplifyDataFrame = FALSE)
  look <- setNames(raw$districts, vapply(raw$districts, function(x) as.character(x$district), character(1)))
  missing <- setdiff(district_names, names(look))
  if (length(missing)) stop("Districts missing in the ", tier, " prior JSON: ", paste(missing, collapse = ", "))
  g <- function(param, field) vapply(district_names, function(dn) as.numeric(look[[dn]][[param]][[field]]), numeric(1))
  list(a_p = g("p", "a"), b_p = g("p", "b"), a_gamma = g("gamma", "a"), b_gamma = g("gamma", "b"),
       mu_alpha = g("alpha", "mu"), sigma_alpha = g("alpha", "sigma"))
}

MODEL_SPECS <- list(

  hierarchical_AR1_constant_noise = list(
    stan = "stan_models/hierarchical_AR1_constant_noise.stan",
    data_fn = function(rows_tr, starts, cov_stats) .tv_base_data(rows_tr, starts)[c("N", "D", "district_id", "y", "pop_u5_ref", "start_index", "p_hyperprior", "gamma_hyperprior", "alpha_hyperprior")],
    pred_fn = .pred_district_params),

  no_pooling_AR1_constant_noise = list(
    stan = "stan_models/no_pooling_AR1_constant_noise.stan",
    data_fn = function(rows_tr, starts, cov_stats) list(N = nrow(rows_tr), D = D, district_id = rows_tr$district_id, y = rows_tr$SAM, start_index = starts),
    pred_fn = .pred_district_params),

  full_pooling_AR1_constant_noise = list(
    stan = "stan_models/full_pooling_AR1_constant_noise.stan",
    data_fn = function(rows_tr, starts, cov_stats) list(N = nrow(rows_tr), y = rows_tr$SAM, D = D, start_index = starts),
    pred_fn = .pred_scalar_params),

  no_pooling_LLM_informed_priors_AR1_constant_noise = list(
    stan = "stan_models/no_pooling_LLM_informed_priors_AR1_constant_noise.stan",
    data_fn = function(rows_tr, starts, cov_stats) {
      pr <- read_old_llm_priors()
      c(list(N = nrow(rows_tr), D = D, district_id = rows_tr$district_id, y = rows_tr$SAM,
             pop_u5_ref = pop_ref_vec, start_index = starts), pr)
    },
    pred_fn = .pred_district_params),

  # District-level covariates: the mean of each risk factor over the district's (training) rows, standardised
  # across districts; Banadir's NDVI set to 0 (no NDVI observed there), as in the original analysis.
  hierarchical_district_risk_factors_AR1_constant_noise = list(
    stan = "stan_models/hierarchical_district_risk_factors_AR1_constant_noise.stan",
    data_fn = function(rows_tr, starts, cov_stats) {
      X <- as.matrix(rows_tr %>% group_by(district_id) %>% summarise(across(all_of(COV_BASES), ~ mean(.x, na.rm = TRUE))) %>% select(-district_id))
      X <- scale(X)
      ban <- which(district_names == "Banadir")
      if (length(ban)) X[ban, "ndvi_score_imp"] <- 0
      stopifnot(all(is.finite(X)))
      list(N = nrow(rows_tr), D = D, district_id = rows_tr$district_id, y = rows_tr$SAM,
           p_hyperprior = HP$risk_model$p, gamma_hyperprior = HP$risk_model$gamma, alpha_hyperprior = HP$risk_model$alpha,
           start_index = starts, X_p = X, X_gamma = X, K_p = ncol(X), K_gamma = ncol(X))
    },
    pred_fn = .pred_district_params),

  hierarchical_time_varying_single_covariate = list(
    stan = "stan_models/hierarchical_time_varying_single_covariate.stan",
    data_fn = function(rows_tr, starts, cov_stats) {
      ch <- .use_channel(SINGLE_X$channel)
      c(.tv_base_data(rows_tr, starts),
        list(x_z = as.vector(z_col(rows_tr, paste0(SINGLE_X$covariate, "_lag", SINGLE_X$lag), cov_stats)),
             use_x_on_gamma = ch$g, use_x_on_p = ch$p,
             beta_gamma_sd = HP$single_x$beta_gamma_sd, beta_p_sd = HP$single_x$beta_p_sd))
    },
    pred_fn = function(post, rows_te, cov_stats) {
      ch <- .use_channel(SINGLE_X$channel)
      x  <- z_col(rows_te, paste0(SINGLE_X$covariate, "_lag", SINGLE_X$lag), cov_stats)
      .pred_time_varying(post, rows_te, function(post, rows) list(
        xb_p = ch$p * outer(as.vector(post$beta_p_x), x),
        xb_g = ch$g * outer(as.vector(post$beta_gamma_x), x)))
    }),

  hierarchical_time_varying_lag_horseshoe = list(
    stan = "stan_models/hierarchical_time_varying_lag_horseshoe.stan",
    data_fn = function(rows_tr, starts, cov_stats) {
      ch <- .use_channel(HS_CFG$channel)
      c(.tv_base_data(rows_tr, starts),
        list(L = N_LAGS, X_lag = lag_matrix(rows_tr, HS_CFG$covariate, cov_stats), use_x_on_gamma = ch$g, use_x_on_p = ch$p,
             hs_tau0 = HP$horseshoe$tau0, hs_slab_scale = HP$horseshoe$slab_scale, hs_slab_df = HP$horseshoe$slab_df))
    },
    pred_fn = function(post, rows_te, cov_stats) {
      ch <- .use_channel(HS_CFG$channel)
      X  <- lag_matrix(rows_te, HS_CFG$covariate, cov_stats)                      # n x L
      .pred_time_varying(post, rows_te, function(post, rows) list(
        xb_p = ch$p * (post$beta_p_lag %*% t(X)), xb_g = ch$g * (post$beta_gamma_lag %*% t(X))))
    }),

  hierarchical_time_varying_lag_dirichlet_shared_weights = list(
    stan = "stan_models/hierarchical_time_varying_lag_dirichlet_shared_weights.stan",
    data_fn = function(rows_tr, starts, cov_stats) {
      c(.tv_base_data(rows_tr, starts),
        list(L = N_LAGS, X_lag_rain = lag_matrix(rows_tr, "rainfall_imp", cov_stats), X_lag_ndvi = lag_matrix(rows_tr, "ndvi_score_imp", cov_stats),
             X_lag_alarms = lag_matrix(rows_tr, "total_alarms_imp", cov_stats), use_x_on_p = 1L, use_x_on_gamma = 1L,
             dir_concentration = HP$dirichlet$concentration, alpha_scale = HP$dirichlet$alpha_scale))
    },
    pred_fn = function(post, rows_te, cov_stats) {
      Xr <- lag_matrix(rows_te, "rainfall_imp", cov_stats); Xn <- lag_matrix(rows_te, "ndvi_score_imp", cov_stats); Xa <- lag_matrix(rows_te, "total_alarms_imp", cov_stats)
      er <- post$w_rain %*% t(Xr); en <- post$w_ndvi %*% t(Xn); ea <- post$w_alarms %*% t(Xa)            # S x n weighted lag composites
      .pred_time_varying(post, rows_te, function(post, rows) list(
        xb_p = as.vector(post$alpha_p_rain) * er + as.vector(post$alpha_p_ndvi) * en + as.vector(post$alpha_p_alarms) * ea,
        xb_g = as.vector(post$alpha_g_rain) * er + as.vector(post$alpha_g_ndvi) * en + as.vector(post$alpha_g_alarms) * ea))
    }),

  hierarchical_time_varying_lag_dirichlet_channel_weights = list(
    stan = "stan_models/hierarchical_time_varying_lag_dirichlet_channel_weights.stan",
    data_fn = function(rows_tr, starts, cov_stats) {
      c(.tv_base_data(rows_tr, starts),
        list(L = N_LAGS, X_lag_rain = lag_matrix(rows_tr, "rainfall_imp", cov_stats), X_lag_ndvi = lag_matrix(rows_tr, "ndvi_score_imp", cov_stats),
             X_lag_alarms = lag_matrix(rows_tr, "total_alarms_imp", cov_stats), use_x_on_p = 1L, use_x_on_gamma = 1L,
             dir_concentration = HP$dirichlet$concentration, alpha_scale = HP$dirichlet$alpha_scale))
    },
    pred_fn = function(post, rows_te, cov_stats) {
      Xr <- lag_matrix(rows_te, "rainfall_imp", cov_stats); Xn <- lag_matrix(rows_te, "ndvi_score_imp", cov_stats); Xa <- lag_matrix(rows_te, "total_alarms_imp", cov_stats)
      .pred_time_varying(post, rows_te, function(post, rows) list(
        xb_p = as.vector(post$alpha_p_rain) * (post$w_rain_p %*% t(Xr)) + as.vector(post$alpha_p_ndvi) * (post$w_ndvi_p %*% t(Xn)) + as.vector(post$alpha_p_alarms) * (post$w_alarms_p %*% t(Xa)),
        xb_g = as.vector(post$alpha_g_rain) * (post$w_rain_g %*% t(Xr)) + as.vector(post$alpha_g_ndvi) * (post$w_ndvi_g %*% t(Xn)) + as.vector(post$alpha_g_alarms) * (post$w_alarms_g %*% t(Xa))))
    })
)
MODEL_IDS <- names(MODEL_SPECS)



# ============================================================
# 4. FITS, CACHE AND CROSS-VALIDATION SCORING
# ============================================================
cache_path  <- function(scheme, model_id) file.path(CACHE_DIR, scheme, paste0(model_id, ".rds"))
cache_read  <- function(scheme, model_id) if (file.exists(cache_path(scheme, model_id))) readRDS(cache_path(scheme, model_id)) else NULL
cache_write <- function(obj, scheme, model_id) {
  p <- cache_path(scheme, model_id); tmp <- paste0(p, ".tmp"); saveRDS(obj, tmp); file.rename(tmp, p); invisible(p)
}
models_to_fit <- function() if (is.null(RUN_MODELS)) MODEL_IDS else intersect(MODEL_IDS, RUN_MODELS)
model_label   <- function(model_id) model_id

.stan_cache <- new.env()
get_stan_model <- function(model_id) {
  # Windows Application Control sometimes blocks a freshly compiled DLL at random; a retry compiles a new one
  for (attempt in 1:4) {
    if (!is.null(.stan_cache[[model_id]])) break
    .stan_cache[[model_id]] <- tryCatch(stan_model(file = MODEL_SPECS[[model_id]]$stan), error = function(e) {
      message("Stan compile/load failed (attempt ", attempt, "): ", substr(conditionMessage(e), 1, 150)); NULL })
  }
  if (is.null(.stan_cache[[model_id]])) stop("Could not compile ", model_id, " after 4 attempts")
  .stan_cache[[model_id]]
}
# Sampler settings, HMC diagnostics, draws and Rhat/ESS (the only place that calls Stan)
.run_stan <- function(model_id, data, seed, label) {
  t0  <- Sys.time()
  fit <- sampling(get_stan_model(model_id), data = data, chains = MCMC$chains, warmup = MCMC$warmup,
                  iter = MCMC$iter, cores = MCMC$cores, seed = seed, refresh = 0)
  mins <- round(as.numeric(difftime(Sys.time(), t0, units = "mins")), 2)
  message("Model fit time [", label, "]: ", mins, " minutes")
  hmc <- .report_hmc_diagnostics(fit, label)
  ps  <- summary(fit)$summary
  list(post = rstan::extract(fit), rhat_df = data.frame(Parameter = rownames(ps), Rhat = ps[, "Rhat"], n_eff = ps[, "n_eff"]),
       hmc = hmc, fit_minutes = mins)
}

# The scoring helpers are the ones of the Draft 1 scripts



# ---- Cross-validation splits (same as the Draft 1 scripts) ----
CV_SPECS <- list(
  llo2 = list(n_train = 7, points = c("8th", "9th"),        title = "Leave-Last-2-Out CV"),
  llo3 = list(n_train = 6, points = c("7th", "8th", "9th"), title = "Leave-Last-3-Out CV (robustness check)")
)
for (.nm in names(CV_SPECS)) {
  .sp <- CV_SPECS[[.nm]]
  .sp$train  <- df_model_rf %>% group_by(district_id) %>% slice_head(n = .sp$n_train) %>% ungroup()
  .sp$test   <- df_model_rf %>% group_by(district_id) %>% slice_tail(n = length(.sp$points)) %>% ungroup()
  .sp$starts <- which(!duplicated(.sp$train$district_id))
  CV_SPECS[[.nm]] <- .sp
}

# Full-data fit: posterior medians of the district parameters, plus fit health and (exploratory) LOO
fit_full_model <- function(model_id) {
  if (!FORCE_REFIT) { cached <- cache_read("full", model_id); if (!is.null(cached)) return(cached) }
  if (!("full" %in% RUN_SCHEMES) || !(model_id %in% models_to_fit())) return(NULL)
  spec <- MODEL_SPECS[[model_id]]
  message("\n==== Fitting full-data model: ", model_id, " ====")
  starts <- which(!duplicated(df_model_rf$district_id))
  st <- .run_stan(model_id, spec$data_fn(df_model_rf, starts, make_cov_ctx(df_model_rf)), SEEDS[["full"]], paste0(model_id, " (full data)"))
  med <- function(x) if (is.matrix(x) && ncol(x) == D) apply(x, 2, median) else NULL
  summ <- tibble(District = district_names)
  for (nm in c("p", "gamma", "alpha", "beta0", "beta1")) { m <- med(st$post[[nm]]); if (!is.null(m)) summ[[nm]] <- m }
  loo_obj <- tryCatch(loo::loo(st$post$log_lik), error = function(e) NULL)
  out <- list(model_id = model_id, scheme = "full", district_medians = summ, rhat_df = st$rhat_df, hmc = st$hmc,
              fit_minutes = st$fit_minutes, elpd_loo = if (is.null(loo_obj)) NA_real_ else loo_obj$estimates["elpd_loo", "Estimate"])
  cache_write(out, "full", model_id)
  out
}

# Fits (if allowed and not cached) and returns the cached CV result, else NULL
fit_cv_model <- function(scheme, model_id) {
  if (!FORCE_REFIT) { cached <- cache_read(scheme, model_id); if (!is.null(cached)) return(cached) }
  if (!(scheme %in% RUN_SCHEMES) || !(model_id %in% models_to_fit())) return(NULL)
  sp <- CV_SPECS[[scheme]]; spec <- MODEL_SPECS[[model_id]]
  message("\n==== Fitting ", toupper(scheme), "-CV model: ", model_id, " ====")
  stats <- make_cov_ctx(sp$train)
  st <- .run_stan(model_id, spec$data_fn(sp$train, sp$starts, stats), SEEDS[[scheme]], paste0(model_id, " (", toupper(scheme), "-CV train)"))
  set.seed(SEEDS[[scheme]] + sum(utf8ToInt(model_id)))
  preds  <- spec$pred_fn(st$post, sp$test, stats)
  stopifnot(nrow(preds) == nrow(st$post[[1]]) || is.matrix(preds), ncol(preds) == nrow(sp$test), all(is.finite(preds)))
  scores <- .score_cv_preds(model_label(model_id), preds, sp$test, sp$points)
  out <- list(model_id = model_id, scheme = scheme, scores = scores, rhat_df = st$rhat_df, hmc = st$hmc, fit_minutes = st$fit_minutes)
  cache_write(out, scheme, model_id)
  out
}

# One CV section: fit (cached) every model, then the score tables and the relative-score outputs
run_cv_section <- function(scheme) {
  sp  <- CV_SPECS[[scheme]]; tag <- toupper(scheme)
  res <- list()
  for (id in MODEL_IDS) {
    r <- fit_cv_model(scheme, id)
    if (is.null(r)) { message("Skipping ", id, " (", tag, "): fit not cached and not scheduled."); next }
    res[[id]] <- r
  }
  if (length(res) == 0) { message(tag, ": no cached or scheduled fits -- section skipped."); return(invisible(NULL)) }
  all_df <- bind_rows(bind_rows(lapply(res, function(r) r$scores)), .benchmark_rows(scheme))
  write_csv(all_df, file.path(OUT_DIR, sprintf("%s_CV_all_models_with_CRPS.csv", tag)))
  summary_df <- bind_rows(all_df %>% mutate(Point = as.character(Point)), all_df %>% mutate(Point = "All")) %>%
    group_by(Model, Point) %>%
    summarise(RMSE = sqrt(mean(SqError, na.rm = TRUE)), MAE = mean(AbsError, na.rm = TRUE),
              MAPE_mean = mean(MAPE, na.rm = TRUE), CRPS_mean = mean(CRPS, na.rm = TRUE), .groups = "drop") %>%
    arrange(Point, RMSE)
  print(as.data.frame(summary_df %>% filter(Point == "All")))
  write_csv(summary_df, file.path(OUT_DIR, sprintf("%s_CV_summary_by_model.csv", tag)))
  report_cv_relative_scores(all_df, tag, paste0(sp$title, ": relative RMSE by district"))
  invisible(all_df)
}

# ============================================================
# 5. LLO-2 AND LLO-3 CROSS-VALIDATION (and the full-data fits)
# ============================================================
for (id in MODEL_IDS) {
  f <- fit_full_model(id)
  if (!is.null(f)) write_csv(f$district_medians, file.path(OUT_DIR, paste0("parameter_summary_", id, ".csv")))
}
llo2_all_df <- run_cv_section("llo2")
llo3_all_df <- run_cv_section("llo3")
