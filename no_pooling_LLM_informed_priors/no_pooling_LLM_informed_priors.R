# ============================================================
# No-pooling SAM burden model with LLM-informed priors -- Somalia, 74 districts
# (Draft 1)
#
# Every district has its OWN Beta prior for p, gamma and omega (no pooling across
# districts), elicited from an LLM and read from JSON files in priors_llm/. Three LLMs
# (claude, gemini, chatgpt) were each prompted 10 times. Models fitted:
#   <llm>_run<NN>      one model per prior JSON (30 models)
#   <llm>_pooled       prior = equal-weight mixture of that LLM's 10 runs, moment-matched
#                      to one Beta per district and parameter (3 models)
#   all_llms_pooled    prior = the three <llm>_pooled priors pooled again (equal weights, moment-
#                      matched), i.e. one prior that combines all three LLMs (1 model)
# The hierarchical model with weakly informative priors lives in the sibling folder
# hierarchical_weakly_informative_priors/. This script READS its cached fits (never
# refits them) so that the comparison tables and maps in this folder include it.
#
# CONTENTS
#   0. Setup: libraries, paths, settings, helper functions
#   1. Data loading and cleaning
#   2. Outlier analysis
#   3. LLM-informed models (full data): one fit per prior JSON plus pooled
#        - posterior tables and output files for every model
#        - prior sensitivity across runs, prior-posterior overlays within each LLM
#   4. Chain statistics for every model, trace plots of the pooled models
#   5. LLO-2 cross-validation: LLM models, hierarchical model, Naive / HistAverage benchmarks
#        Metrics: RMSE, wRMSE, MAE, MAD, MAPE, wMAPE, bias, CRPS, per-district relative scores
#        (arithP / geoP for RMSE and for MAD vs Naive and HistAverage), outlier diagnostics,
#        in-sample (training-window) vs held-out comparison of the pooled LLMs, per-point ranks and
#        average rank across metrics, and the per-district plot of every metric.
#        Rab Dhuure (imputed SAM) is left out of every metric (EXCLUDE_DISTRICTS).
#   6. LLO-3 cross-validation: same
#   7. Daily model (placeholder: the LLM-informed daily model is not built yet)
#   8. Maps and graphs (hierarchical and pooled LLM models) and burden comparison
#      with the K = 1.6 benchmark
#
# FILES IN THIS FOLDER
#   no_pooling_LLM_informed_priors_monthly.stan   the Stan model (district priors as data)
#   priors_llm/priors_llm_<llm>_<run>.json        the elicited priors
#   cache/<scheme>/<model_id>.rds                 cached fits (scheme = full, llo2, llo3)
#   outputs/, plots/                              tables and figures written by this script
#   presentation_figures/                         scripts for the slide figures
#
# RUNNING IN BATCHES
#   Every Stan fit is saved to cache/ and skipped when that file exists. Anything already
#   cached is always used; the two settings below only limit what gets FITTED:
#     RUN_MODELS  = NULL (all models) or e.g. c("claude_pooled", "gemini_run03")
#     RUN_SCHEMES = any of "full", "llo2", "llo3"
#   Tables and plots are built from whatever is in the cache, so a partial batch gives
#   partial (not wrong) outputs; rerun after the next batch to complete them.
#   Delete a cache file (or set FORCE_REFIT <- TRUE) to refit a model.
#
# Model ids and labels in every table: "hierarchical" (read from the sibling folder),
# "<llm>_run<NN>", "<llm>_pooled". Benchmarks: "Naive" (previous
# observed value) and "HistAverage" (mean of all earlier observed values).
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
sf::sf_use_s2(FALSE)   # planar GEOS instead of s2: Windows Application Control blocks s2.dll
library(viridis)
library(scales)
library(stringr)
library(jsonlite)
library(scoringRules)
library(ggrepel)

# ============================================================
# 0. SETUP
# ============================================================
PROJECT_DIR    <- "C:/Users/ecemy/OneDrive - INSEAD/Desktop/Waste prediction/Code/Draft 1/no_pooling_LLM_informed_priors"
HIER_DIR       <- file.path(dirname(PROJECT_DIR), "hierarchical_weakly_informative_priors")
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

# The hierarchical model is fitted in HIER_DIR; its cached results are read from there.
HIER_ID        <- "hierarchical"
HIER_AVAILABLE <- file.exists(file.path(HIER_DIR, "cache", "full", paste0(HIER_ID, ".rds")))
if (!HIER_AVAILABLE) message("Hierarchical model results not found in ", HIER_DIR,
                             " -- comparisons with it are skipped.")

# ---- MCMC settings and seeds per scheme ----
MCMC  <- list(chains = 4, warmup = 5000, iter = 10000, cores = 4)
SEEDS <- c(full = 123, llo2 = 456, llo3 = 789)
if (exists("MCMC_OVERRIDE")) MCMC <- MCMC_OVERRIDE   # hook for quick code tests only

# ---- Batch controls (see the header) ----
RUN_MODELS  <- NULL
RUN_SCHEMES <- c("full", "llo2", "llo3")
FORCE_REFIT <- FALSE

# Cached full-data draws for p, gamma, omega and the district-mean burden are
# thinned (every THIN_EVERY-th draw) before saving: enough for density plots and
# run-to-run comparisons, small enough for ~35 models. All summary tables, Rhat
# and ESS are computed from the full, unthinned draws before thinning.
THIN_EVERY <- 5

SENS_OUT_DIR  <- "outputs/llm_prior_sensitivity"
SENS_PLOT_DIR <- "plots/llm_prior_sensitivity"
dir.create(SENS_OUT_DIR,  showWarnings = FALSE, recursive = TRUE)
dir.create(SENS_PLOT_DIR, showWarnings = FALSE, recursive = TRUE)

# Beta shapes of the fallback prior, used only for a district that is entirely absent
# from an LLM's JSON (same shapes as the hierarchical model's global hyperpriors)
FALLBACK_PRIOR_BETA <- list(
  p     = c(2, 5),
  gamma = c(3, 200),
  omega = c(3, 200)
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

# ---- LLM prior inputs: constants, JSON reader, pooling ----
# (LLM_NAMES / file naming: priors_llm/priors_llm_<llm>_<run>.json)
PRIOR_JSON_DIR <- file.path(getwd(), "priors_llm")
LLM_NAMES      <- c("claude", "gemini", "chatgpt")

# ---- In-sample (training-window) evaluation, for choosing between LLMs without the test points ----
# Every CV fit (LLO-2: first 7 time points, LLO-3: first 6) already produces posterior predictive draws
# `y_rep` for its own training points. For the models listed here the CV fit also scores those training
# points (one step ahead from the observed previous value, time points 2..n_train; time point 1 is
# excluded because it depends on the free latent P0). A CV fit that is already cached WITHOUT this
# in-sample record is refitted once (same seed, so the CV scores do not change) and the record is added to
# its cache file. Set TRAIN_EVAL <- FALSE to skip all of this; add model ids to TRAIN_EVAL_MODELS to include
# them (each model is one refit per scheme).
TRAIN_EVAL        <- TRUE
TRAIN_EVAL_MODELS <- c(paste0(LLM_NAMES, "_pooled"), "all_llms_pooled")


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

# ---- Models shown in the main plots/maps: the hierarchical model and the pooled model of each LLM ----
# (tables always cover every model: hierarchical, every LLM run, every pooled model)
# ALL_LLM_ID = the pooled model of the three LLMs' pooled priors (moment-matched again, equal weights).
ALL_LLM_ID    <- "all_llms_pooled"
MODEL_LABELS  <- c("hierarchical", paste0(LLM_NAMES, "_pooled"), ALL_LLM_ID)
MODEL_COLORS  <- setNames(c("grey40", "#2a78d6", "#eb6834", "#1baf7a", "#8b5cf6"), MODEL_LABELS)
MODEL_DISPLAY <- setNames(
  c("Hierarchical", paste0(c("Claude", "Gemini", "ChatGPT"), " (pooled)"), "All 3 LLMs (pooled)"),
  MODEL_LABELS
)

# ============================================================
# Model registry and fit cache
# Model ids: "<llm>_run<NN>" (one per prior JSON), "<llm>_pooled".
# The label of a model in outputs is its id. The hierarchical model is not in the
# registry (it is fitted in the sibling folder) but is added to every comparison.
# ============================================================
model_label <- function(model_id) model_id

build_model_registry <- function() {
  rows <- list()
  for (llm in LLM_NAMES) {
    runs <- discover_prior_runs(llm)
    if (length(runs) == 0) {
      stop("No prior JSON files found for '", llm, "' in ", PRIOR_JSON_DIR,
           " (expected priors_llm_", llm, "_<run>.json)")
    }
    rows[[length(rows) + 1]] <- tibble(model_id = sprintf("%s_run%02d", llm, runs),
                                       type = "llm_run", llm = llm, run = runs)
    rows[[length(rows) + 1]] <- tibble(model_id = paste0(llm, "_pooled"),
                                       type = "llm_pooled", llm = llm, run = NA_integer_)
  }
  # one more model: the three LLMs' pooled priors pooled again (see load_llm_priors)
  rows[[length(rows) + 1]] <- tibble(model_id = ALL_LLM_ID, type = "all_llms_pooled", llm = NA_character_, run = NA_integer_)
  bind_rows(rows)
}


MODEL_REGISTRY <- build_model_registry()
# Every model that appears in the tables: the hierarchical model (when its results exist) first
ALL_MODEL_IDS  <- c(if (HIER_AVAILABLE) HIER_ID, MODEL_REGISTRY$model_id)

# Model ids this session may FIT (cached models are always usable regardless).
models_to_fit <- function() {
  ids <- MODEL_REGISTRY$model_id
  if (!is.null(RUN_MODELS)) ids <- intersect(ids, RUN_MODELS)
  ids
}

# The hierarchical model is read from HIER_DIR's cache; every other model lives in this folder's cache.
cache_path  <- function(scheme, model_id) {
  if (model_id == HIER_ID) file.path(HIER_DIR, "cache", scheme, paste0(model_id, ".rds"))
  else                     file.path(CACHE_DIR, scheme, paste0(model_id, ".rds"))
}
cache_has   <- function(scheme, model_id) file.exists(cache_path(scheme, model_id))
cache_read  <- function(scheme, model_id) {
  if (cache_has(scheme, model_id)) readRDS(cache_path(scheme, model_id)) else NULL
}
cache_write <- function(obj, scheme, model_id) {
  stopifnot(model_id != HIER_ID)   # never write into the hierarchical folder
  p <- cache_path(scheme, model_id)
  tmp <- paste0(p, ".tmp")
  saveRDS(obj, tmp)
  file.rename(tmp, p)   # a crash mid-write never leaves a corrupt cache file
  invisible(p)
}

# ---- Priors for every informed model: one per JSON + one pooled per LLM ----
.write_prior_summary <- function(prior, source, path) {
  summ <- bind_rows(lapply(district_names, function(dn) bind_rows(lapply(c("p", "gamma", "omega"), function(param) {
    a <- prior[[dn]][[param]]$a
    b <- prior[[dn]][[param]]$b
    tibble(district = dn, parameter = param, source = source, a = a, b = b,
           concentration = a + b, prior_mean = a / (a + b),
           prior_sd = sqrt(a * b / ((a + b)^2 * (a + b + 1))),
           prior_q05 = qbeta(0.05, a, b), prior_q50 = qbeta(0.50, a, b), prior_q95 = qbeta(0.95, a, b))
  }))))
  write_csv(summ, path)
  message("Saved: ", path)
}

load_llm_priors <- function() {
  priors <- list()
  for (llm in LLM_NAMES) {
    runs <- discover_prior_runs(llm)
    run_priors <- list()
    for (run in runs) {
      id <- sprintf("%s_run%02d", llm, run)
      priors[[id]] <- read_priors_from_json(
        prior_json_path(llm, run), default_prior_ip,
        summary_csv = file.path(SENS_OUT_DIR, sprintf("prior_summary_%s_run%d.csv", llm, run))
      )
      run_priors[[length(run_priors) + 1]] <- priors[[id]]
    }
    priors[[paste0(llm, "_pooled")]] <- pool_priors_moment_match(run_priors)
    .write_prior_summary(priors[[paste0(llm, "_pooled")]], "pooled_runs",
                         file.path(SENS_OUT_DIR, sprintf("prior_summary_%s_pooled.csv", llm)))
  }
  # Pool of the pooled LLM priors: the three LLM-pooled Betas, equal weights, moment-matched again
  # (same law-of-total-variance rule as above), so the all-LLM prior is wider where the LLMs disagree.
  priors[[ALL_LLM_ID]] <- pool_priors_moment_match(lapply(paste0(LLM_NAMES, "_pooled"), function(id) priors[[id]]))
  .write_prior_summary(priors[[ALL_LLM_ID]], "pooled_llms",
                       file.path(SENS_OUT_DIR, sprintf("prior_summary_%s.csv", ALL_LLM_ID)))
  priors
}

# ---- Stan model: compiled on first use only, so a session that just reads the
# cache never needs the compiler (rstan caches the compiled .rds next to the .stan) ----
.stan_model_cache <- new.env()
get_stan_model <- function(model_id) {
  if (is.null(.stan_model_cache[["informed"]])) {
    .stan_model_cache[["informed"]] <- stan_model(file = "no_pooling_LLM_informed_priors_monthly.stan")
  }
  .stan_model_cache[["informed"]]
}

.stan_data <- function(model_id, df_tr, starts) {
  d <- list(N = nrow(df_tr), D = D, district_id = df_tr$district_id, y = df_tr$SAM,
            pop_u5_ref = as.vector(pop_u5_ref), start_index = starts)
  if (!exists("LLM_PRIORS")) LLM_PRIORS <<- load_llm_priors()
  fp <- LLM_PRIORS[[model_id]]
  c(d, list(a_p     = extract_prior_vec(fp, "p",     "a"),
            b_p     = extract_prior_vec(fp, "p",     "b"),
            a_gamma = extract_prior_vec(fp, "gamma", "a"),
            b_gamma = extract_prior_vec(fp, "gamma", "b"),
            a_omega = extract_prior_vec(fp, "omega", "a"),
            b_omega = extract_prior_vec(fp, "omega", "b")))
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
# Districts left out of every CV / in-sample METRIC (the models are still fitted on them). Rab Dhuure has
# imputed SAM (the same value at every missing time point), which makes the Naive benchmark look better
# than it should. Set to character(0) to score all districts.
EXCLUDE_DISTRICTS <- c("Rab Dhuure")
.drop_excluded <- function(x) if (length(EXCLUDE_DISTRICTS) && !is.null(x)) x[!x$District %in% EXCLUDE_DISTRICTS, , drop = FALSE] else x

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
  if (model_id == HIER_ID) return(cache_read(scheme, model_id))   # read-only: fitted in HIER_DIR
  cached <- NULL
  if (!FORCE_REFIT) {
    cached <- cache_read(scheme, model_id)
    if (!is.null(cached)) {
      need_train <- isTRUE(TRAIN_EVAL) && model_id %in% TRAIN_EVAL_MODELS && is.null(cached$train_scores) &&
                    scheme %in% RUN_SCHEMES && model_id %in% models_to_fit()
      if (!need_train) return(cached)
      message("\n==== ", toupper(scheme), "-CV model ", model_label(model_id),
              " is cached without its in-sample record: refitting once to add it (CV scores are kept) ====")
    }
  }
  if (is.null(cached) && (!(scheme %in% RUN_SCHEMES) || !(model_id %in% models_to_fit()))) return(NULL)
  sp <- CV_SPECS[[scheme]]
  message("\n==== Fitting ", toupper(scheme), "-CV model: ", model_label(model_id), " ====")
  st <- .run_stan(model_id, .stan_data(model_id, sp$train, sp$starts), SEEDS[[scheme]],
                  paste0(model_label(model_id), " (", toupper(scheme), "-CV train)"))
  # reproducible predictive draws: the seed depends on the scheme and the model id
  set.seed(SEEDS[[scheme]] + sum(utf8ToInt(model_id)))
  preds  <- sp$preds_fn(st$post, sp$train, sp$test, df_model, pop_u5_ref)
  scores <- .score_cv_preds(model_label(model_id), preds, sp$test, sp$points)
  train_scores <- if (isTRUE(TRAIN_EVAL) && model_id %in% TRAIN_EVAL_MODELS)
    .score_train_fit(model_label(model_id), st$post, sp$train) else NULL
  if (!is.null(cached)) {
    # refit only to record the in-sample fit: keep the published CV objects, report any difference
    d <- max(abs(scores$Predicted - cached$scores$Predicted), na.rm = TRUE)
    message("Refit check [", model_label(model_id), ", ", toupper(scheme), "]: max |new - cached| held-out prediction = ",
            signif(d, 3), " SAM counts")
    out <- c(cached[setdiff(names(cached), c("train_scores", "train_refit_max_diff"))],
             list(train_scores = train_scores, train_refit_max_diff = d))
  } else {
    out <- list(model_id = model_id, scheme = scheme, scores = scores, rhat_df = st$rhat_df,
                hmc = st$hmc, fit_minutes = st$fit_minutes, train_scores = train_scores)
  }
  cache_write(out, scheme, model_id)
  out
}

# In-sample scores of ONE CV fit: the posterior predictive draws `y_rep` of its own training points
# (one step ahead from the observed previous value; time points 2..n_train per district, the first
# point depends on the free latent P0 and is dropped). Same columns as .score_cv_preds.
.score_train_fit <- function(model_label_, post, df_tr) {
  tr <- df_tr %>% mutate(row = row_number()) %>% group_by(district_id) %>% mutate(t = row_number()) %>% ungroup()
  keep <- which(tr$t >= 2)
  yrep <- post$y_rep
  stopifnot(ncol(yrep) == nrow(tr))
  tibble(Model     = model_label_,
         District  = tr$District[keep],
         Point     = paste0("t", tr$t[keep]),
         Predicted = apply(yrep[, keep, drop = FALSE], 2, median),
         Observed  = tr$SAM[keep],
         CRPS      = vapply(keep, function(n) scoringRules::crps_sample(y = tr$SAM[n], dat = yrep[, n]), numeric(1))) %>%
    mutate(AbsError = abs(Predicted - Observed),
           SqError  = (Predicted - Observed)^2,
           MAPE     = ifelse(Observed == 0, NA_real_, 100 * abs(Predicted - Observed) / Observed))
}

# Benchmarks on the same training points: Naive = previous observed value, HistAverage = mean of all earlier ones.
.train_benchmark_rows <- function(scheme, which = c("Naive", "HistAverage")) {
  n0 <- CV_SPECS[[scheme]]$n_train
  bind_rows(lapply(seq_len(D), function(d) {
    y  <- df_model$SAM[df_model$district_id == d][seq_len(n0)]
    ts <- 2:n0
    bind_rows(lapply(which, function(w) {
      pred <- if (w == "Naive") y[ts - 1] else vapply(ts, function(k) mean(y[seq_len(k - 1)]), numeric(1))
      tibble(Model = w, District = district_names[d], Point = paste0("t", ts), Predicted = pred, Observed = y[ts],
             AbsError = abs(pred - y[ts]), SqError = (pred - y[ts])^2,
             MAPE = ifelse(y[ts] == 0, NA_real_, 100 * abs(pred - y[ts]) / y[ts]), CRPS = NA_real_)
    }))
  })) %>% .drop_excluded()
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
    mutate(Pop = unname(pop_lookup[District]),
           # LOG-SCALE error, log(1 + x) so that zeros are allowed: log(predicted) - log(observed) = log(pred/obs).
           # Over- and under-prediction by the same factor get the same size (+/- the same number), unlike MAPE.
           LogErr = log1p(pmax(Predicted, 0)) - log1p(pmax(Observed, 0)))
  stopifnot(!anyNA(long$Pop))

  dist_rmse <- long %>%
    group_by(Model, Point, District, Pop) %>%
    summarise(RMSE_d = sqrt(mean(SqError, na.rm = TRUE)),
              MAD_d  = median(AbsError, na.rm = TRUE),        # district's median absolute error over its held-out points
              RMSLE_d = sqrt(mean(LogErr^2, na.rm = TRUE)),   # district's log-scale RMSE over its held-out points
              .groups = "drop")

  base <- long %>%
    group_by(Model, Point) %>%
    summarise(
      n_districts      = n_distinct(District),
      RMSE             = sqrt(mean(SqError, na.rm = TRUE)),
      wRMSE_pop        = sqrt(weighted.mean(SqError, Pop, na.rm = TRUE)),
      wMAPE_pop        = weighted.mean(MAPE, Pop, na.rm = TRUE),   # MAPE (%) weighted by under-5 reference population
      MAD              = median(AbsError, na.rm = TRUE),           # MEDIAN absolute error, median(|pred - obs|) (robust to a few huge misses); the mean version is MAE
      bias_pred_over_obs = sum(Predicted, na.rm = TRUE) / sum(Observed, na.rm = TRUE),  # < 1: predicts too low
      # log-scale metrics (log(1 + x)): symmetric between over- and under-prediction, and relative, so the largest
      # districts do not dominate. exp(MALE) - 1 ~ typical % error; MeanLogErr < 0: predicts too low.
      RMSLE            = sqrt(mean(LogErr^2, na.rm = TRUE)),
      MALE             = mean(abs(LogErr), na.rm = TRUE),
      MedALE           = median(abs(LogErr), na.rm = TRUE),
      MeanLogErr       = mean(LogErr, na.rm = TRUE),
      eff_n_districts  = (sum(Pop[!is.na(SqError)])^2) / sum(Pop[!is.na(SqError)]^2),
      .groups = "drop"
    )

  by_district <- list()
  out <- base
  for (b in names(benchmarks)) {
    ref <- dist_rmse %>%
      filter(Model == benchmarks[[b]]) %>%
      select(Point, District, RMSE_ref = RMSE_d, MAD_ref = MAD_d, RMSLE_ref = RMSLE_d)
    rel <- dist_rmse %>%
      left_join(ref, by = c("Point", "District")) %>%
      mutate(Benchmark = b,
             Rel     = ifelse(!is.na(RMSE_ref) & RMSE_ref > 0, RMSE_d / RMSE_ref, NA_real_),
             Rel_MAD = ifelse(!is.na(MAD_ref) & MAD_ref > 0, MAD_d / MAD_ref, NA_real_),   # same ratio, with MAD_d
             Rel_Log = ifelse(!is.na(RMSLE_ref) & RMSLE_ref > 0, RMSLE_d / RMSLE_ref, NA_real_))   # same ratio, on the log scale
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
    # relative log-scale RMSE: district RMSLE / benchmark's district RMSLE, averaged over districts
    summ_log <- rel %>%
      group_by(Model, Point) %>%
      summarise(a = mean(Rel_Log, na.rm = TRUE),
                g = { r <- Rel_Log[!is.na(Rel_Log) & Rel_Log > 0]; if (length(r)) exp(mean(log(r))) else NA_real_ },
                .groups = "drop")
    names(summ_log)[-(1:2)] <- c(paste0("RelRMSLE_mean_vs", b), paste0("RelRMSLE_geomean_vs", b))
    out <- left_join(out, summ, by = c("Model", "Point")) %>% left_join(summ_mad, by = c("Model", "Point")) %>%
      left_join(summ_log, by = c("Model", "Point"))
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
# 3. LLM-INFORMED MODELS (full data) -- no_pooling_LLM_informed_priors_monthly.stan
# Same SIS-derived variance as the hierarchical model, but the district-specific priors for
# p, gamma, omega are Beta(a, b) distributions elicited from an LLM and read directly
# from JSON (priors_llm/priors_llm_<llm>_<run>.json). One model per JSON, plus one
# POOLED model per LLM whose prior is the moment-matched mixture of that LLM's runs
# (pool_priors_moment_match()). Each model gets
# the posterior tables and fit statistics below. The hierarchical model's cached fit is
# added to the comparison tables when it exists.
# ============================================================
message("Models: ", sum(MODEL_REGISTRY$type == "llm_run"), " LLM runs + ",
        sum(MODEL_REGISTRY$type == "llm_pooled"), " pooled + ",
        sum(MODEL_REGISTRY$type == "all_llms_pooled"), " all-LLM pooled = ", nrow(MODEL_REGISTRY),
        if (HIER_AVAILABLE) " (+ the hierarchical model read from the sibling folder)" else "")

FULL <- list()    # cached full-data results by model id
hier_full <- if (HIER_AVAILABLE) fit_full_model(HIER_ID) else NULL
if (!is.null(hier_full)) FULL[[HIER_ID]] <- hier_full

LLM_PRIORS <- load_llm_priors()     # one prior per JSON + pooled per LLM; summary CSVs in SENS_OUT_DIR

llm_model_ids <- MODEL_REGISTRY$model_id
for (id in llm_model_ids) {
  full <- fit_full_model(id)
  if (!is.null(full)) {
    FULL[[id]] <- full
    write_full_outputs(id, full)
  } else {
    message("Skipping ", model_label(id), ": full-data fit not cached and not scheduled.")
  }
}

# ---- Posterior mean / median / 95% CI of every model (hierarchical included) in one table ----
if (length(FULL) > 0) {
  param_ci_all <- bind_rows(lapply(FULL, function(f) f$param_ci))
  write_csv(param_ci_all, "outputs/parameter_CI_hierarchical_vs_LLM_informed.csv")
  message("Saved: outputs/parameter_CI_hierarchical_vs_LLM_informed.csv (", length(FULL), " models)")
}

# ---- Pooled model of each LLM: objects used by the overlay below and by section 8 ----
pooled_full <- setNames(lapply(LLM_NAMES, function(llm) FULL[[paste0(llm, "_pooled")]]), LLM_NAMES)

# ---- Prior sensitivity across the repeated runs of each LLM ----
# Question: are the POSTERIORS materially different from run to run? Compared within
# each LLM (pairwise across runs) on: posterior median difference in posterior SDs,
# 1-Wasserstein distance in posterior SDs, KS statistic, overlap of the 95% credible
# intervals, Spearman correlation of the district medians across runs, and the same
# distance between the PRIORS. For p, gamma, omega and the district-mean 12-month
# burden. ELPD is exploratory only (see fit_stats). Needs >= 2 runs per LLM.
SENS_MEDIAN_DIFF_FLAG <- 0.5   # pairs "disagree" for a district above this many posterior SDs
sens_runs    <- list()
sens_run_ids <- list()
for (llm in LLM_NAMES) {
  reg  <- MODEL_REGISTRY[MODEL_REGISTRY$type == "llm_run" & MODEL_REGISTRY$llm %in% llm, ]
  runs <- reg$run[reg$model_id %in% names(FULL)]
  sens_run_ids[[llm]] <- runs
  sens_runs[[llm]] <- setNames(lapply(runs, function(r) {
    f <- FULL[[sprintf("%s_run%02d", llm, r)]]
    list(post = f$draws_thin, prior = f$prior, loo = f$loo, hmc = f$hmc, rhat_df = f$rhat_df)
  }), as.character(runs))
}

# ---- Comparison helpers ----
.SENS_PARAMS <- c("p", "gamma", "omega", "burden")
.SENS_U      <- (seq_len(1000) - 0.5) / 1000   # quantile grid for 1-D Wasserstein distances

# S x D matrix of posterior draws for one comparison quantity. The cached
# full-data results already hold "burden" as the per-draw mean of the district's
# 12-month burden over its observed timepoints.
.sens_param_draws <- function(run_obj, param) run_obj$post[[param]]

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
      LLM = llm, run = as.integer(k),
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


# Prior (dashed) and posterior (solid) densities within ONE LLM: every prompt run
# plus the pooled model, one panel per district. "burden" has no prior of its own,
# so it is posterior-only. Each panel's x-range spans the posteriors and the
# central 98% of every prior, so the priors are not clipped; where a prior is much
# wider than the posteriors, the posterior peaks dominate the panel's y-axis.
# `pooled` is the cached full-data result of the LLM's pooled model (or NULL).
plot_prior_sensitivity <- function(llm, runs, param, pooled = NULL, x_n = 300) {
  keys <- as.character(runs)
  log_scale <- param %in% c("gamma", "omega")
  has_prior <- param != "burden"

  # one list entry per curve set: runs, then "pooled"
  sets <- lapply(keys, function(k) list(name = k, post = sens_runs[[llm]][[k]]$post,
                                        prior = sens_runs[[llm]][[k]]$prior))
  if (!is.null(pooled)) sets[[length(sets) + 1]] <- list(name = "pooled", post = pooled$draws_thin, prior = pooled$prior)
  set_names <- vapply(sets, function(s) s$name, character(1))

  dens_rows <- list()
  for (di in seq_len(D)) {
    xs <- lapply(sets, function(s) {
      v <- s$post[[param]][, di]
      if (log_scale) log(v[v > 0]) else v
    })
    rng <- quantile(unlist(xs), c(0.001, 0.999), names = FALSE)

    prior_ab <- NULL
    if (has_prior) {
      prior_ab <- lapply(sets, function(s) s$prior[[district_names[di]]][[param]])
      q_prior  <- unlist(lapply(prior_ab, function(ab) qbeta(c(0.01, 0.99), ab$a, ab$b)))
      if (log_scale) q_prior <- log(q_prior[q_prior > 0])
      rng <- range(rng, q_prior)
    }
    x_grid <- seq(rng[1], rng[2], length.out = x_n)

    for (i in seq_along(sets)) {
      kd <- density(xs[[i]], from = rng[1], to = rng[2], n = x_n)
      dens_rows[[length(dens_rows) + 1]] <- tibble(
        District = district_names[di], Run = set_names[i], Type = "Posterior", x = kd$x, density = kd$y
      )
      if (has_prior) {
        ab <- prior_ab[[i]]
        pd <- if (log_scale) dbeta(exp(x_grid), ab$a, ab$b) * exp(x_grid) else dbeta(x_grid, ab$a, ab$b)
        pd[!is.finite(pd)] <- NA
        dens_rows[[length(dens_rows) + 1]] <- tibble(
          District = district_names[di], Run = set_names[i], Type = "Prior", x = x_grid, density = pd
        )
      }
    }
  }
  plot_df <- bind_rows(dens_rows) %>%
    filter(!is.na(density)) %>%
    mutate(District = factor(District, levels = district_names),
           Run      = factor(Run, levels = set_names),
           Type     = factor(Type, levels = c("Prior", "Posterior")))

  x_lab <- if (log_scale) paste0("log(", param, ")") else
           if (param == "burden") "district-mean 12-month burden" else param

  run_cols <- setNames(viridisLite::viridis(max(length(keys), 1), end = 0.9), keys)
  if (!is.null(pooled)) run_cols <- c(run_cols, pooled = "black")

  p_out <- ggplot(plot_df, aes(x = x, y = density, color = Run, linetype = Type,
                               group = interaction(Run, Type))) +
    geom_line(aes(linewidth = (Run == "pooled")), alpha = 0.9) +
    scale_linewidth_manual(values = c(`FALSE` = 0.45, `TRUE` = 0.9), guide = "none") +
    facet_wrap(~District, scales = "free", ncol = 9) +
    scale_color_manual(values = run_cols, name = "Prompt run") +
    scale_linetype_manual(values = c(Prior = "dashed", Posterior = "solid")) +
    labs(
      title    = paste0("Prior vs posterior of ", x_lab, " within ", llm, ": every prompt run and the pooled model"),
      subtitle = paste0("Dashed = prior, solid = posterior, one color per run (pooled in black); panels use their own axes",
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


# ---- Fit health / ELPD table of the single runs (exploratory ELPD, see fit_stats) ----
llms_with_runs <- LLM_NAMES[vapply(LLM_NAMES, function(l) length(sens_run_ids[[l]]) >= 1, logical(1))]
if (length(llms_with_runs) > 0) {
  sens_fit_df <- bind_rows(lapply(llms_with_runs, function(llm) .sens_fit_table(llm, sens_run_ids[[llm]])))
  write_csv(sens_fit_df, file.path(SENS_OUT_DIR, "sensitivity_fit_summary.csv"))
  message("Saved: ", file.path(SENS_OUT_DIR, "sensitivity_fit_summary.csv"))
  print(as.data.frame(sens_fit_df))
}

# ---- Pairwise posterior comparison across runs (needs >= 2 runs for an LLM) ----
llms_to_compare <- LLM_NAMES[vapply(LLM_NAMES, function(l) length(sens_run_ids[[l]]) >= 2, logical(1))]
if (length(llms_to_compare) == 0) {
  message("Prior sensitivity: fewer than two cached runs per LLM -- fit more runs to compare them.")
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

# ---- Prior-posterior overlay within each LLM: every run plus the pooled model ----
for (llm in llms_with_runs) {
  for (param in .SENS_PARAMS) {
    plot_prior_sensitivity(llm, sens_run_ids[[llm]], param, pooled = pooled_full[[llm]])
  }
}

# ============================================================
# End of the LLM-informed models
# ============================================================

# -------------------------------------------------
# 4. CHAIN STATISTICS FOR EVERY MODEL
# -------------------------------------------------
# One table row per model x scheme (full data, LLO-2, LLO-3) from the cache: divergences,
# max-treedepth hits, low-E-BFMI chains, max Rhat, min ESS, fit minutes. Trace and density
# plots are made for the pooled models only (the cached trace array holds p[1], gamma[1],
# omega[1], beta0[1]). The table is refreshed again after the CV sections (section 6),
# because the CV fits do not exist yet when this section first runs.
report_chain_stats <- function() {
  rows <- list()
  for (id in ALL_MODEL_IDS) {
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

# ---- Trace and density overlays: pooled model of each LLM ----
for (llm in LLM_NAMES) {
  f <- pooled_full[[llm]]
  if (is.null(f)) next
  ggsave(paste0("plots/trace_plot_", llm, "_pooled.jpg"),
         mcmc_trace(f$trace_array), width = 12, height = 6, dpi = 300)
  ggsave(paste0("plots/density_overlay_", llm, "_pooled.jpg"),
         mcmc_dens_overlay(f$trace_array), width = 8, height = 6, dpi = 300)
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
              CRPS_mean = mean(CRPS, na.rm = TRUE), .groups = "drop")   # MAD and bias come with `rel`
  out <- left_join(rel, extra, by = c("Model", "Point")) %>%
    mutate(scheme = scheme, divergences = r$hmc$divergent, max_treedepth_hits = r$hmc$max_treedepth,
           low_bfmi_chains = r$hmc$low_bfmi_chains,
           max_rhat = max(r$rhat_df$Rhat, na.rm = TRUE), min_n_eff = min(r$rhat_df$n_eff, na.rm = TRUE),
           fit_minutes = r$fit_minutes) %>%
    relocate(scheme, .after = Model)
  write_csv(out, file.path(dir_out, paste0(lab, ".csv")))
  invisible(out)
}

# All headline metrics of a long score table (Model, District, Point, Predicted, Observed, AbsError, SqError,
# MAPE, CRPS incl. the benchmarks): relative scores + RMSE, wRMSE, wMAPE, MAD, bias, MAE, MAPE, CRPS.
.metric_table <- function(all_df) {
  rel <- compute_cv_relative_scores(all_df, setNames(as.numeric(pop_u5_ref), district_names))$summary
  extra <- bind_rows(all_df %>% mutate(Point = as.character(Point)), all_df %>% mutate(Point = "All")) %>%
    group_by(Model, Point) %>%
    summarise(MAE = mean(AbsError, na.rm = TRUE), MAPE_mean = mean(MAPE, na.rm = TRUE),
              CRPS_mean = mean(CRPS, na.rm = TRUE), .groups = "drop")
  left_join(rel, extra, by = c("Model", "Point"))
}

# ---- Are the measures driven by a few districts? ----
# Per model and held-out point (and all points): RMSE, MAE, MAD (median absolute error), RMSE/MAD (about
# 1.5 for well-behaved errors; much larger when a few huge misses dominate), the share of the total squared
# error that sits in the model's own k worst districts, and the RMSE after dropping the k districts with the
# largest observed SAM (the same districts for every model). Plots: cumulative share of squared error,
# the district-level error distribution, RMSE after dropping the largest districts, RMSE / MAE / MAD.
# Output: outputs/<TAG>_CV_outlier_diagnostics.csv and plots/<TAG>_CV_outliers_*.jpg
report_cv_outlier_diagnostics <- function(all_df, tag, sp, top_k = c(1, 3, 5, 10),
                                          models_shown = c(MODEL_LABELS, "Naive", "HistAverage")) {
  long <- bind_rows(all_df %>% mutate(Point = as.character(Point)), all_df %>% mutate(Point = "All"))
  big  <- all_df %>% group_by(District) %>% summarise(m = mean(Observed, na.rm = TRUE), .groups = "drop") %>%
    arrange(desc(m)) %>% pull(District)           # districts from the largest observed SAM down

  diag_tab <- long %>% group_by(Model, Point) %>% group_modify(function(d, key) {
    se_d <- d %>% group_by(District) %>% summarise(se = sum(SqError, na.rm = TRUE), .groups = "drop") %>% arrange(desc(se))
    out <- tibble(RMSE = sqrt(mean(d$SqError, na.rm = TRUE)), MAE = mean(d$AbsError, na.rm = TRUE),
                  MAD = median(d$AbsError, na.rm = TRUE))
    out$RMSE_over_MAD <- out$RMSE / out$MAD
    for (k in top_k) out[[paste0("share_SSE_worst", k)]] <- sum(head(se_d$se, k)) / sum(se_d$se)
    for (k in top_k) out[[paste0("RMSE_without_largest", k)]] <- sqrt(mean(d$SqError[!(d$District %in% head(big, k))], na.rm = TRUE))
    out
  }) %>% ungroup()
  write_csv(diag_tab, sprintf("outputs/%s_CV_outlier_diagnostics.csv", tag))
  message("Saved: outputs/", tag, "_CV_outlier_diagnostics.csv")

  cols <- c(MODEL_COLORS, Naive = "grey50", HistAverage = "darkgreen")
  labs_ <- c(MODEL_DISPLAY, Naive = "Naive", HistAverage = "Hist. Average")
  shown <- long %>% filter(Model %in% models_shown)
  base_theme <- theme_minimal(base_size = 9) + theme(legend.position = "bottom")

  # 1. cumulative share of the squared error by number of districts (districts sorted worst first, per model)
  cum <- shown %>% group_by(Model, Point, District) %>% summarise(se = sum(SqError, na.rm = TRUE), .groups = "drop") %>%
    group_by(Model, Point) %>% arrange(desc(se), .by_group = TRUE) %>%
    mutate(rank = row_number(), cum_share = cumsum(se) / sum(se)) %>% ungroup()
  p1 <- ggplot(cum, aes(rank, cum_share, color = Model)) +
    geom_abline(slope = 1 / length(big), intercept = 0, linetype = "dotted", color = "grey40") +
    geom_line(linewidth = 0.7) + facet_wrap(~Point) +
    scale_color_manual(values = cols, labels = labs_, name = NULL) +
    labs(title = paste0(sp$title, ": how concentrated is the squared error?"),
         subtitle = "Share of the total squared error in the k worst districts of each model (dotted = all districts equal); a steep start = a few districts drive RMSE",
         x = "Number of districts (worst first)", y = "Cumulative share of squared error") + base_theme
  ggsave(sprintf("plots/%s_CV_outliers_error_concentration.jpg", tag), p1, width = 11, height = 6, dpi = 250)

  # 2. district-level absolute errors (pseudo-log axis), the 3 worst districts of each model labelled
  d2 <- shown %>% filter(Point == "All") %>% group_by(Model, District) %>%
    summarise(AbsErr = mean(AbsError, na.rm = TRUE), .groups = "drop")
  lab2 <- d2 %>% group_by(Model) %>% slice_max(AbsErr, n = 3, with_ties = FALSE) %>% ungroup()
  p2 <- ggplot(d2, aes(Model, AbsErr, color = Model)) +
    geom_boxplot(outlier.shape = NA, fill = NA, linewidth = 0.4) +
    geom_jitter(width = 0.15, size = 0.8, alpha = 0.6) +
    geom_text(data = lab2, aes(label = District), size = 2.3, nudge_x = 0.32, check_overlap = TRUE, show.legend = FALSE) +
    scale_y_continuous(trans = scales::pseudo_log_trans(base = 10), breaks = c(0, 10, 100, 1000, 10000)) +
    scale_color_manual(values = cols, guide = "none") + scale_x_discrete(labels = labs_) +
    labs(title = paste0(sp$title, ": absolute error of each district (mean over its held-out points)"),
         subtitle = "One dot per district, log-type axis; labels = the 3 worst districts of each model", x = NULL, y = "Absolute error (SAM)") +
    base_theme + theme(axis.text.x = element_text(angle = 20, hjust = 1))
  ggsave(sprintf("plots/%s_CV_outliers_district_errors.jpg", tag), p2, width = 10, height = 6, dpi = 250)

  # 3. RMSE after dropping the k districts with the largest observed SAM
  ks <- c(0, 1, 2, 3, 5, 10, 15, 20)
  d3 <- bind_rows(lapply(ks, function(k) shown %>% filter(Point == "All", !(District %in% head(big, k))) %>%
    group_by(Model) %>% summarise(RMSE = sqrt(mean(SqError, na.rm = TRUE)), .groups = "drop") %>% mutate(k = k)))
  p3 <- ggplot(d3, aes(k, RMSE, color = Model)) + geom_line(linewidth = 0.7) + geom_point(size = 1.3) +
    scale_color_manual(values = cols, labels = labs_, name = NULL) + scale_x_continuous(breaks = ks) +
    labs(title = paste0(sp$title, ": RMSE after dropping the k districts with the largest observed SAM"),
         subtitle = "If the ranking of the models changes with k, it is decided by the large districts",
         x = "Districts dropped (largest first)", y = "RMSE (all held-out points)") + base_theme
  ggsave(sprintf("plots/%s_CV_outliers_RMSE_drop_largest.jpg", tag), p3, width = 10, height = 6, dpi = 250)

  # 4. RMSE vs MAE vs MAD: a MAD far below RMSE means a few big misses drive the squared error
  d4 <- diag_tab %>% filter(Point == "All", Model %in% models_shown) %>% select(Model, RMSE, MAE, MAD) %>%
    tidyr::pivot_longer(c(RMSE, MAE, MAD), names_to = "Measure", values_to = "Value") %>%
    mutate(Measure = factor(Measure, levels = c("RMSE", "MAE", "MAD")))
  p4 <- ggplot(d4, aes(Model, Value, fill = Measure)) + geom_col(position = "dodge") +
    scale_x_discrete(labels = labs_) + scale_fill_manual(values = c(RMSE = "#2a78d6", MAE = "#1baf7a", MAD = "#eb6834"), name = NULL) +
    labs(title = paste0(sp$title, ": RMSE, MAE and MAD (median absolute error), all held-out points"),
         x = NULL, y = "SAM") + base_theme + theme(axis.text.x = element_text(angle = 20, hjust = 1))
  ggsave(sprintf("plots/%s_CV_outliers_RMSE_MAE_MAD.jpg", tag), p4, width = 10, height = 6, dpi = 250)
  message("Saved: plots/", tag, "_CV_outliers_*.jpg (4 files)")
  invisible(diag_tab)
}

# ---- In-sample (training window) scores next to the held-out scores, per LLM ----
# `res` = the cached CV objects of this scheme, `all_df` = the held-out scores. Only models whose CV object
# carries a `train_scores` record (see TRAIN_EVAL_MODELS) are scored. Outputs:
#   outputs/<TAG>_train_all_models.csv            per district x training point
#   outputs/<TAG>_train_vs_test_summary_by_model.csv   all headline metrics, in-sample and held-out
#   outputs/<TAG>_train_vs_test_ranking.csv       rank of each LLM (All points) on each metric, in-sample vs held-out
#   plots/<TAG>_train_vs_test_by_llm.jpg
# In-sample numbers describe how well the prior + training data reproduce the training points one step ahead;
# they use no held-out point, so they can be used to choose between LLMs.
run_train_eval <- function(scheme, res, all_df) {
  tag <- toupper(scheme)
  tr  <- Filter(Negate(is.null), lapply(res, function(r) r$train_scores))
  if (length(tr) == 0) { message(tag, ": no in-sample records yet (TRAIN_EVAL_MODELS not fitted for this scheme)."); return(invisible(NULL)) }
  train_df <- bind_rows(bind_rows(tr), .train_benchmark_rows(scheme))
  write_csv(train_df, sprintf("outputs/%s_train_all_models.csv", tag))
  models <- unique(bind_rows(tr)$Model)
  m_train <- .metric_table(train_df) %>% mutate(Sample = "train (in-sample)")
  m_test  <- .metric_table(all_df %>% filter(Model %in% c(models, "Naive", "HistAverage"))) %>% mutate(Sample = "test (held-out)")
  tab <- bind_rows(m_train, m_test) %>% relocate(Sample, .after = Point)
  write_csv(tab, sprintf("outputs/%s_train_vs_test_summary_by_model.csv", tag))
  message("Saved: outputs/", tag, "_train_all_models.csv and _train_vs_test_summary_by_model.csv")
  print(as.data.frame(tab %>% filter(Point == "All") %>%
    select(Model, Sample, RMSE, wRMSE_pop, MAD, MAE, MAPE_mean, wMAPE_pop, CRPS_mean, RelRMSE_geomean_vsPersistence, bias_pred_over_obs) %>%
    mutate(across(where(is.numeric), ~ round(.x, 3))) %>% arrange(Sample, RMSE)))

  metrics <- c(RMSE = "RMSE", wRMSE = "wRMSE_pop", MAD = "MAD", MAE = "MAE", MAPE = "MAPE_mean", wMAPE = "wMAPE_pop",
               CRPS = "CRPS_mean", geoP = "RelRMSE_geomean_vsPersistence", arithP = "RelRMSE_mean_vsPersistence",
               `geoP (MAD)` = "RelMAD_geomean_vsPersistence", `arithP (MAD)` = "RelMAD_mean_vsPersistence", `|bias-1|` = "abs_bias",
               RMSLE = "RMSLE", MALE = "MALE", MedALE = "MedALE", `|mean log err|` = "abs_logbias", `geoP (log)` = "RelRMSLE_geomean_vsPersistence")
  tab <- tab %>% mutate(abs_bias = abs(bias_pred_over_obs - 1), abs_logbias = abs(MeanLogErr))
  llm_tab <- tab %>% filter(Point == "All", Model %in% models)
  long <- llm_tab %>% select(Model, Sample, all_of(unname(metrics))) %>%
    tidyr::pivot_longer(-c(Model, Sample), names_to = "col", values_to = "Value") %>%
    mutate(Metric = factor(names(metrics)[match(col, metrics)], levels = names(metrics))) %>%
    group_by(Sample, Metric) %>% mutate(Rank = rank(Value, ties.method = "min", na.last = "keep")) %>% ungroup()
  rk <- long %>% select(Metric, Model, Sample, Value, Rank) %>%
    tidyr::pivot_wider(names_from = Sample, values_from = c(Value, Rank)) %>% arrange(Metric, Model)
  write_csv(rk, sprintf("outputs/%s_train_vs_test_ranking.csv", tag))
  message("Saved: outputs/", tag, "_train_vs_test_ranking.csv")

  bench <- tab %>% filter(Point == "All", Model %in% c("Naive", "HistAverage")) %>% mutate(abs_bias = abs(bias_pred_over_obs - 1)) %>%
    select(Model, Sample, all_of(unname(metrics))) %>% tidyr::pivot_longer(-c(Model, Sample), names_to = "col", values_to = "Value") %>%
    mutate(Metric = factor(names(metrics)[match(col, metrics)], levels = names(metrics))) %>% filter(!is.na(Value))
  p <- ggplot(long, aes(Model, Value, fill = Sample)) + geom_col(position = "dodge") +
    geom_hline(data = bench %>% filter(Model == "Naive"), aes(yintercept = Value, linetype = "Naive"), color = "grey30", linewidth = 0.4) +
    facet_grid(Metric ~ ., scales = "free_y") + scale_linetype_manual(values = c(Naive = "dashed"), name = NULL) +
    scale_fill_manual(values = c(`train (in-sample)` = "#2a78d6", `test (held-out)` = "#eb6834"), name = NULL) +
    labs(title = paste0(CV_SPECS[[scheme]]$title, ": pooled LLM priors, in-sample vs held-out (all points)"),
         subtitle = "Dashed line = Naive on the same points. Metrics are comparable in rank within a sample; levels differ between samples",
         x = NULL, y = NULL) + theme_minimal(base_size = 8) + theme(legend.position = "bottom")
  ggsave(sprintf("plots/%s_train_vs_test_by_llm.jpg", tag), p, width = 7, height = 22, dpi = 220)
  message("Saved: plots/", tag, "_train_vs_test_by_llm.jpg")
  invisible(tab)
}

# ---- Pooled LLM comparison: every metric by point, and the average rank across the metrics ----
# For the three pooled LLM models (the prior of each LLM is the moment-matched pool of its 10 runs), on each
# point separately and on all points pooled: the values of all metrics, the rank of each LLM on each metric
# (1 = best; for bias the distance of pred/obs from 1), and the average rank across the 10 metrics.
# In-sample = the training points t2.. (needs the in-sample records, see TRAIN_EVAL); held-out = the CV points.
# Outputs: outputs/<TAG>_pooled_llm_by_point.csv           all metric values, per Sample x Model x Point
#          outputs/<TAG>_pooled_llm_ranks_by_point.csv     rank of each LLM on each metric, per Sample x Point
#          outputs/<TAG>_pooled_llm_average_rank.csv       average rank across the metrics (+ mean over the single points)
POOLED_RANK_METRICS <- c(RMSE = "RMSE", wRMSE = "wRMSE_pop", MAE = "MAE", MAD = "MAD", MAPE = "MAPE_mean", wMAPE = "wMAPE_pop",
                         `|bias-1|` = "abs_bias", CRPS = "CRPS_mean", geoP = "RelRMSE_geomean_vsPersistence",
                         arithP = "RelRMSE_mean_vsPersistence",
                         # log-scale metrics (symmetric between over- and under-prediction)
                         RMSLE = "RMSLE", MALE = "MALE", MedALE = "MedALE", `|mean log err|` = "abs_logbias",
                         `geoP (log)` = "RelRMSLE_geomean_vsPersistence")
report_pooled_llm_comparison <- function(scheme, res, all_df) {
  tag  <- toupper(scheme)
  llms <- c(paste0(LLM_NAMES, "_pooled"), ALL_LLM_ID)   # the three LLM-pooled models and the pool of the three
  tr   <- Filter(Negate(is.null), lapply(res, function(r) r$train_scores))
  parts <- list()
  if (length(tr) > 0) {
    train_df <- bind_rows(bind_rows(tr), .train_benchmark_rows(scheme))
    parts$train <- .metric_table(train_df) %>% filter(Model %in% llms) %>% mutate(Sample = "train (in-sample)")
  } else message(tag, ": no in-sample records -- the pooled-LLM comparison covers the held-out points only.")
  parts$test <- .metric_table(all_df %>% filter(Model %in% c(llms, "Naive", "HistAverage"))) %>%
    filter(Model %in% llms) %>% mutate(Sample = "test (held-out)")
  tab <- bind_rows(parts) %>% mutate(abs_bias = abs(bias_pred_over_obs - 1), abs_logbias = abs(MeanLogErr)) %>% relocate(Sample, .before = Model)
  write_csv(tab, sprintf("outputs/%s_pooled_llm_by_point.csv", tag))

  long <- tab %>% select(Sample, Model, Point, all_of(unname(POOLED_RANK_METRICS))) %>%
    tidyr::pivot_longer(-c(Sample, Model, Point), names_to = "col", values_to = "Value") %>%
    mutate(Metric = factor(names(POOLED_RANK_METRICS)[match(col, POOLED_RANK_METRICS)], levels = names(POOLED_RANK_METRICS))) %>%
    group_by(Sample, Point, Metric) %>% mutate(Rank = rank(Value)) %>% ungroup() %>% select(-col)
  write_csv(long %>% arrange(Sample, Point, Metric, Rank), sprintf("outputs/%s_pooled_llm_ranks_by_point.csv", tag))

  avg <- long %>% group_by(Sample, Point, Model) %>% summarise(AvgRank = mean(Rank), .groups = "drop")
  avg_pts <- avg %>% filter(Point != "All") %>% group_by(Sample, Model) %>%
    summarise(AvgRank = mean(AvgRank), .groups = "drop") %>% mutate(Point = "mean of the points")
  avg_all <- bind_rows(avg, avg_pts) %>% tidyr::pivot_wider(names_from = Model, values_from = AvgRank) %>% arrange(Sample, Point)
  write_csv(avg_all, sprintf("outputs/%s_pooled_llm_average_rank.csv", tag))
  message("Saved: outputs/", tag, "_pooled_llm_by_point.csv, _pooled_llm_ranks_by_point.csv, _pooled_llm_average_rank.csv")
  cat("\n", tag, ": average rank across", length(POOLED_RANK_METRICS), "metrics (1 = best)\n")
  print(as.data.frame(avg_all %>% mutate(across(where(is.numeric), ~ round(.x, 2)))), row.names = FALSE)
  invisible(avg_all)
}

# ---- Every metric by district, one panel per measure, for each held-out point ----
# One dot per district (x = the district's observed SAM, log scale, so the large districts -- the outlier
# candidates -- sit on the right) and one colour per model: the hierarchical model, the three pooled LLM models and
# Naive. Dashed (and dotted) horizontal lines = the reported value of the measure for each model, the same numbers
# as in the tables. Panels: A absolute error (dashed RMSE, dotted MAD), B signed error (dashed = mean),
# C absolute % error (dashed MAPE, dotted wMAPE), D CRPS (dashed = mean), E error relative to Naive (dashed geoP),
# F cumulative share of the squared error. Output, per point: plots/measures_by_district/<TAG>_<point>_all_measures.jpg
# (all panels) and one file per measure.
N_LABEL_DISTRICTS <- 4    # number of largest districts labelled in the plots
.fmt_val <- function(x) ifelse(abs(x) >= 100, formatC(x, format = "f", digits = 0, big.mark = ","), formatC(x, format = "f", digits = 2))

.measure_scatter <- function(df, ycol, title, ylab, show, cols, labs_, logy = TRUE, hline = NULL, big, lines = NULL) {
  df$y <- df[[ycol]]
  if (logy) df <- df[is.finite(df$y) & df$y > 0, ]
  p <- ggplot(df, aes(Observed, y, colour = Model)) +
    { if (!is.null(hline)) geom_hline(yintercept = hline, colour = "grey60", linetype = 2) } +
    { if (!is.null(lines)) {
        lines <- lines[lines$Model %in% df$Model & is.finite(lines$y) & (!logy | lines$y > 0), ]
        lines$name <- factor(lines$name, unique(lines$name))
        v <- lines %>% arrange(Model, name) %>% group_by(Model) %>% summarise(txt = paste(.fmt_val(y), collapse = "  /  "), .groups = "drop")
        v$k <- as.numeric(factor(v$Model, show[show %in% v$Model])); v$txt <- paste0(labs_[as.character(v$Model)], ":  ", v$txt)
        list(geom_hline(data = lines, aes(yintercept = y, colour = Model, linetype = name), linewidth = 0.7, alpha = 0.9),
             scale_linetype_manual(values = c("dashed", "dotted")[seq_along(levels(lines$name))], guide = "none"),
             geom_label(data = v, aes(x = min(df$Observed), y = Inf, label = txt, colour = Model, vjust = 1.25 * k - 0.05), hjust = 0, size = 2.7,
                        fontface = "bold", fill = scales::alpha("white", 0.85), linewidth = 0, label.padding = unit(0.08, "lines"), show.legend = FALSE))
      } } +
    geom_point(alpha = 0.75, size = 1.6) +
    geom_text_repel(data = df[df$Model == show[1] & df$District %in% big, ], aes(label = District), colour = "black", size = 2.6,
                    min.segment.length = 0, box.padding = 0.4, max.overlaps = Inf, seed = 1) +
    scale_colour_manual(values = cols, labels = labs_, name = NULL) +
    scale_x_log10(labels = scales::comma) +
    labs(title = title, x = "Observed SAM (log scale)", y = ylab) +
    theme_minimal(base_size = 11) + theme(legend.position = "none", plot.title = element_text(face = "bold", size = 11))
  if (logy) p + scale_y_log10(labels = scales::comma) else p + scale_y_continuous(labels = scales::comma)
}

plot_measures_by_district <- function(scheme, all_df) {
  sp <- CV_SPECS[[scheme]]; tag <- toupper(scheme)
  show  <- c("Naive", MODEL_LABELS)
  cols  <- c(Naive = "grey60", MODEL_COLORS)
  labs_ <- c(Naive = "Naive", MODEL_DISPLAY)
  out_dir <- "plots/measures_by_district"; dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)
  mt <- .metric_table(all_df %>% filter(Model %in% c(show, "HistAverage")))   # HistAverage: needed by the relative scores
  for (point in sp$points) {
    d <- all_df %>% filter(Point == point, Model %in% show) %>% mutate(Model = factor(Model, show), Error = Predicted - Observed)
    if (nrow(d) == 0) next
    nv <- d %>% filter(Model == "Naive") %>% select(District, NaiveAbs = AbsError)
    d <- left_join(d, nv, by = "District") %>% mutate(RelAbs = ifelse(NaiveAbs > 0, AbsError / NaiveAbs, NA_real_))
    big <- d %>% filter(Model == "Naive") %>% slice_max(Observed, n = N_LABEL_DISTRICTS) %>% pull(District)
    rel <- mt %>% filter(Point == point)
    L  <- function(col, nm) data.frame(Model = factor(rel$Model, show), y = rel[[col]], name = nm)
    sg <- d %>% group_by(Model) %>% summarise(MeanSigned = mean(Error), .groups = "drop")
    Ls <- data.frame(Model = sg$Model, y = sg$MeanSigned, name = "mean signed error")

    cum <- d %>% group_by(Model) %>% arrange(desc(SqError), .by_group = TRUE) %>%
      mutate(Rank = row_number(), Share = cumsum(SqError) / sum(SqError)) %>% ungroup()
    pF <- ggplot(cum, aes(Rank, Share, colour = Model)) + geom_line(linewidth = 0.9) +
      scale_colour_manual(values = cols, labels = labs_, name = NULL) + scale_y_continuous(labels = scales::percent, limits = c(0, 1)) +
      labs(title = "F. Cumulative share of squared error (RMSE)", x = "Districts, largest squared error first", y = "Share of total squared error") +
      theme_minimal(base_size = 11) + theme(legend.position = "none", plot.title = element_text(face = "bold", size = 11))

    pl <- list(
      A_abs_error     = .measure_scatter(d, "AbsError", "A. Absolute error: dashed = RMSE, dotted = MAD", "Absolute error", show, cols, labs_, TRUE, NULL, big, rbind(L("RMSE", "RMSE"), L("MAD", "MAD"))),
      B_signed_error  = .measure_scatter(d, "Error", "B. Signed error: dashed = mean", "Predicted - observed", show, cols, labs_, FALSE, 0, big, Ls),
      C_abs_pct_error = .measure_scatter(d, "MAPE", "C. Absolute % error: dashed = MAPE, dotted = wMAPE", "Absolute % error", show, cols, labs_, TRUE, NULL, big, rbind(L("MAPE_mean", "MAPE"), L("wMAPE_pop", "wMAPE"))),
      D_CRPS          = .measure_scatter(d %>% filter(!is.na(CRPS)), "CRPS", "D. CRPS: dashed = mean CRPS", "CRPS (counts)", show, cols, labs_, TRUE, NULL, big, L("CRPS_mean", "mean CRPS")[!is.na(rel$CRPS_mean), ]),
      E_rel_to_naive  = .measure_scatter(d %>% filter(Model != "Naive"), "RelAbs", "E. Abs. error / Naive abs. error: dashed = geoP", "Ratio to Naive (log scale)", show, cols, labs_, TRUE, 1, big, L("RelRMSE_geomean_vsPersistence", "geoP")),
      F_cum_sq_share  = pF)
    leg <- cowplot::get_legend(pl[[1]] + theme(legend.position = "bottom", legend.text = element_text(size = 10)))
    ttl <- cowplot::ggdraw() + cowplot::draw_label(paste0(sp$title, ", held-out ", point, " point: every measure by district"), fontface = "bold", x = 0.01, hjust = 0, size = 13)
    sub <- cowplot::ggdraw() + cowplot::draw_label(paste0("Labels: the ", N_LABEL_DISTRICTS, " districts with the largest observed SAM; dots far from the rest on the right of panels A-E are the outlier candidates"), x = 0.01, hjust = 0, size = 9.5, colour = "grey30")
    allp <- cowplot::plot_grid(ttl, sub, cowplot::plot_grid(plotlist = unname(pl), ncol = 3), leg, ncol = 1, rel_heights = c(0.05, 0.04, 1, 0.06))
    ggsave(file.path(out_dir, sprintf("%s_%s_all_measures.jpg", tag, point)), allp, width = 17, height = 10, dpi = 150, bg = "white")
    for (nm in names(pl)) {
      one <- pl[[nm]] + theme(legend.position = "bottom") + labs(subtitle = paste0(sp$title, ", held-out ", point, " point"))
      ggsave(file.path(out_dir, sprintf("%s_%s_%s.jpg", tag, point, nm)), one, width = 8, height = 5.5, dpi = 150, bg = "white")
    }
  }
  message("Saved: ", out_dir, "/", tag, "_<point>_all_measures.jpg and one file per measure (", paste(sp$points, collapse = ", "), ")")
  invisible(NULL)
}

# One CV section: fit (cached) every model, then tables, relative scores and plots.
# Tables cover every cached model (base, each LLM run, each pooled model) plus the
# benchmarks; the bar plots and district panels show the hierarchical model, the pooled models and
# the benchmarks so they stay readable.
run_cv_section <- function(scheme) {
  sp  <- CV_SPECS[[scheme]]
  tag <- toupper(scheme)

  res <- list()
  for (id in ALL_MODEL_IDS) {
    r <- fit_cv_model(scheme, id)
    if (is.null(r)) { message("Skipping ", model_label(id), " (", tag, "): fit not cached and not scheduled."); next }
    r$scores <- .drop_excluded(r$scores)                       # metrics skip EXCLUDE_DISTRICTS (see above)
    if (!is.null(r$train_scores)) r$train_scores <- .drop_excluded(r$train_scores)
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
      MAD       = median(AbsError, na.rm = TRUE),       # MEDIAN absolute error (the mean version is MAE)
      MAPE_mean = mean(MAPE, na.rm = TRUE),
      CRPS_mean = mean(CRPS, na.rm = TRUE),
      # log-scale metrics on log(1 + x): symmetric between over- and under-prediction
      RMSLE     = sqrt(mean((log1p(pmax(Predicted, 0)) - log1p(pmax(Observed, 0)))^2, na.rm = TRUE)),
      MALE      = mean(abs(log1p(pmax(Predicted, 0)) - log1p(pmax(Observed, 0))), na.rm = TRUE),
      MedALE    = median(abs(log1p(pmax(Predicted, 0)) - log1p(pmax(Observed, 0))), na.rm = TRUE),
      .groups   = "drop"
    ) %>%
    arrange(Point, RMSE)
  print(as.data.frame(summary_df %>% filter(Point == "All")))
  write_csv(summary_df, sprintf("outputs/%s_CV_summary_by_model.csv", tag))
  message("Saved: outputs/", tag, "_CV_summary_by_model.csv")

  # --- Relative (per district, then averaged) and population-weighted scores ---
  report_cv_relative_scores(all_df, tag, paste0(sp$title, ": relative RMSE by district"))

  # --- Are the measures driven by a few districts? (tables + plots) ---
  report_cv_outlier_diagnostics(all_df, tag, sp)

  # --- In-sample (training window) scores of the pooled models, next to the held-out scores ---
  run_train_eval(scheme, res, all_df)

  # --- Pooled LLMs: every metric by point (in-sample and held-out) and the average rank across metrics ---
  report_pooled_llm_comparison(scheme, res, all_df)

  # --- Every metric by district, one panel per measure, for each held-out point ---
  plot_measures_by_district(scheme, all_df)

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

  # Overview of every model (all held-out points): RMSE and mean CRPS side by side
  ov <- summary_df %>%
    filter(Point == "All", !(Model %in% c("Naive", "HistAverage"))) %>%
    mutate(Group  = case_when(Model == "hierarchical" ~ "hierarchical", Model == ALL_LLM_ID ~ "All 3 LLMs",
                              grepl("claude", Model) ~ "Claude",
                              grepl("gemini", Model) ~ "Gemini", TRUE ~ "ChatGPT"),
           Pooled = grepl("pooled", Model)) %>%
    select(Model, Group, Pooled, RMSE, CRPS_mean) %>%
    tidyr::pivot_longer(c(RMSE, CRPS_mean), names_to = "Metric", values_to = "Value")
  bench <- summary_df %>% filter(Point == "All", Model %in% c("Naive", "HistAverage")) %>%
    transmute(Model, Metric = "RMSE", Value = RMSE)
  p_over <- ggplot(ov, aes(x = Value, y = reorder(Model, -Value), color = Group, shape = Pooled)) +
    geom_point(size = 2) +
    geom_vline(data = bench, aes(xintercept = Value, linetype = Model), color = "grey40") +
    facet_wrap(~Metric, scales = "free_x") +
    labs(title = paste0(sp$title, ": every model, all held-out points"),
         subtitle = "Vertical lines = Naive / HistAverage RMSE", x = NULL, y = NULL) +
    theme_minimal(base_size = 9) +
    theme(legend.position = "bottom")
  ggsave(sprintf("plots/%s_CV_overview_all_models.jpg", tag), p_over, width = 12, height = 9, dpi = 250)
  message("Saved: plots/", tag, "_CV_overview_all_models.jpg")

  invisible(all_df)
}

# ============================================================
# 5. LLO-2 CROSS-VALIDATION -- LLM models, hierarchical model, benchmarks
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
# 7. DAILY MODEL
# Placeholder: the daily model with LLM-informed priors is not built yet. The daily
# hierarchical model is in hierarchical_weakly_informative_priors/.
# ============================================================

# ============================================================
# 8. MAPS AND GRAPHS
# Models mapped: the hierarchical model and the pooled model of each LLM (Claude, Gemini,
# ChatGPT). The inputs come from the cached full-data fits (section 3; the hierarchical one is read
# from the sibling folder).
# (placeholder: LLM-generated burden estimates will be added to the burden comparison)
# ============================================================
maps_ready <- !is.null(hier_full) && !any(vapply(pooled_full, is.null, logical(1)))
if (!maps_ready) message("Section 8 skipped: needs the cached full-data fits of the hierarchical model and all three pooled models.")
if (maps_ready) {

# ---- Prepare map inputs from the cached fits ----
.prep_from_full <- function(f) {
  list(
    param_medians   = f$param_medians %>% select(District, p, gamma, gamma_count),
    burden_sum      = f$burden_summary,
    burden_expected = f$burden_expected
  )
}
map_inputs_tv     <- .prep_from_full(hier_full)
map_inputs_by_llm <- lapply(pooled_full, .prep_from_full)

# K_expected = E[burden] / E[SAM] - 1, computed for the hierarchical model and each pooled LLM model
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
for (llm in LLM_NAMES) {
  .make_k_expected_map(K_expected_summary_by_llm[[llm]], paste0(llm, "_pooled"))
}


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
  model_display_names <- c(hierarchical = "Hierarchical", claude = "Claude", gemini = "Gemini", chatgpt = "ChatGPT")
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

# Generate maps for the hierarchical model and the pooled model of each LLM
.make_model_maps(map_inputs_tv$param_medians, map_inputs_tv$burden_sum, "hierarchical",
                    name_map, somalia_map, duration_colors, gamma_common_cap, gamma_rate_common_cap)
for (llm in LLM_NAMES) {
  .make_model_maps(map_inputs_by_llm[[llm]]$param_medians, map_inputs_by_llm[[llm]]$burden_sum,
                      paste0(llm, "_pooled"), name_map, somalia_map, duration_colors, gamma_common_cap, gamma_rate_common_cap)
}




# ============================================================
# Burden comparison: hierarchical, the pooled model of each LLM (Claude/Gemini/ChatGPT), K=1.6
# (placeholder: the LLM-generated burden estimate will be added as another model here)
# ============================================================
bs_tv     <- hier_full$burden_summary
bs_by_llm <- lapply(pooled_full, function(f) f$burden_summary)

# Long-format: one row per (Model, District, Timepoint)
burden_compare_long <- bind_rows(
  bs_tv %>% mutate(Model = "hierarchical"),
  bind_rows(lapply(LLM_NAMES, function(llm) bs_by_llm[[llm]] %>% mutate(Model = paste0("llm_", llm, "_pooled")))),
  if (!is.null(FULL[[ALL_LLM_ID]])) FULL[[ALL_LLM_ID]]$burden_summary %>% mutate(Model = paste0("llm_", ALL_LLM_ID)),   # pool of the three LLM-pooled priors
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

for (col in intersect(c(paste0("llm_", LLM_NAMES, "_pooled"), paste0("llm_", ALL_LLM_ID)), names(burden_compare_wide))) {
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

# Total burden over all districts at each time point, per model, and its ratio to the K = 1.6 benchmark
burden_total_by_time <- burden_compare_long %>%
  group_by(Timepoint, Model) %>% summarise(burden_total = sum(burden_median, na.rm = TRUE), .groups = "drop") %>%
  group_by(Timepoint) %>% mutate(ratio_to_k16 = burden_total / burden_total[Model == "k1.6"]) %>% ungroup() %>%
  arrange(Timepoint, Model)
write_csv(burden_total_by_time, "outputs/burden_comparison_total_by_timepoint.csv")
message("Saved: outputs/burden_comparison_total_by_timepoint.csv")

# ============================================================
# End of burden comparison
# ============================================================
}   # end if (maps_ready)
