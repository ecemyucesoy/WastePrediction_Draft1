# Draft 1 -- SAM burden models for Somalia

Three folders, each self-contained (R script, Stan files, data it needs, cache, outputs, plots).

| Folder | What it is |
|---|---|
| `no_pooling_LLM_informed_priors/` | No pooling across districts; every district has its own Beta prior for p, gamma, omega elicited from an LLM (3 LLMs x 10 runs, plus pooled and tempered versions). Also holds the comparisons with the hierarchical model, the K = 1.6 benchmark and the slide figures. |
| `hierarchical_weakly_informative_priors/` | Partial pooling across districts with weakly informative hyperpriors (monthly and daily model). No LLM information. |
| `archive/` | Earlier models (constant-noise AR(1), risk-factor, Dirichlet, horseshoe), old outputs, old prior files, the old monolithic script, and `archive_models_runner.R`, which runs the archived models through the same pipeline. |

## Names

| Old | New |
|---|---|
| `sam_theoretical_variance.stan` | `hierarchical_weakly_informative_priors/hierarchical_weakly_informative_priors_monthly.stan` |
| `sam_theoretical_variance_daily.stan` | `hierarchical_weakly_informative_priors/hierarchical_weakly_informative_priors_daily.stan` |
| `sam_theoretical_variance_informed_priors.stan` | `no_pooling_LLM_informed_priors/no_pooling_LLM_informed_priors_monthly.stan` |
| `hierarchical_discrete_model.R` (one script for everything) | `no_pooling_LLM_informed_priors.R` and `hierarchical_weakly_informative_priors.R` |
| model id / label `tv_base` | `hierarchical` |
| `tv_<llm>_run<NN>`, `tv_<llm>_pooled` | `<llm>_run<NN>`, `<llm>_pooled` (tempered: `<llm>_pooled_t050`, `_t025`, `_t010`) |
| benchmark `MeanBenchmark` (column suffix `vsHistMean`) | `HistAverage` (`vsHistAverage`); `Naive` unchanged |
| `outputs/parameter_and_burden_summary_theoretical_variance.csv` | `hierarchical_.../outputs/parameter_and_burden_summary_hierarchical.csv` |
| `outputs/parameter_and_burden_summary_tv_<id>.csv` | `outputs/parameter_and_burden_summary_<id>.csv` |
| `outputs/parameter_CI_tv_vs_tv_informed.csv` | `no_pooling_.../outputs/parameter_CI_hierarchical_vs_LLM_informed.csv` |
| `rhat_and_ess_llto_cv_*`, `rhat_and_ess_llto3_cv_*` | `rhat_and_ess_llo2_cv_*`, `rhat_and_ess_llo3_cv_*` |

The numbers did not change: the new scripts were run from the copied caches and reproduce the old tables
(see "Checks" below).

## How the folders fit together

* `no_pooling_LLM_informed_priors.R` **reads** the cached fits of the hierarchical model from
  `../hierarchical_weakly_informative_priors/cache/` (it never refits or writes there). If that folder is missing,
  the comparisons with the hierarchical model are skipped and everything else still runs.
* Both folders use the same data, the same leave-last-2/3-out splits, the same benchmarks and the same scores,
  so their tables are directly comparable.
* The maps in `hierarchical_.../plots/` use bins computed from the hierarchical model alone; the maps in
  `no_pooling_.../plots/` use one common scale across the hierarchical and the pooled LLM models.

## Running

Each script starts with a header explaining its settings. In short: fits are cached under `cache/<scheme>/` and
skipped when present, so running a script from the top only refits what is missing. `RUN_MODELS` and `RUN_SCHEMES`
limit what is fitted in a session, `FORCE_REFIT <- TRUE` refits everything. Paths are set at the top of each script
(`PROJECT_DIR`, `DATA_PATH`, `SHAPEFILE_PATH`).

## Git

This folder is its own git repository. `.gitignore` tracks code, Stan files, JSON, markdown and the CSV tables
and ignores caches (`cache/`, `*.rds`), figures and the large archive folders (`archive/old_outputs/`,
`archive/old_plots/`, `archive/workspace_backups/`, `archive/risk_factor_results/`).

## Not copied

`Hierarchical model/archived/outputs/posteriors/` (1.9 GB of old posterior draws) stays where it is.
The old project folder `Hierarchical model/` is untouched.

## Checks

* **Numbers unchanged.** `hierarchical_weakly_informative_priors.R` and `no_pooling_LLM_informed_priors.R` were run
  from the copied caches (no refitting) and every output table was compared with the old files (CV tables for all
  models, relative scores, per-model files, parameter summaries and CIs, burden comparison, chain stats, sensitivity
  summary, outlier tables): largest difference 0.
* **Daily model.** The cache did not hold a daily fit, so the new script fitted it (same seed and settings) and
  reproduced the old `parameter_and_burden_summary_tv_daily.csv` exactly.
* **Archive runner.** For all nine archived models and all three schemes, the Stan data supplied by the runner matches
  the data block of the Stan file, and the predictive functions return finite draws of the right shape. A wiring test
  with short chains (2 chains, 300 warm-up; full data and LLO-2) ran end to end for all nine models. Those fits are
  **not results**: with so little warm-up several models had not converged (e.g. the single-covariate model, whose Rhat
  was far above 1), so run the models with the full MCMC settings before comparing them.
