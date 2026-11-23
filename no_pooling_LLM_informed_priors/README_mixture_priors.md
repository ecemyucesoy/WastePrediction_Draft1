# Mixture-prior version of the no-pooling LLM model

New files only; nothing existing was modified.

| File | What it is |
|---|---|
| `no_pooling_LLM_mixture_priors_monthly.stan` | Same model as `no_pooling_LLM_informed_priors_monthly.stan` (likelihood, recursion, P0 prior, bounds, generated quantities). Only the prior on (p, gamma, omega) differs: the K elicited runs of every district enter as a mixture, `target += log_sum_exp(...)`, using the full `beta_lpdf`. |
| `no_pooling_LLM_mixture_priors.R` | Builds the K x D arrays from `priors_llm/*.json`, runs the pre-checks, fits the models, writes outputs, compares with the pooled models, runs LLO-2 / LLO-3. Reuses the existing script's helpers, data cleaning, sampler settings, seeds, predictive samplers and scores. |

## Stan data flags

| Flag | Meaning |
|---|---|
| `K` | number of runs (10) |
| `a_p, b_p, a_gamma, b_gamma, a_omega, b_omega` | `K x D` matrices of Beta hyperparameters (floored at 1e-6, as before) |
| `joint = 1` | joint mixture: run k supplies its p, gamma and omega together (one `log_sum_exp` per district) |
| `joint = 0` | separate mixture per parameter (three independent `log_sum_exp`) |
| `estimate_w = 0` | weights fixed at `1/K` (primary) |
| `estimate_w = 1` | one `simplex[K] w` shared by all districts, Dirichlet(1,...,1) prior (mean `1/K`; fixed weights are the special case without learning) |

## Model ids

| Id | Model |
|---|---|
| `<llm>_mixture_joint_fixedw` | **primary**: joint mixture, w = 1/K |
| `<llm>_mixture_joint_estw` | joint mixture, estimated shared weights |
| `<llm>_mixture_sep_fixedw` | separate mixture per parameter, w = 1/K |
| `chatgpt_mixture_check_K1_run01` | validation: K = 1 (run 1 only); must reproduce `chatgpt_run01` |
| `chatgpt_mixture_check_identical_run01` | validation: 10 copies of run 1; must reproduce `chatgpt_run01` |

`<llm>` is `chatgpt`, `gemini` or `claude`. Cache files and tables use the same layout as the existing models
(`cache/<scheme>/<id>.rds`, `outputs/parameter_and_burden_summary_<id>.csv`, `outputs/diagnostics/rhat_and_ess_<id>.csv`,
`outputs/diagnostics/fit_stats_<id>.csv`). The cached object has the same fields as the existing fits; `prior` holds the
moment-matched Beta of the same runs (standard format), `prior_mixture` the mixture specification, and `diag` / `w_draws`
are extra fields.

## Running (environment variables; defaults in brackets)

| Variable | Meaning |
|---|---|
| `MIX_LLMS` | LLMs to fit, comma separated [`chatgpt,gemini,claude`]; `none` fits nothing |
| `MIX_VARIANTS` | [`joint_fixedw,joint_estw,sep_fixedw`] |
| `MIX_SCHEMES` | [`full,llo2`]; add `llo3` for LLO-3 |
| `MIX_VALIDATION` | `1` also fits the two validation models (chatgpt) [`1`] |
| `MIX_SMOKE` | `1` = 2 short chains and ids with the suffix `_smoke`: a wiring test, not a result [`0`] |

Sampler settings and seeds are those of the existing fits (4 chains, 5000 warm-up, 10000 iterations; seeds 123 / 456 / 789).
Everything already cached is reused, so the script can be run in batches (ChatGPT first, then Gemini and Claude).

## Outputs specific to the mixtures (`outputs/mixture_priors/`)

* `precheck_within_run_dependence*.csv` - correlation across districts of a run's p, gamma, omega prior means
* `precheck_mixture_vs_moment_match_overlap*.csv` - overlap coefficient between the 10-component mixture and its moment-matched Beta
* `precheck_between_run_disagreement.csv` - between-run spread per district and parameter; the 10 most disagreeing districts for p or gamma per LLM are flagged `high_disagreement`
* `diagnostics_by_district_<id>.csv`, `diagnostics_summary.csv` - Rhat, bulk/tail ESS, divergences, treedepth hits, flags (Rhat > 1.01)
* `mixture_weights_<id>.csv` - posterior mean and 90% interval of every weight (estimated-weight models)
* `compare_summary.csv`, `compare_top_districts.csv`, `compare_by_district_*.csv` - validation and comparison with the pooled models
* Cross-validation tables: `outputs/LLO2_mixture_CV_*.csv` (same style as `LLO2_CV_*`, mixture models next to the pooled models, the hierarchical model and the benchmarks), per-model files in `outputs/LLO2_CV_per_model/<id>.csv`
