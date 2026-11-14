# Archive

Models and material that are not part of the two main folders.

| Item | What it is |
|---|---|
| `archive_models_runner.R` | Runs the archived models through the Draft 1 pipeline (same data cleaning, LLO-2 / LLO-3 splits, Naive / HistAverage benchmarks, score and relative-score tables). |
| `stan_models/` | The nine archived Stan models in the current Stan array syntax, with descriptive names (the models themselves are unchanged). |
| `stan_models_original_syntax/` | The original files, untouched. |
| `risk_factor_results/` | CSV results of the earlier Dirichlet and horseshoe analyses (lagged risk factors). |
| `old_priors_json/` | Earlier LLM prior files (`priors_*_base_steady.json`, `priors_llm*.json`) and the tiered `priors_informative / moderate / weak` files used by the constant-noise informed model. |
| `old_code/` | The old monolithic script before the restructure, the old daily-model notebook and Stan file, and the summary written for the paper. |
| `old_outputs/`, `old_plots/` | Outputs and figures of earlier drafts and of the second-year paper (not tracked by git). |
| `normality_simulations/` | The two Python normality simulations and their outputs. |
| `workspace_backups/` | R workspace backup (not tracked by git). |

## Archived models in `archive_models_runner.R`

All are AR(1) models with a constant observation noise alpha:

| Model id | Stan file | Pooling / covariates | Needs |
|---|---|---|---|
| `hierarchical_AR1_constant_noise` | `hierarchical_AR1_constant_noise.stan` | partial pooling of p, gamma, alpha | SAM, population |
| `no_pooling_AR1_constant_noise` | `no_pooling_AR1_constant_noise.stan` | independent districts | SAM |
| `full_pooling_AR1_constant_noise` | `full_pooling_AR1_constant_noise.stan` | one p, gamma, alpha for all | SAM |
| `no_pooling_LLM_informed_priors_AR1_constant_noise` | `no_pooling_LLM_informed_priors_AR1_constant_noise.stan` | district priors from the tier JSON (`PRIOR_TIER`) | `old_priors_json/old variance model json files/` |
| `hierarchical_district_risk_factors_AR1_constant_noise` | `hierarchical_district_risk_factors_AR1_constant_noise.stan` | district-mean risk factors shift p and gamma | FSNAU risk factors |
| `hierarchical_time_varying_single_covariate` | `hierarchical_time_varying_single_covariate.stan` | p_t / gamma_t driven by one lagged covariate (`SINGLE_X`) | FSNAU risk factors |
| `hierarchical_time_varying_lag_horseshoe` | `hierarchical_time_varying_lag_horseshoe.stan` | horseshoe on 6 lag coefficients (`HS_CFG`) | FSNAU risk factors |
| `hierarchical_time_varying_lag_dirichlet_shared_weights` | `..._lag_dirichlet_shared_weights.stan` | Dirichlet lag weights shared by p and gamma | FSNAU risk factors |
| `hierarchical_time_varying_lag_dirichlet_channel_weights` | `..._lag_dirichlet_channel_weights.stan` | separate Dirichlet lag weights for p and gamma | FSNAU risk factors |

Choices that the original analyses left to the user are collected at the top of the runner (`HP`, `SINGLE_X`,
`HS_CFG`, `PRIOR_TIER`). Lagged covariates are standardised with the training rows only in LLO-2 / LLO-3.
These models predate the theoretical-variance models; their results are not comparable with
the earlier outputs in `old_outputs/` unless the same settings are used.
