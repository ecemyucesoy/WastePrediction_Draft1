# SAM Hierarchical Model — Results Summary for Paper Draft

Somalia district-level SAM (severe acute malnutrition) modeling: theoretical-variance SIS model with flat (`tv_base`) vs. LLM-informed priors (`tv_claude`, `tv_gemini`, `tv_chatgpt`), 74 districts, 9 biannual observations/district (July 2017 – July 2021).

Source files referenced below are all in `outputs/` and `plots/` of the project directory unless noted otherwise.

---

## 1. Posterior parameter summaries

### 1a. District-level medians, per model (74 districts)

| Model | median p | duration (months) | median γ_count (incidence) | median ω_count (arrival SD) |
|---|---|---|---|---|
| tv_base | 0.275 | 3.64 | 331 | 384 |
| tv_claude | 0.153 | 6.54 | 186 | 208 |
| tv_gemini | 0.180 | 5.54 | 206 | 194 |
| tv_chatgpt | 0.136 | 7.37 | 153 | 227 |

*(γ_count and ω_count are gamma/omega expressed in count scale, i.e. γ×population_U5 and ω×population_U5, matching the "gamma_count_out"/"omega_count_out" generated quantities in the Stan model.)*

No implausible values in any model: zero districts with p outside (0,1), zero with duration >24 or <1 months, zero negative counts.

### 1b. p and duration with 95% credible intervals (source: `outputs/parameter_CI_tv_vs_tv_informed.csv`)

Aggregated across districts (median of each district's own posterior median/95% CI):

| Model | p median [95% CI] | duration median [95% CI], months |
|---|---|---|
| tv_base | 0.275 [narrowest district 2.60–5.26 range] | 3.64 [range of district medians: **3.15–4.55**] |
| tv_claude | 0.153 | 6.54 [4.06–15.13] |
| tv_gemini | 0.180 | 5.54 [3.56–8.63] |
| tv_chatgpt | 0.136 | 7.37 [4.50–13.90] |

> **On "3.4 to 4.2 months":** I could not find an exact match to this figure in the saved outputs. The closest quantity is **tv_base's range of district-level median durations: 3.15–4.55 months** (narrowest: Buuhoodle 3.15, Baki 3.15, Borama 3.18; widest: Ceel Barde 4.55, Rab Dhuure 4.40, Xudur 4.37). If "3.4 to 4.2" is meant to describe this same range, it's a close paraphrase, not an exact figure from this run — please double check against whichever run/version produced that number, or let me know and I'll recompute the exact match.

### 1c. γ and ω/α — gap to flag

`gamma`'s CI is in the same file as (1b) (not tabulated above for space; available per-district in the CSV). **`omega`'s 95% CI is not currently saved anywhere** — only the posterior median (via `omega_count` in the per-model summary CSVs) was persisted; the code that builds `parameter_CI_tv_vs_tv_informed.csv` only pulls `p` and `gamma` draws, not `omega`. Same for `alpha_equiv` (the code's stand-in for a base-model-comparable dispersion parameter, currently only a per-district median, no CI). **To get formal omega/alpha credible intervals, a small code addition (pulling `post$omega` the same way `p`/`gamma` are pulled) plus a re-run of the main fits is needed** — the raw posterior draws aren't persisted to disk, so this can't be recovered without refitting.

---

## 2. Regional sanity check (Somaliland vs. Bakool)

Rationale (as framed for the paper): there is no ground truth for incidence — we never observe it directly — so model validity is assessed via face validity: does a model fed very little data (9 observations/district) recover parameter patterns consistent with known context? Somaliland (Hargeysa, Berbera, Borama, Burco, Gebiley, etc.) is comparatively stable, has its own administration, and better health infrastructure access; Bakool (Xudur, Waajid, Ceel Barde, Rab Dhuure, Tayeeglow) is drought-prone, conflict-affected, and has poor healthcare access. Expectation: **shorter duration / lower incidence in Somaliland, longer duration / higher incidence in Bakool.**

| Model | Somaliland median duration | Bakool median duration | Somaliland median γ_count | Bakool median γ_count |
|---|---|---|---|---|
| tv_base | 3.30 mo | 4.37 mo | 267.8 | 282.2 |
| tv_claude | 5.15 mo | 8.48 mo | 142.3 | 160.0 |
| tv_gemini | 4.65 mo | 6.49 mo | 130.3 | 175.0 |
| tv_chatgpt | 6.22 mo | 10.79 mo | 118.8 | 116.4 |

**Duration**: confirms the hypothesis in **all four models**, direction and magnitude consistent — Somaliland always shorter, gap ranging from ~1.1 months (base) to ~4.6 months (chatgpt).

**Incidence (γ)**: confirms the hypothesis in **3 of 4 models** (base, claude, gemini all show Bakool higher); **chatgpt is a near-tie** (118.8 vs. 116.4, Somaliland marginally higher) — worth reporting honestly as a partial rather than universal confirmation, rather than claiming a uniformly clean pattern across all four.

This is a genuinely useful, model-agnostic result: it holds up under the flat prior too, so it's not an artifact of the LLM elicitation itself — the *data* (SAM trajectories), combined with the model structure, recovers a context-consistent pattern even before any informed prior is added, and the LLM-informed models sharpen (in the case of claude/chatgpt, widen) that gap further on duration.

---

## 3. Convergence diagnostics (Rhat / effective sample size ranges)

Source: `outputs/rhat_ess_ranges_ALL_FITS.csv` (generated fresh for this summary).

| Fit | n params | Rhat range | n_eff range |
|---|---|---|---|
| tv_base (main) | 4,225 | 1.000 – 1.009 | 323 – 33,875 |
| tv_claude (main) | 5,551 | 1.000 – 1.003 | 1,784 – 35,423 |
| tv_gemini (main) | 5,551 | 1.000 – 1.002 | 1,974 – 37,902 |
| tv_chatgpt (main) | 5,551 | 1.000 – 1.003 | 1,362 – 40,211 |
| tv_base (LL2O-CV) | 3,485 | 1.000 – 1.016 | 357 – 32,707 |
| tv_claude (LL2O-CV) | 4,515 | 1.000 – 1.004 | 662 – 33,830 |
| tv_gemini (LL2O-CV) | 4,515 | 1.000 – 1.003 | 1,732 – 33,571 |
| tv_chatgpt (LL2O-CV) | 4,515 | 1.000 – 1.010 | 380 – 40,583 |
| tv_base (LL3O-CV) | 3,115 | 1.000 – 1.014 | 231 – 30,156 |
| tv_claude (LL3O-CV) | 3,997 | 1.000 – 1.005 | 999 – 30,823 |
| tv_gemini (LL3O-CV) | 3,997 | 1.000 – 1.010 | 337 – 35,744 |
| tv_chatgpt (LL3O-CV) | 3,997 | 1.000 – 1.007 | 615 – 37,270 |

All 12 fits: max Rhat ≤ 1.016 (well under the 1.05 convergence threshold), minimum n_eff ≥ 231 (a couple of parameters dip just under the informal 400 rule-of-thumb, but nowhere near a level that would call convergence into question). Every result in this document rests on fits that converged cleanly.

---

## 4. In-sample fit (approximate — see caveat)

Source: `outputs/insample_fit_approximate.csv`.

| Model | n obs | RMSE (in-sample) | MAE (in-sample) |
|---|---|---|---|
| tv_base | 592 | 1,560.6 | 656.4 |
| tv_gemini | 592 | 1,564.6 | 616.8 |
| tv_claude | 592 | 1,599.2 | 632.6 |
| tv_chatgpt | 592 | 1,645.1 | 646.7 |

**Caveat**: this is a *point-estimate plug-in* — one-step-ahead prediction using each district's posterior-median `beta0`/`beta1`, applied to the true previous observed value, across all 9 timepoints (592 = 74 districts × 8 transitions, since the first observation has no predecessor). It is **not** a full posterior-predictive in-sample RMSE (that would require per-draw predictions from the raw posterior, which aren't currently persisted). Note the key qualitative point: in-sample, all four models fit comparably (1,561–1,645) — the large differences between models only show up **out-of-sample** (Section 5), which is exactly consistent with the paper's core claim that informed priors matter when data is scarce, not when there's enough of it for the likelihood to dominate.

---

## 5. Out-of-sample cross-validation results

Two designs, both "leave-last-N-out": **LL2O** trains on the first 7 observations/district, holds out the 8th and 9th; **LL3O** (robustness check) trains on the first 6, holds out the 7th, 8th, and 9th. Both use the same convention: each held-out point is predicted one step ahead from the *true* observed value at the preceding timepoint (not recursively from the model's own prior prediction).

### 5a. Full CV summary table

Source: `outputs/LLO2_CV_summary_by_model.csv`, `outputs/LLO3_CV_summary_by_model.csv`.

| CV | Point | Model | RMSE | MAPE (%) | CRPS |
|---|---|---|---|---|---|
| LL3O | 7th | tv_chatgpt | 456.6 | 36.0 | 323.6 |
| LL3O | 7th | tv_claude | 463.9 | 36.6 | 332.3 |
| LL3O | 7th | Naive | 467.2 | 37.8 | — |
| LL3O | 7th | tv_gemini | 488.6 | 34.9 | 355.5 |
| LL3O | 7th | tv_base | **739.1** | 65.6 | 503.7 |
| LL2O | 8th | tv_claude | 255.5 | 22.2 | 193.8 |
| LL2O | 8th | tv_chatgpt | 257.3 | 21.2 | 215.6 |
| LL2O | 8th | tv_gemini | 272.3 | 21.3 | 191.9 |
| LL2O | 8th | Naive | 384.2 | 26.5 | — |
| LL2O | 8th | tv_base | 405.0 | 38.4 | 305.4 |
| LL3O | 8th | tv_chatgpt | 280.1 | 22.6 | 221.0 |
| LL3O | 8th | tv_claude | 292.4 | 24.1 | 209.7 |
| LL3O | 8th | tv_gemini | 312.2 | 22.8 | 220.9 |
| LL3O | 8th | Naive | 384.2 | 26.5 | — |
| LL3O | 8th | tv_base | **528.3** | 48.5 | 370.6 |
| LL2O | 9th | Naive | 1,288.6 | 58.5 | — |
| LL2O | 9th | tv_base | 1,297.1 | 60.1 | 980.4 |
| LL2O | 9th | tv_claude | 1,340.8 | 55.3 | 1,115.5 |
| LL2O | 9th | tv_chatgpt | 1,354.8 | 55.2 | 1,074.4 |
| LL2O | 9th | tv_gemini | 1,370.3 | 53.6 | 1,165.8 |
| LL3O | 9th | Naive | 1,288.6 | 58.5 | — |
| LL3O | 9th | tv_base | 1,330.5 | 63.5 | 987.8 |
| LL3O | 9th | tv_claude | 1,355.8 | 54.9 | 1,129.1 |
| LL3O | 9th | tv_chatgpt | 1,364.0 | 54.8 | 1,075.9 |
| LL3O | 9th | tv_gemini | 1,392.7 | 53.1 | 1,190.4 |

### 5b. Formal significance tests (paired Wilcoxon signed-rank, by district, Holm-corrected across the 3 base-vs-LLM comparisons per point)

Source: `outputs/cv_significance_tests_base_vs_llm.csv`.

| CV | Point | Comparison | p (Holm-adjusted) | Significant at 0.05? |
|---|---|---|---|---|
| LL3O | 7th | base vs. claude | 1.06×10⁻⁶ | **Yes** |
| LL3O | 7th | base vs. gemini | 8.27×10⁻⁷ | **Yes** |
| LL3O | 7th | base vs. chatgpt | 1.54×10⁻⁶ | **Yes** |
| LL2O | 8th | base vs. claude | 4.39×10⁻⁴ | **Yes** |
| LL2O | 8th | base vs. gemini | 1.01×10⁻⁴ | **Yes** |
| LL2O | 8th | base vs. chatgpt | 4.39×10⁻⁴ | **Yes** |
| LL3O | 8th | base vs. claude | 4.01×10⁻⁷ | **Yes** |
| LL3O | 8th | base vs. gemini | 2.08×10⁻⁷ | **Yes** |
| LL3O | 8th | base vs. chatgpt | 6.25×10⁻⁷ | **Yes** |
| LL2O | 9th | base vs. claude | 0.307 | No |
| LL2O | 9th | base vs. gemini | 0.254 | No |
| LL2O | 9th | base vs. chatgpt | 0.307 | No |
| LL3O | 9th | base vs. claude | 0.612 | No |
| LL3O | 9th | base vs. gemini | 0.336 | No |
| LL3O | 9th | base vs. chatgpt | 0.612 | No |

**Headline finding**: LLM-informed priors deliver a statistically significant improvement over flat priors at *every* one-step-ahead test point — replicated across two independent test months (July 2020, January 2021) and two independent CV designs (6-month and 7-month training windows). At the second held-out point (July 2021), no model is statistically distinguishable from any other — the honest claim there is "no model could be distinguished," not "informed priors underperform two steps ahead."

### 5c. Why the 9th point behaves differently: a dated, shared shock, not noise

All 74 districts' 9th observation falls on the **exact same calendar date: July 2021** (panel is perfectly aligned: 1st obs = July 2017, 8th = January 2021, 9th = July 2021, for every district). This lines up with the documented onset of the 2021 Horn of Africa drought (failed Gu rains). Diagnostics (`outputs/timepoint_outlier_summary.csv`, `outputs/district_timepoint_outlier_diagnostics.csv`):
- Median month-to-month SAM jump: 22% (month 7→8) vs. **60%** (month 8→9); mean jump 29% vs. **100%** (literal doubling on average).
- 17 of 74 districts (23%) exceed a modified z-score of 3.5 (Iglewicz & Hoya, 1993 convention) at month 9; **zero** districts do at month 8.
- Extreme examples: Jamaame 675→7,740 (11.5×); Afgooye +163%; Badhaadhe +224%.

No model — informed or not — had any basis to anticipate this. `plots/llo2_cv_district_panel_observed_vs_predicted.jpg` and `plots/llo3_cv_district_panel_observed_vs_predicted.jpg` show this visually across all 74 districts: every model's prediction clusters well below the actual month-9 spike in the affected districts.

---

## 6. Burden estimates vs. current K=1.6 practice

Source: `outputs/burden_comparison_global_summary.csv`, `outputs/burden_vs_k16_by_model_CURRENT.csv`.

### 6a. Grand totals (sum across all districts and timepoints, in children)

| Model | Grand mean burden | Grand total burden (children) |
|---|---|---|
| tv_base (hierarchical) | 9,201 | 6,127,619 |
| tv_gemini | 6,531 | 4,349,359 |
| tv_claude | 5,995 | 3,992,435 |
| tv_chatgpt | 5,368 | 3,575,374 |
| **K=1.6 benchmark** | **3,602** | **2,398,774** |

### 6b. Ratio to K=1.6 and ranking agreement

| Model | Mean ratio to K=1.6 | % districts where model > K=1.6 | Pearson r | Spearman ρ |
|---|---|---|---|---|
| tv_base | 2.61× | 100% | 0.995 | 0.982 |
| tv_gemini | 1.77× | 100% | 0.997 | 0.984 |
| tv_claude | 1.66× | 98.6% | 0.998 | 0.980 |
| tv_chatgpt | 1.50× | 100% | 0.995 | 0.985 |

**Interpretation**: every model implies substantially higher total burden than the current K=1.6 heuristic — even the most conservative (ChatGPT) implies 50% more. But the ranking of districts by burden is nearly identical between every model and K=1.6 (r > 0.99, ρ ≈ 0.98 throughout). **The current heuristic appears to get the relative ranking right but the absolute level wrong, systematically underestimating total burden** — a decision-relevant finding if K=1.6 is used to size resource allocation.

---

## 7. Daily-rate model (`sam_theoretical_variance_daily.stan`)

**Not yet available.** `outputs/parameter_and_burden_summary_tv_daily.csv` does not exist yet — the most recent full run's file timestamps stop at the "chain and model convergence stats" section (shrinkage plots, 15:30), which comes immediately *before* the daily-model section in the script. Either it's still running or stopped there; I'll pull daily-model numbers once that file appears.

---

## 8. Maps

Generated and current as of this run (`plots/`): `duration_map_tv_base.jpg`, `duration_map_tv_claude.jpg`, `duration_map_tv_gemini.jpg`, `duration_map_tv_chatgpt.jpg`, `incidence_map_tv_base.jpg` (+ claude/gemini/chatgpt), `burden_map_tv_base.jpg` (+ claude/gemini/chatgpt), `burden_map_benchmark.jpg` (K=1.6), `prevalence_map_observed.jpg`, `K_expected_map.jpg`.

**Framing for the paper** (as discussed): there is no way to directly evaluate the quality of the incidence/duration estimates, because true incidence is never observed — SAM prevalence is observed, incidence and duration are latent. The validity argument is therefore a face-validity/sanity-check argument, not a ground-truth accuracy argument: *does a model fed very little data (9 biannual observations per district) recover a duration/incidence pattern consistent with known context?* Somaliland — relatively stable, functionally autonomous administration, better healthcare access — should show shorter duration and lower incidence; Bakool — drought-prone, conflict-affected, poor healthcare access — should show the opposite. Section 2 above confirms this directionally across all four models for duration, and in 3 of 4 models for incidence (ChatGPT is a near-tie). The maps make this visually inspectable district-by-district, including for districts outside this specific two-region comparison.

---

## 9. Recommendations for a Management Science submission

Beyond what's assembled above, items worth adding before submission:

**Statistical rigor still open:**
- Pairwise **LLM-vs-LLM** significance tests (is Gemini significantly better than Claude/ChatGPT, or just numerically ahead?) — not yet run, only base-vs-each-LLM.
- **Prior-to-posterior contraction ratios** (posterior SD / prior SD per parameter) — the direct evidence for the "T≈9 isn't enough to overwhelm the prior" claim used to interpret Section 1; currently an assertion, needs a table.
- Formal **inter-LLM reliability** statistic (e.g., ICC) rather than pairwise correlations, if framing LLMs as "expert raters."

**Framing Management Science will push on:**
- A **decision-theoretic translation** of forecast-accuracy gains — what does a 35% RMSE reduction mean for a resource-allocation decision? Is there an asymmetric cost to under- vs. over-estimating burden?
- **Cost/scalability of LLM elicitation vs. traditional expert panels** — a natural, distinctive contribution for this journal, currently only implicit.

**Methodological validity findings needing their own subsection:**
- The **quantile self-consistency finding**: Claude and ChatGPT's stated `p` quantiles are internally inconsistent in ~100% of districts (the fitted Beta's implied median differs from the LLM's stated median beyond a hybrid abs/rel threshold); Gemini's are consistent in 100% of districts. This is a reportable finding about LLM elicitation reliability, independent of the forecasting result. (Source: `outputs/quantile_fit_diagnostics_<llm>.csv`.)
- The **archetype-reuse finding**: LLMs (most extremely ChatGPT) assign near-identical numeric triples to groups of contextually-similar districts rather than fully bespoke values — bears on how many *effectively independent* judgments were actually elicited, versus the raw district count.
- **LLM version/reproducibility caveat**: state which model snapshots were used; LLM outputs aren't a fixed instrument like a survey, and this is worth flagging as an inherent limitation.

**Robustness still worth adding, resources permitting:**
- **Prompt-wording sensitivity** — only one wording variant per LLM has been run; even one additional wording per LLM would pre-empt the likely first reviewer question about whether results are an artifact of one specific prompt.

---

## 10. Outstanding items before this document is complete
- [ ] Daily-model numbers (Section 7) — pending the run reaching that section.
- [ ] Confirm the exact provenance of "3.4 to 4.2 months" (Section 1b).
- [ ] Add omega/alpha 95% credible intervals (Section 1c) — requires a small code change + re-run.
- [ ] Full posterior-predictive in-sample RMSE (Section 4) if the point-estimate plug-in isn't precise enough for the paper.
