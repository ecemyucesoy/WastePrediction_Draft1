# %% [markdown]
# # How fast does P_t | P_{t-6} become normal? A simulation study
#
# Proposition 1 approximates the conditional distribution of prevalence by
#     P_t | P_{t-6}  ~  N( E[P_t | P_{t-6}], Var(P_t | P_{t-6}) ),
# with mean (6) and variance (7). This script checks how good that
# approximation is in the parameter regime of the Somalia data.
#
# Data-generating process (exact under the model's assumptions):
#     P_t = Binomial(P_{t-6}, pi_old) + sum_{s=1}^{6} Binomial(N_s, pi_s),
#     pi_old = (1-p)^6,   pi_s = (1-p)^(6-s),
#     N_s iid with mean lambda and variance sigma^2.
# Because departures are geometric (memoryless), this binomial thinning is
# exactly what an individual-level simulation would produce, at a fraction
# of the cost.
#
# The arrival distribution is the only part the model leaves unspecified, so
# we vary it while matching the first two moments:
#     - negative binomial (overdispersed counts)
#     - gamma, discretised (moderate skew)
#     - lognormal, discretised (heavier right tail)
#     - shock mixture: Poisson months, 10% of which are surge months
#
# Metric: predictive interval coverage. For each scenario we build the
# central 50/80/90/95% intervals of the normal approximation and record the
# share of exact draws inside them, and the share missing above / below.
#
# Experiments:
#     1. District-calibrated: each district's posterior-median (p, lambda,
#        sigma) and each observed P_{t-6} in the data.
#     2. Population scaling: lambda and sigma grow together (sigma/lambda
#        fixed), P_{t-6} at the steady state. This is how districts of
#        different size differ under the omega = sigma / U parameterisation.
#     3. Theorem limit: lambda and sigma fixed, P_{t-6} -> infinity. This is
#        the limit stated in Proposition 1.

# %% Setup
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

try:
    ROOT = Path(__file__).resolve().parent
except NameError:  # running as a notebook
    ROOT = Path.cwd()

PARAM_CSV = ROOT / "outputs" / "parameter_and_burden_summary_theoretical_variance.csv"
DATA_CSV = ROOT.parent.parent / "Datasets" / "Final_Clean_Dataset.csv"
OUT_DIR = ROOT / "outputs" / "normality_simulation"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DELTA = 6
LEVELS = [0.50, 0.80, 0.90, 0.95]
N_SIM = 100_000          # draws per scenario; MC s.e. of 90% coverage ~ 0.001
SHOCK_PROB = 0.10        # share of surge months in the shock mixture
SEED = 20261001

DISTRIBUTIONS = ["Negative binomial", "Gamma", "Lognormal", "Shock mixture"]

# Validated categorical palette (fixed order) plus markers as secondary encoding
COLORS = dict(zip(DISTRIBUTIONS, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]))
MARKERS = dict(zip(DISTRIBUTIONS, ["o", "s", "^", "D"]))
INK, INK_MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "font.size": 9, "axes.edgecolor": INK_MUTED, "axes.labelcolor": INK,
    "xtick.color": INK_MUTED, "ytick.color": INK_MUTED, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False,
    "axes.spines.right": False, "legend.frameon": False, "lines.linewidth": 2,
    "figure.dpi": 150, "savefig.bbox": "tight",
})

rng = np.random.default_rng(SEED)


# %% Model quantities
def survival_probs(p, delta=DELTA):
    """pi_old and pi_s (s = 1..delta), equations (3)-(4) with constant p."""
    pi_old = (1 - p) ** delta
    pi_s = (1 - p) ** (delta - np.arange(1, delta + 1))
    return pi_old, pi_s


def normal_moments(P_prev, p, lam, sig2):
    """Conditional mean (6) and variance (7)."""
    pi_old, pi_s = survival_probs(p)
    mean = pi_old * P_prev + lam * pi_s.sum()
    var = (pi_old * (1 - pi_old) * P_prev
           + np.sum(lam * pi_s * (1 - pi_s) + sig2 * pi_s ** 2))
    return mean, var


def steady_state(p, lam):
    """Stationary mean P* = beta0 / (1 - beta1)."""
    pi_old, pi_s = survival_probs(p)
    return lam * pi_s.sum() / (1 - pi_old)


# %% Arrival distributions, each matched to mean lam and variance sig2
def _stochastic_round(x, rng):
    # Rounds up with probability equal to the fractional part, so the mean is
    # preserved; adds at most 0.25 to the variance, negligible at these scales.
    fl = np.floor(x)
    return (fl + (rng.random(x.shape) < (x - fl))).astype(np.int64)


def draw_arrivals(dist, lam, sig2, size, rng):
    if sig2 <= lam:  # under- or equi-dispersed: no overdispersed version exists
        return rng.poisson(lam, size)
    if dist == "Negative binomial":
        n = lam ** 2 / (sig2 - lam)
        return rng.negative_binomial(n, n / (n + lam), size)
    if dist == "Gamma":
        return _stochastic_round(rng.gamma(lam ** 2 / sig2, sig2 / lam, size), rng)
    if dist == "Lognormal":
        s2 = np.log1p(sig2 / lam ** 2)
        return _stochastic_round(rng.lognormal(np.log(lam) - s2 / 2, np.sqrt(s2), size), rng)
    if dist == "Shock mixture":
        # Month rate c * (1 + a * B), B ~ Bernoulli(q); c and a solve the
        # mean and variance constraints. Feasible while q * sqrt(v) < 1.
        q = SHOCK_PROB
        root_v = np.sqrt((sig2 - lam) / (lam ** 2 * q * (1 - q)))
        if q * root_v >= 1:
            raise ValueError("shock mixture cannot reach this variance")
        a = root_v / (1 - q * root_v)
        c = lam / (1 + q * a)
        surge = rng.random(size) < q
        return rng.poisson(c * (1 + a * surge))
    raise ValueError(dist)


# %% One scenario
def simulate_scenario(P_prev, p, lam, sig2, dist, n_sim=N_SIM, rng=rng):
    P_prev = int(round(P_prev))
    pi_old, pi_s = survival_probs(p)
    survivors = rng.binomial(P_prev, pi_old, n_sim)
    arrivals = draw_arrivals(dist, lam, sig2, (n_sim, DELTA), rng)
    P_t = survivors + rng.binomial(arrivals, pi_s).sum(axis=1)

    mean, var = normal_moments(P_prev, p, lam, sig2)
    sd = np.sqrt(var)
    out = {
        "dist": dist, "P_prev": P_prev, "p": p, "lam": lam, "sigma": np.sqrt(sig2),
        "cv_arrivals": np.sqrt(sig2) / lam, "expected_P": mean, "sd_P": sd,
        # Share of the conditional variance coming from new arrivals
        "arrival_var_share": 1 - pi_old * (1 - pi_old) * P_prev / var,
        # Sanity checks against (6)-(7); should be ~0 and ~1
        "mean_rel_err": P_t.mean() / mean - 1,
        "var_ratio": P_t.var() / var,
        "skewness": stats.skew(P_t),
        "excess_kurtosis": stats.kurtosis(P_t),
        "ks_distance": stats.kstest((P_t - mean) / sd, "norm").statistic,
    }
    for lev in LEVELS:
        half = stats.norm.ppf(0.5 + lev / 2) * sd
        above = np.mean(P_t > mean + half)
        below = np.mean(P_t < mean - half)
        tag = f"{int(lev * 100)}"
        out[f"cover_{tag}"] = 1 - above - below
        out[f"miss_above_{tag}"] = above
        out[f"miss_below_{tag}"] = below
    return out


# %% Load posterior medians and observed prevalence
params = pd.read_csv(PARAM_CSV)[["District", "p", "gamma_count", "omega_count"]]

# Same filtering as hierarchical_discrete_model.R
obs = pd.read_csv(DATA_CSV, encoding="utf-8-sig")
obs = obs[~(((obs.Year == 2016) & obs.Month.isin([1, 7])) |
            ((obs.Year == 2017) & (obs.Month == 1)))].copy()
rab = obs.District == "Rab Dhuure"
obs.loc[rab, "SAM"] = obs.loc[rab, "SAM"].fillna(obs.loc[rab, "SAM"].mean())
obs = obs.sort_values(["District", "Year", "Month"])

missing = set(params.District) - set(obs.District)
assert not missing, f"districts without prevalence data: {missing}"

cv = params.omega_count / params.gamma_count
print(f"{len(params)} districts")
print(f"monthly departure p:      {params.p.min():.3f} - {params.p.max():.3f}")
print(f"monthly incident cases:   {params.gamma_count.min():.0f} - {params.gamma_count.max():.0f}")
print(f"arrival CV (sigma/lambda): {cv.min():.2f} - {cv.max():.2f} (median {cv.median():.2f})")
print(f"observed SAM prevalence:  {obs.SAM.min():.0f} - {obs.SAM.max():.0f}")


# %% Experiment 1: district-calibrated scenarios
rows = []
for d in params.itertuples():
    P_prevs = obs.loc[obs.District == d.District, "SAM"].to_numpy()[:-1]
    for P_prev in P_prevs:
        for dist in DISTRIBUTIONS:
            r = simulate_scenario(P_prev, d.p, d.gamma_count, d.omega_count ** 2, dist)
            r["District"] = d.District
            rows.append(r)
exp1 = pd.DataFrame(rows)
exp1.to_csv(OUT_DIR / "exp1_district_calibrated.csv", index=False)

summary1 = exp1.groupby("dist").agg(
    scenarios=("cover_90", "size"),
    cover_50=("cover_50", "mean"), cover_80=("cover_80", "mean"),
    cover_90=("cover_90", "mean"), cover_95=("cover_95", "mean"),
    cover_90_min=("cover_90", "min"),
    share_cover_90_below_88=("cover_90", lambda x: np.mean(x < 0.88)),
    miss_above_90=("miss_above_90", "mean"), miss_below_90=("miss_below_90", "mean"),
    skewness=("skewness", "median"), arrival_var_share=("arrival_var_share", "median"),
    max_abs_mean_rel_err=("mean_rel_err", lambda x: np.abs(x).max()),
    var_ratio_range=("var_ratio", lambda x: f"{x.min():.3f}-{x.max():.3f}"),
).reindex(DISTRIBUTIONS)
summary1.to_csv(OUT_DIR / "exp1_summary_by_distribution.csv")
print("\nExperiment 1: district-calibrated coverage (averaged over district-periods)")
print(summary1.round(3).to_string())


# %% Experiment 2: population scaling at fixed arrival CV
p_med = params.p.median()
cv_grid = {"lowest district CV": cv.min(), "median district CV": cv.median(),
           "highest district CV": cv.max()}
lam_grid = np.geomspace(5, 20_000, 22)

rows = []
for cv_label, cv_val in cv_grid.items():
    for lam in lam_grid:
        P_star = steady_state(p_med, lam)
        for dist in DISTRIBUTIONS:
            r = simulate_scenario(P_star, p_med, lam, (cv_val * lam) ** 2, dist)
            r["cv_label"] = cv_label
            rows.append(r)
exp2 = pd.DataFrame(rows)
exp2.to_csv(OUT_DIR / "exp2_population_scaling.csv", index=False)


# %% Experiment 3: theorem limit (arrivals fixed, P_{t-6} grows)
lam_med = params.gamma_count.median()
sig_med = params.omega_count.median()
P_grid = np.geomspace(10, 1_000_000, 22)

rows = []
for P_prev in P_grid:
    for dist in DISTRIBUTIONS:
        rows.append(simulate_scenario(P_prev, p_med, lam_med, sig_med ** 2, dist))
exp3 = pd.DataFrame(rows)
exp3.to_csv(OUT_DIR / "exp3_theorem_limit.csv", index=False)


# %% Figures
obs_lo, obs_hi = obs.SAM.quantile([0.05, 0.95])


def shade_observed(ax):
    ax.axvspan(obs_lo, obs_hi, color=GRID, alpha=0.6, lw=0, zorder=0)


def nominal_line(ax, y):
    ax.axhline(y, color=INK_MUTED, lw=1, ls="--", zorder=1)


def plot_lines(ax, df, x, y):
    for dist in DISTRIBUTIONS:
        sub = df[df.dist == dist].sort_values(x)
        ax.plot(sub[x], sub[y], color=COLORS[dist], marker=MARKERS[dist],
                ms=4, label=dist)


# Figure 1: coverage and upper-tail misses as the population grows
fig, axes = plt.subplots(2, 3, figsize=(11, 6), sharex=True, sharey="row")
for j, (cv_label, cv_val) in enumerate(cv_grid.items()):
    sub = exp2[exp2.cv_label == cv_label]
    for i, (col, nominal) in enumerate([("cover_90", 0.90), ("miss_above_90", 0.05)]):
        ax = axes[i, j]
        shade_observed(ax)
        nominal_line(ax, nominal)
        plot_lines(ax, sub, "expected_P", col)
        ax.set_xscale("log")
    axes[0, j].set_title(f"{cv_label} ({cv_val:.2f})", color=INK)
    axes[1, j].set_xlabel("Expected prevalence E[P_t] (steady state)")
axes[0, 0].set_ylabel("Coverage of nominal 90% interval")
axes[1, 0].set_ylabel("Share above the 90% interval\n(nominal 5%)")
axes[0, 0].legend(loc="lower left")
fig.suptitle("Normal approximation as district size grows (arrival CV held fixed)\n"
             "Shaded: 5th-95th percentile of observed SAM prevalence",
             color=INK, fontsize=10)
fig.savefig(OUT_DIR / "fig1_population_scaling.png")

# Figure 2: district-calibrated coverage, one panel per arrival distribution
fig, axes = plt.subplots(1, 4, figsize=(12, 3.4), sharey=True)
for ax, dist in zip(axes, DISTRIBUTIONS):
    sub = exp1[exp1.dist == dist]
    nominal_line(ax, 0.90)
    ax.scatter(sub.expected_P, sub.cover_90, s=10, color=COLORS[dist],
               edgecolor="white", linewidth=0.4, alpha=0.8)
    ax.set_xscale("log")
    ax.set_title(dist, color=INK)
    ax.set_xlabel("Expected prevalence E[P_t]")
axes[0].set_ylabel("Coverage of nominal 90% interval")
fig.suptitle("District-calibrated scenarios: posterior-median parameters and "
             "observed P_{t-6} for each district-period", color=INK, fontsize=10)
fig.savefig(OUT_DIR / "fig2_district_calibrated.png")

# Figure 3: the limit in Proposition 1
fig, ax = plt.subplots(figsize=(5.5, 3.6))
shade_observed(ax)
nominal_line(ax, 0.90)
plot_lines(ax, exp3, "P_prev", "cover_90")
ax.set_xscale("log")
ax.set_xlabel("Previous prevalence P_{t-6} (arrivals held at district medians)")
ax.set_ylabel("Coverage of nominal 90% interval")
ax.set_title("Theorem limit: P_{t-6} grows with arrivals fixed", color=INK)
ax.legend(loc="lower right")
fig.savefig(OUT_DIR / "fig3_theorem_limit.png")

print(f"\nSaved tables and figures to {OUT_DIR}")
