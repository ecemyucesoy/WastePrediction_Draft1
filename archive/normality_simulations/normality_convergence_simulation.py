# %% [markdown]
# # How quickly does prevalence converge to a normal distribution?
#
# Individual-level simulation of the disease process in Section 4:
#   - Arrivals follow a renewal process: inter-arrival times are iid draws
#     from a chosen distribution with mean 1 / lambda (lambda = expected
#     incident cases per month).
#   - Every case gets a geometric duration: at each month end it departs
#     with probability p. Cases arriving during a month join the system at
#     the end of that month (the paper's end-of-period convention).
#   - Prevalence is observed every DELTA = 6 months. We start from
#     P_{t-6} cases (set to the steady-state mean) and record P_t.
#
# Inter-arrival distributions, all with mean 1 / lambda:
#   - Exponential gaps   -> Poisson arrivals (CV of gaps = 1)
#   - Gamma(0.5) gaps    -> bursty arrivals  (CV of gaps = 1.41)
#   - Beta(2, 2) gaps    -> regular arrivals (CV of gaps = 0.45),
#                           scaled to the interval [0, 2 / lambda]
#
# We increase lambda and track how far the distribution of P_t is from
# normal, measured by skewness, excess kurtosis, the Kolmogorov-Smirnov
# distance, and coverage of central normal intervals. Each draw is
# standardised by the simulated mean and SD, so the metrics measure shape
# only: how normal P_t is once its mean and variance are known.

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

DATA_CSV = ROOT.parent.parent / "Datasets" / "Final_Clean_Dataset.csv"
OUT_DIR = ROOT / "outputs" / "normality_convergence"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DELTA = 6                                # months between observations
P_DEPART = 0.27                          # monthly departure prob. (median district estimate)
LAMBDA_GRID = np.geomspace(1, 2000, 13)  # expected incident cases per month
N_REPS = 20_000                          # simulated P_t per scenario
LEVELS = [0.80, 0.90, 0.95]
SEED = 20261001

# "Close enough to normal" thresholds used for the summary table.
# Monte Carlo s.e. with N_REPS = 20,000: skewness ~0.017, kurtosis ~0.035,
# 90% coverage ~0.002; the KS distance of an exactly normal sample is ~0.006.
SKEW_TOL, KURT_TOL, KS_TOL, COVER_TOL = 0.10, 0.20, 0.02, 0.01

# name -> (sampler of gaps with mean m, CV of gaps)
INTERARRIVALS = {
    "Poisson (exponential gaps)": (lambda rng, m, size: rng.exponential(m, size), 1.0),
    "Gamma(0.5) gaps, bursty": (lambda rng, m, size: rng.gamma(0.5, m / 0.5, size), np.sqrt(2)),
    "Beta(2,2) gaps, regular": (lambda rng, m, size: 2 * m * rng.beta(2, 2, size), np.sqrt(0.2)),
}
NAMES = list(INTERARRIVALS)

# Validated categorical palette (fixed order) plus markers as secondary encoding
COLORS = dict(zip(NAMES, ["#2a78d6", "#eb6834", "#1baf7a"]))
MARKERS = dict(zip(NAMES, ["o", "s", "^"]))
INK, INK_MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "font.size": 9, "axes.edgecolor": INK_MUTED, "axes.labelcolor": INK,
    "xtick.color": INK_MUTED, "ytick.color": INK_MUTED, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False,
    "axes.spines.right": False, "legend.frameon": False, "lines.linewidth": 2,
    "figure.dpi": 150, "savefig.bbox": "tight",
})

rng = np.random.default_rng(SEED)


# %% Simulation of one scenario
def steady_state(p, lam, delta=DELTA):
    """Stationary mean prevalence beta0 / (1 - beta1)."""
    pi_s = (1 - p) ** (delta - np.arange(1, delta + 1))
    return lam * pi_s.sum() / (1 - (1 - p) ** delta)


def simulate_prevalence(gap_sampler, gap_cv, lam, P_prev, p=P_DEPART,
                        n_reps=N_REPS, rng=rng):
    """Draw n_reps values of P_t given P_{t-6} = P_prev.

    Returns P_t and the arrival counts in each month of the window.
    """
    # The renewal process starts before the window so that arrivals inside
    # it are not affected by the start-up (an ordinary renewal process
    # started at a fixed time is not yet stationary).
    burn_in = max(2.0, 50.0 / lam)
    horizon = burn_in + DELTA
    n_expected = horizon * lam
    n_gaps = int(np.ceil(n_expected + 10 * gap_cv * np.sqrt(n_expected) + 30))

    chunk = max(1, int(4_000_000 // n_gaps))
    P_t = np.empty(n_reps, dtype=np.int64)
    monthly_counts = np.empty((n_reps, DELTA), dtype=np.int64)

    for start in range(0, n_reps, chunk):
        r = min(chunk, n_reps - start)
        times = np.cumsum(gap_sampler(rng, 1.0 / lam, (r, n_gaps)), axis=1) - burn_in
        if np.any(times[:, -1] <= DELTA):
            raise RuntimeError("not enough gaps drawn; increase the buffer")

        in_window = (times > 0) & (times <= DELTA)
        month = np.ceil(times).astype(np.int64)          # 1..6 inside the window
        # Geometric duration: number of month ends a case faces until it departs.
        duration = rng.geometric(p, (r, n_gaps))
        present_at_t = in_window & (duration > DELTA - month)

        for s in range(1, DELTA + 1):
            monthly_counts[start:start + r, s - 1] = (in_window & (month == s)).sum(axis=1)

        # Cases present at t-6 face 6 month ends; with geometric durations each
        # survives with probability (1-p)^6, whatever its time since onset.
        survivors = rng.binomial(int(P_prev), (1 - p) ** DELTA, r)
        P_t[start:start + r] = survivors + present_at_t.sum(axis=1)

    return P_t, monthly_counts


def shape_metrics(P_t):
    mean, sd = P_t.mean(), P_t.std()
    z = (P_t - mean) / sd
    out = {
        "mean_P": mean, "sd_P": sd,
        "skewness": stats.skew(P_t),
        "excess_kurtosis": stats.kurtosis(P_t),
        "ks_distance": stats.kstest(z, "norm").statistic,
    }
    for lev in LEVELS:
        half = stats.norm.ppf(0.5 + lev / 2)
        tag = int(lev * 100)
        out[f"miss_above_{tag}"] = np.mean(z > half)
        out[f"miss_below_{tag}"] = np.mean(z < -half)
        out[f"cover_{tag}"] = 1 - out[f"miss_above_{tag}"] - out[f"miss_below_{tag}"]
    return out


# %% Run the grid
rows, examples = [], {}
for name, (sampler, gap_cv) in INTERARRIVALS.items():
    for lam in LAMBDA_GRID:
        P_prev = max(1, round(steady_state(P_DEPART, lam)))
        P_t, counts = simulate_prevalence(sampler, gap_cv, lam, P_prev)
        row = {"arrivals": name, "lambda": lam, "P_prev": P_prev,
               # Dispersion of monthly counts; tends to CV(gaps)^2 for large lambda
               "monthly_count_var_to_mean": counts.var(axis=0).mean() / counts.mean(),
               **shape_metrics(P_t)}
        rows.append(row)
        examples[(name, lam)] = P_t
        print(f"{name:28s} lambda={lam:8.1f}  E[P]={row['mean_P']:9.1f}  "
              f"skew={row['skewness']:6.3f}  KS={row['ks_distance']:.4f}  "
              f"cover90={row['cover_90']:.3f}")

results = pd.DataFrame(rows)
results.to_csv(OUT_DIR / "convergence_results.csv", index=False)


# %% How fast: rate of decay and size needed
def size_needed(df, col, tol, target=0.0):
    """Smallest simulated E[P] from which all larger sizes are within tol."""
    ok = (df[col] - target).abs() < tol
    bad = np.where(~ok.to_numpy())[0]
    if len(bad) == 0:
        return df["mean_P"].iloc[0]
    if bad[-1] == len(df) - 1:
        return np.nan  # never reached on this grid
    return df["mean_P"].iloc[bad[-1] + 1]


summary = []
for name in NAMES:
    df = results[results.arrivals == name].sort_values("mean_P")
    # Skewness of a sum of many small independent pieces decays like 1/sqrt(size)
    slope = np.polyfit(np.log(df.mean_P), np.log(df.skewness.abs()), 1)[0]
    summary.append({
        "arrivals": name,
        "skewness_decay_slope (theory -0.5)": slope,
        "E[P] for |skew| < %.2f" % SKEW_TOL: size_needed(df, "skewness", SKEW_TOL),
        "E[P] for |kurtosis| < %.2f" % KURT_TOL: size_needed(df, "excess_kurtosis", KURT_TOL),
        "E[P] for KS < %.2f" % KS_TOL: size_needed(df, "ks_distance", KS_TOL),
        "E[P] for 90%% coverage within %.2f" % COVER_TOL: size_needed(df, "cover_90", COVER_TOL, 0.90),
    })
summary = pd.DataFrame(summary).set_index("arrivals")
summary.to_csv(OUT_DIR / "convergence_summary.csv")
print("\nExpected prevalence needed for P_t to be close to normal")
print(summary.round(3).to_string())


# %% Figures
observed = None
if DATA_CSV.exists():
    sam = pd.read_csv(DATA_CSV, encoding="utf-8-sig")["SAM"].dropna()
    observed = sam.quantile([0.05, 0.95]).to_numpy()


def decorate(ax):
    if observed is not None:
        ax.axvspan(*observed, color=GRID, alpha=0.6, lw=0, zorder=0)
    ax.set_xscale("log")
    ax.set_xlabel("Expected prevalence E[P_t]")


def plot_lines(ax, col, transform=lambda y: y):
    for name in NAMES:
        df = results[results.arrivals == name].sort_values("mean_P")
        ax.plot(df.mean_P, transform(df[col]), color=COLORS[name],
                marker=MARKERS[name], ms=4, label=name)


# Figure 1: distance from normality as prevalence grows
fig, axes = plt.subplots(1, 4, figsize=(15, 3.8), layout="constrained")

ax = axes[0]
plot_lines(ax, "skewness", np.abs)
x_ref = np.geomspace(results.mean_P.min(), results.mean_P.max(), 50)
ax.plot(x_ref, x_ref ** -0.5, color=INK_MUTED, lw=1, ls=":", label="1 / sqrt(E[P])")
ax.axhline(SKEW_TOL, color=INK_MUTED, lw=1, ls="--")
ax.set_yscale("log")
ax.set_ylabel("|Skewness|")
ax.legend(loc="lower left", fontsize=8)

ax = axes[1]
plot_lines(ax, "excess_kurtosis", np.abs)
ax.axhline(KURT_TOL, color=INK_MUTED, lw=1, ls="--")
ax.set_yscale("log")
ax.set_ylabel("|Excess kurtosis|")

ax = axes[2]
plot_lines(ax, "ks_distance")
ax.axhline(KS_TOL, color=INK_MUTED, lw=1, ls="--")
ax.set_yscale("log")
ax.set_ylabel("KS distance to N(0, 1)")

ax = axes[3]
plot_lines(ax, "cover_90")
ax.axhline(0.90, color=INK_MUTED, lw=1, ls="--")
ax.set_ylabel("Coverage of central 90% normal interval")

for ax in axes:
    decorate(ax)
fig.suptitle(f"Convergence of P_t | P_(t-6) to normality (geometric durations, p = {P_DEPART}). "
             "Dashed: tolerance or nominal value. Shaded: 5th-95th percentile of observed SAM.",
             color=INK, fontsize=10)
fig.savefig(OUT_DIR / "fig1_convergence.png")

# Figure 2: normal QQ plots at a small, a medium and a large arrival rate
show_lams = [LAMBDA_GRID[0], LAMBDA_GRID[len(LAMBDA_GRID) // 2], LAMBDA_GRID[-1]]
fig, axes = plt.subplots(len(NAMES), len(show_lams), figsize=(10, 9),
                         sharex=True, sharey=True, layout="constrained")
probs = (np.arange(1, 400) - 0.5) / 399
theo = stats.norm.ppf(probs)
for i, name in enumerate(NAMES):
    for j, lam in enumerate(show_lams):
        P_t = examples[(name, lam)]
        z = (P_t - P_t.mean()) / P_t.std()
        ax = axes[i, j]
        ax.plot([-4, 4], [-4, 4], color=INK_MUTED, lw=1, ls="--")
        ax.scatter(theo, np.quantile(z, probs), s=6, color=COLORS[name])
        ax.set_xlim(-4, 4)
        ax.set_ylim(-4, 5)
        if i == 0:
            ax.set_title(f"lambda = {lam:.0f} / month\nE[P] = {P_t.mean():.0f}", color=INK)
        if j == 0:
            ax.set_ylabel(f"{name}\nsample quantile (standardised)")
        if i == len(NAMES) - 1:
            ax.set_xlabel("Normal quantile")
fig.suptitle("Normal QQ plots of simulated P_t", color=INK, fontsize=10)
fig.savefig(OUT_DIR / "fig2_qq_plots.png")

print(f"\nSaved results and figures to {OUT_DIR}")
