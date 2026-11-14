data {
  int<lower=1> N;                       // Number of observations
  int<lower=1> D;                       // Number of districts
  int<lower=1, upper=D> district_id[N]; // District ID per observation
  vector[N] y;                          // Observed SAM values
  vector<lower=0>[D] pop_u5_ref;        // One fixed U5 population per district
  int<lower=1, upper=N> start_index[D]; // Index of the first obs per district

  vector[2] p_hyperprior;
  vector[2] gamma_hyperprior;
  vector[2] alpha_hyperprior;

  // ---- Single standardized covariate aligned with y[n] ----
  vector[N] x_z;

  // ---- Toggle whether x_z affects p and/or gamma ----
  int<lower=0, upper=1> use_x_on_p;
  int<lower=0, upper=1> use_x_on_gamma;

  // ---- Prior scales for the single covariate effect ----
  // (keep in data so you can reuse this file across covariates)
  real<lower=0> beta_p_sd;
  real<lower=0> beta_gamma_sd;
}

parameters {
  // Baseline (district-level) parameters in (0,1) / positive
  real<lower=1e-10, upper=1-1e-10> p[D];
  real<lower=1e-10, upper=1-1e-10> gamma[D];
  real<lower=1e-10> alpha[D];
  real<lower=1e-10> P0[D];

  // Global means for p, gamma (Beta-Beta pooling)
  real<lower=1e-10, upper=1-1e-10> p_global;
  real<lower=1e-10, upper=1-1e-10> gamma_global;

  real<lower=1e-10> kappa_p;
  real<lower=1e-10> kappa_gamma;

  // Observation noise pooling
  real<lower=1e-10> alpha_global;
  real<lower=1e-10> sigma_alpha;

  // ---- Single-covariate effects ----
  real beta_gamma_x;   // effect of x_z on logit(gamma_t)
  real beta_p_x;       // effect of x_z on logit(p_t)
}

transformed parameters {
  // Baseline logits per district
  vector[D] logit_p_base;
  vector[D] logit_gamma_base;

  // Time-varying parameters and AR(1) coefficients per observation
  vector[N] p_t;       // p_{t,d}
  vector[N] gamma_t;   // gamma_{t,d}
  vector[N] beta1_n;   // beta1 at time n
  vector[N] beta0_n;   // beta0 at time n

  // Compute baseline logits
  for (d in 1:D) {
    logit_p_base[d]     = logit(p[d]);
    logit_gamma_base[d] = logit(gamma[d]);
  }

  // Compute time-varying p_t, gamma_t and corresponding beta0_n, beta1_n
  for (n in 1:N) {
    int d = district_id[n];

    // Time-varying p_t
    {
      real lp = logit_p_base[d] + (use_x_on_p * beta_p_x * x_z[n]);
      p_t[n] = inv_logit(lp);
    }

    // Time-varying gamma_t
    {
      real lg = logit_gamma_base[d] + (use_x_on_gamma * beta_gamma_x * x_z[n]);
      gamma_t[n] = inv_logit(lg);
    }

    // AR(1) coefficients from p_t and gamma_t
    {
      real one_minus_p = 1 - p_t[n];
      real sum_terms = 0;
      for (i in 1:6) {
        sum_terms += pow(one_minus_p, 6 - i);
      }
      beta1_n[n] = pow(one_minus_p, 6);
      beta0_n[n] = (gamma_t[n] * pop_u5_ref[d]) * sum_terms;
    }
  }
}

model {
  // Priors on global-level hyperparameters
  p_global     ~ beta(p_hyperprior[1],     p_hyperprior[2]);
  gamma_global ~ beta(gamma_hyperprior[1], gamma_hyperprior[2]);

  // Pooling strengths (half-t on positive scale)
  kappa_p     ~ student_t(3, 0, 10);
  kappa_gamma ~ student_t(3, 0, 10);

  // Observation noise pooling
  alpha_global ~ lognormal(alpha_hyperprior[1], alpha_hyperprior[2]);
  sigma_alpha  ~ student_t(3, 0, 1);

  // Hierarchical priors for district baselines
  for (d in 1:D) {
    p[d]     ~ beta(p_global     * kappa_p,      (1 - p_global)     * kappa_p);
    gamma[d] ~ beta(gamma_global * kappa_gamma,  (1 - gamma_global) * kappa_gamma);
    alpha[d] ~ lognormal(log(alpha_global), sigma_alpha);
    P0[d]    ~ lognormal(5, 2.5);
  }

  // ---- Priors for covariate effects (controlled from R) ----
  // If toggled off, pin very close to 0 to avoid meaningless wandering.
  if (use_x_on_gamma == 1) beta_gamma_x ~ normal(0, beta_gamma_sd);
  else                     beta_gamma_x ~ normal(0, 1e-6);

  if (use_x_on_p == 1)     beta_p_x     ~ normal(0, beta_p_sd);
  else                     beta_p_x     ~ normal(0, 1e-6);

  // Likelihood with time-varying beta0_n, beta1_n
  for (n in 1:N) {
    int d = district_id[n];
    if (n == start_index[d]) {
      y[n] ~ normal(beta0_n[n] + beta1_n[n] * P0[d], alpha[d]);
    } else {
      y[n] ~ normal(beta0_n[n] + beta1_n[n] * y[n - 1], alpha[d]);
    }
  }
}

generated quantities {
  vector[N] log_lik;

  // Time-varying burden and gamma counts
  vector[N] burden_6mo;
  vector[N] burden;
  vector[N] gamma_count_t;   // incidence counts per month (since gamma_t is fraction of U5)
  vector[D] mu;              // baseline mean inter-arrival time from baseline gamma

  // Baseline mu (for comparability with base model)
  for (d in 1:D) {
    real gamma_count_baseline = gamma[d] * pop_u5_ref[d];
    mu[d] = 1 / gamma_count_baseline;
  }

  for (n in 1:N) {
    int d = district_id[n];

    gamma_count_t[n] = gamma_t[n] * pop_u5_ref[d];

    if (n == start_index[d]) {
      log_lik[n] = normal_lpdf(y[n] | beta0_n[n] + beta1_n[n] * P0[d], alpha[d]);
    } else {
      log_lik[n] = normal_lpdf(y[n] | beta0_n[n] + beta1_n[n] * y[n - 1], alpha[d]);
    }

    burden[n]     = y[n] + 12 * gamma_count_t[n];
    burden_6mo[n] = y[n] +  6 * gamma_count_t[n];
  }
}
