data {
  int<lower=1> N;
  int<lower=1> D;
  int<lower=1> L;                        // number of lags (here 6)
  int<lower=1, upper=D> district_id[N];
  vector[N] y;
  vector<lower=0>[D] pop_u5_ref;
  int<lower=1, upper=N> start_index[D];

  vector[2] p_hyperprior;
  vector[2] gamma_hyperprior;
  vector[2] alpha_hyperprior;

  matrix[N, L] X_lag;

  int<lower=0, upper=1> use_x_on_p;
  int<lower=0, upper=1> use_x_on_gamma;

  real<lower=0> hs_tau0;        // global shrinkage scale
  real<lower=0> hs_slab_scale;  // slab scale
  real<lower=0> hs_slab_df;     // slab df (>0)
}

parameters {
  // District baselines
  real<lower=1e-10, upper=1-1e-10> p[D];
  real<lower=1e-10, upper=1-1e-10> gamma[D];
  real<lower=1e-10> alpha[D];
  real<lower=1e-10> P0[D];

  // Global means and pooling strengths
  real<lower=1e-10, upper=1-1e-10> p_global;
  real<lower=1e-10, upper=1-1e-10> gamma_global;
  real<lower=1e-10> kappa_p;
  real<lower=1e-10> kappa_gamma;

  // Observation noise pooling
  real<lower=1e-10> alpha_global;
  real<lower=1e-10> sigma_alpha;

  // Regularized horseshoe components (p channel)
  vector[L] z_p;
  vector<lower=0>[L] lambda_p;
  real<lower=0> tau_p;

  // Regularized horseshoe components (gamma channel)
  vector[L] z_g;
  vector<lower=0>[L] lambda_g;
  real<lower=0> tau_g;

  // slab raw scale (we’ll scale it in transformed parameters)
  real<lower=0> c2_raw;
}

transformed parameters {
  vector[D] logit_p_base;
  vector[D] logit_gamma_base;

  vector[N] p_t;
  vector[N] gamma_t;

  vector[N] beta1_n;
  vector[N] beta0_n;

  vector[L] beta_p_lag;
  vector[L] beta_gamma_lag;

  real<lower=0> c2;  // slab variance parameter (scaled)
  c2 = square(hs_slab_scale) * c2_raw;

  for (d in 1:D) {
    logit_p_base[d]     = logit(p[d]);
    logit_gamma_base[d] = logit(gamma[d]);
  }

  // regularized horseshoe transform
  {
    vector[L] lambda_p_tilde;
    vector[L] lambda_g_tilde;

    for (l in 1:L) {
      lambda_p_tilde[l] = sqrt( (c2 * square(lambda_p[l])) /
                                (c2 + square(tau_p) * square(lambda_p[l])) );
      lambda_g_tilde[l] = sqrt( (c2 * square(lambda_g[l])) /
                                (c2 + square(tau_g) * square(lambda_g[l])) );
    }

    beta_p_lag     = z_p .* (tau_p * lambda_p_tilde);
    beta_gamma_lag = z_g .* (tau_g * lambda_g_tilde);
  }

  // time-varying p_t, gamma_t and AR coefficients
  for (n in 1:N) {
    int d = district_id[n];

    // row n of X_lag is a row_vector; convert to vector for dot_product
    real xb_p = use_x_on_p     * dot_product(to_vector(X_lag[n]), beta_p_lag);
    real xb_g = use_x_on_gamma * dot_product(to_vector(X_lag[n]), beta_gamma_lag);

    p_t[n]     = inv_logit(logit_p_base[d]     + xb_p);
    gamma_t[n] = inv_logit(logit_gamma_base[d] + xb_g);

    {
      real one_minus_p = 1 - p_t[n];
      real sum_terms = 0;
      for (i in 1:6) sum_terms += pow(one_minus_p, 6 - i);

      beta1_n[n] = pow(one_minus_p, 6);
      beta0_n[n] = (gamma_t[n] * pop_u5_ref[d]) * sum_terms;
    }
  }
}

model {
  // Global-level priors
  p_global     ~ beta(p_hyperprior[1],     p_hyperprior[2]);
  gamma_global ~ beta(gamma_hyperprior[1], gamma_hyperprior[2]);

  kappa_p     ~ student_t(3, 0, 10);
  kappa_gamma ~ student_t(3, 0, 10);

  alpha_global ~ lognormal(alpha_hyperprior[1], alpha_hyperprior[2]);
  sigma_alpha  ~ student_t(3, 0, 1);

  // District-level baselines
  for (d in 1:D) {
    p[d]     ~ beta(p_global     * kappa_p,     (1 - p_global)     * kappa_p);
    gamma[d] ~ beta(gamma_global * kappa_gamma, (1 - gamma_global) * kappa_gamma);
    alpha[d] ~ lognormal(log(alpha_global), sigma_alpha);
    P0[d]    ~ lognormal(5, 2.5);
  }

  // Regularized horseshoe priors
  z_p ~ normal(0, 1);
  z_g ~ normal(0, 1);

  lambda_p ~ cauchy(0, 1);
  lambda_g ~ cauchy(0, 1);

  tau_p ~ cauchy(0, hs_tau0);
  tau_g ~ cauchy(0, hs_tau0);

  // slab raw component
  c2_raw ~ inv_gamma(0.5 * hs_slab_df, 0.5 * hs_slab_df);

  // Likelihood
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
  vector[N] burden_6mo;
  vector[N] burden;
  vector[N] gamma_count_t;
  vector[D] mu;

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