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

  // Separate lag matrices for each covariate (z-scored)
  matrix[N, L] X_lag_rain;
  matrix[N, L] X_lag_ndvi;
  matrix[N, L] X_lag_alarms;

  int<lower=0, upper=1> use_x_on_p;
  int<lower=0, upper=1> use_x_on_gamma;

  real<lower=0> dir_concentration;  // Dirichlet concentration (1 = flat/symmetric)
  real<lower=0> alpha_scale;        // prior SD for effect-size scalars
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

  // Dirichlet lag weight vectors (shared across p/gamma channels per covariate)
  simplex[L] w_rain;
  simplex[L] w_ndvi;
  simplex[L] w_alarms;

  // Effect-size scalars: one per covariate × channel
  real alpha_p_rain;
  real alpha_p_ndvi;
  real alpha_p_alarms;

  real alpha_g_rain;
  real alpha_g_ndvi;
  real alpha_g_alarms;
}

transformed parameters {
  vector[D] logit_p_base;
  vector[D] logit_gamma_base;

  vector[N] p_t;
  vector[N] gamma_t;

  vector[N] beta1_n;
  vector[N] beta0_n;

  for (d in 1:D) {
    logit_p_base[d]     = logit(p[d]);
    logit_gamma_base[d] = logit(gamma[d]);
  }

  // Time-varying p_t, gamma_t driven by Dirichlet-weighted lag composites
  for (n in 1:N) {
    int d = district_id[n];

    // Weighted average of lagged values for each covariate
    real x_eff_rain   = dot_product(to_vector(X_lag_rain[n]),   w_rain);
    real x_eff_ndvi   = dot_product(to_vector(X_lag_ndvi[n]),   w_ndvi);
    real x_eff_alarms = dot_product(to_vector(X_lag_alarms[n]), w_alarms);

    real xb_p = use_x_on_p     * (alpha_p_rain   * x_eff_rain +
                                   alpha_p_ndvi   * x_eff_ndvi +
                                   alpha_p_alarms * x_eff_alarms);

    real xb_g = use_x_on_gamma * (alpha_g_rain   * x_eff_rain +
                                   alpha_g_ndvi   * x_eff_ndvi +
                                   alpha_g_alarms * x_eff_alarms);

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

  // Dirichlet priors on lag weights
  w_rain   ~ dirichlet(rep_vector(dir_concentration, L));
  w_ndvi   ~ dirichlet(rep_vector(dir_concentration, L));
  w_alarms ~ dirichlet(rep_vector(dir_concentration, L));

  // Normal priors on effect-size scalars
  alpha_p_rain   ~ normal(0, alpha_scale);
  alpha_p_ndvi   ~ normal(0, alpha_scale);
  alpha_p_alarms ~ normal(0, alpha_scale);

  alpha_g_rain   ~ normal(0, alpha_scale);
  alpha_g_ndvi   ~ normal(0, alpha_scale);
  alpha_g_alarms ~ normal(0, alpha_scale);

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

  // Effective lag index: E[l] = sum_{l=1}^{L} l * w[l]
  real eff_lag_rain;
  real eff_lag_ndvi;
  real eff_lag_alarms;

  {
    vector[L] lag_index;
    for (l in 1:L) lag_index[l] = l;
    eff_lag_rain   = dot_product(lag_index, w_rain);
    eff_lag_ndvi   = dot_product(lag_index, w_ndvi);
    eff_lag_alarms = dot_product(lag_index, w_alarms);
  }

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
