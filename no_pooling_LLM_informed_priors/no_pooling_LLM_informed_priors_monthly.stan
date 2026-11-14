// Theoretical variance model with district-specific informed priors (no pooling).
// Variance is derived from the SIS process structure (affine in P_prev via omega).
// District priors for p, gamma, omega arrive as Beta(a,b) hyperparameter vectors,
// read in R directly from the LLM-elicited JSON files (or pooled across runs) -- see
// no_pooling_LLM_informed_priors.R.

data {
  int<lower=1> N;                         // Number of observations
  int<lower=1> D;                         // Number of districts
  array[N] int<lower=1, upper=D> district_id;   // District ID per observation
  vector[N] y;                            // Observed SAM prevalence
  vector<lower=0>[D] pop_u5_ref;          // Fixed U5 population per district
  array[D] int<lower=1, upper=N> start_index;   // First obs per district

  // District-specific Beta prior hyperparameters (a, b): read directly from an LLM prior JSON
  // (one model per JSON) or pooled across an LLM's runs, in no_pooling_LLM_informed_priors.R.
  // Floored at 1e-6 (not 0) to match the positivity floor enforced on the
  // fitted (a,b) before they reach Stan.
  vector<lower=1e-6>[D] a_p;
  vector<lower=1e-6>[D] b_p;
  vector<lower=1e-6>[D] a_gamma;
  vector<lower=1e-6>[D] b_gamma;
  vector<lower=1e-6>[D] a_omega;
  vector<lower=1e-6>[D] b_omega;
}

parameters {
  array[D] real<lower=1e-10, upper=1-1e-10> p;       // Monthly departure probability
  array[D] real<lower=1e-10, upper=1-1e-10> gamma;   // Monthly incidence rate (fraction of U5 pop)
  array[D] real<lower=1e-10, upper=1-1e-10> omega;   // SD of monthly arrival rate (fraction of U5 pop)
  array[D] real<lower=1e-10> P0;                     // Initial latent prevalence (count scale, population-agnostic)
}

transformed parameters {
  array[D] real beta1;
  array[D] real beta0;
  array[D] real c1;
  array[D] real c0_arrival_bernoulli;
  array[D] real sum_pi_sq;

  for (d in 1:D) {
    real q            = 1 - p[d];
    real gamma_count  = gamma[d] * pop_u5_ref[d];

    beta1[d] = pow(q, 6);
    c1[d]    = beta1[d] * (1 - beta1[d]);

    real sum_pi_s           = 0;
    real sum_pi_s_one_minus = 0;
    real sum_pi_s_sq_local  = 0;
    for (s in 1:6) {
      real pi_s = pow(q, 6 - s);
      sum_pi_s           += pi_s;
      sum_pi_s_one_minus += pi_s * (1 - pi_s);
      sum_pi_s_sq_local  += pi_s * pi_s;
    }
    beta0[d]                = gamma_count * sum_pi_s;
    c0_arrival_bernoulli[d] = gamma_count * sum_pi_s_one_minus;
    sum_pi_sq[d]            = sum_pi_s_sq_local;
  }
}

model {
  // District-specific independent priors (no pooling)
  for (d in 1:D) {
    p[d]     ~ beta(a_p[d],     b_p[d]);
    gamma[d] ~ beta(a_gamma[d], b_gamma[d]);
    omega[d] ~ beta(a_omega[d], b_omega[d]);
    P0[d]    ~ lognormal(5, 2.5);
  }

  // Likelihood: variance derived from SIS process (affine in P_prev)
  for (n in 1:N) {
    int d = district_id[n];
    real P_prev = (n == start_index[d]) ? P0[d] : y[n - 1];

    real omega_count   = omega[d] * pop_u5_ref[d];
    real variance_term = c1[d] * P_prev
                       + c0_arrival_bernoulli[d]
                       + omega_count^2 * sum_pi_sq[d];
    y[n] ~ normal(beta0[d] + beta1[d] * P_prev, sqrt(variance_term));
  }
}

generated quantities {
  vector[N] log_lik;
  vector[N] mu_pred;
  vector[N] sd_pred;
  vector[N] y_rep;        // posterior predictive draws, for PPC / CV coverage checks
  vector[N] p_negative;   // P(Normal likelihood < 0 | mu_pred, sd_pred): mass on impossible values
  vector[D] gamma_count_out;
  vector[D] omega_count_out;
  vector[D] mean_duration;
  vector[N] burden;
  vector[N] burden_6mo;

  for (d in 1:D) {
    gamma_count_out[d] = gamma[d] * pop_u5_ref[d];
    omega_count_out[d] = omega[d] * pop_u5_ref[d];
    mean_duration[d]   = 1 / p[d];
  }

  for (n in 1:N) {
    int d = district_id[n];
    real P_prev = (n == start_index[d]) ? P0[d] : y[n - 1];

    real omega_count   = omega[d] * pop_u5_ref[d];
    real variance_term = c1[d] * P_prev
                       + c0_arrival_bernoulli[d]
                       + omega_count^2 * sum_pi_sq[d];
    real sd_term = sqrt(variance_term);

    mu_pred[n]    = beta0[d] + beta1[d] * P_prev;
    sd_pred[n]    = sd_term;
    log_lik[n]    = normal_lpdf(y[n] | mu_pred[n], sd_term);
    y_rep[n]      = normal_rng(mu_pred[n], sd_pred[n]);
    p_negative[n] = Phi(-mu_pred[n] / sd_pred[n]);

    burden[n]     = y[n] + 12 * gamma_count_out[d];
    burden_6mo[n] = y[n] +  6 * gamma_count_out[d];
  }
}
