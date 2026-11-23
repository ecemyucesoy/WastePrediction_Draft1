// Theoretical variance model with district-specific LLM MIXTURE priors (no pooling across districts).
// Likelihood, state-space recursion, P0 prior, parameter bounds and generated quantities are those of
// no_pooling_LLM_informed_priors_monthly.stan. Only the prior on (p, gamma, omega) changes: instead of one
// Beta per parameter (a moment-matched summary of the LLM's runs), Stan receives the K elicited runs of
// every district intact and uses the actual mixture over runs as the prior:
//
//   joint = 1 :  prior(p[d], gamma[d], omega[d]) = sum_k w[k] Beta(p[d]|a_p[k,d], b_p[k,d])
//                                                        * Beta(gamma[d]|a_gamma[k,d], b_gamma[k,d])
//                                                        * Beta(omega[d]|a_omega[k,d], b_omega[k,d])
//                (each run supplies its p, gamma and omega together)
//   joint = 0 :  three independent mixtures (one per parameter), each sum_k w[k] Beta(.|a[k,d], b[k,d])
//
// Weights w:
//   estimate_w = 0 : w[k] = 1/K for every district (primary version)
//   estimate_w = 1 : one simplex[K] w shared by ALL districts, prior Dirichlet(1,...,1) (mean 1/K)
// The full beta_lpdf is used (not the dropped-constant "~" statement): the normalising constants differ
// between components, so they matter for the mixture.
// Run by no_pooling_LLM_mixture_priors.R.

data {
  int<lower=1> N;                         // Number of observations
  int<lower=1> D;                         // Number of districts
  array[N] int<lower=1, upper=D> district_id;   // District ID per observation
  vector[N] y;                            // Observed SAM prevalence
  vector<lower=0>[D] pop_u5_ref;          // Fixed U5 population per district
  array[D] int<lower=1, upper=N> start_index;   // First obs per district

  int<lower=1> K;                         // Number of elicited runs (mixture components)
  // Beta(a, b) hyperparameters of run k for district d, floored at 1e-6 (as in the single-Beta model)
  matrix<lower=1e-6>[K, D] a_p;
  matrix<lower=1e-6>[K, D] b_p;
  matrix<lower=1e-6>[K, D] a_gamma;
  matrix<lower=1e-6>[K, D] b_gamma;
  matrix<lower=1e-6>[K, D] a_omega;
  matrix<lower=1e-6>[K, D] b_omega;

  int<lower=0, upper=1> estimate_w;       // 0: w = 1/K fixed, 1: shared simplex with Dirichlet(1,...,1) prior
  int<lower=0, upper=1> joint;            // 1: joint mixture over runs, 0: separate mixture per parameter
}

transformed data {
  int K_w = estimate_w == 1 ? K : 1;      // size of the estimated simplex (1 = no free parameters)
}

parameters {
  array[D] real<lower=1e-10, upper=1-1e-10> p;       // Monthly departure probability
  array[D] real<lower=1e-10, upper=1-1e-10> gamma;   // Monthly incidence rate (fraction of U5 pop)
  array[D] real<lower=1e-10, upper=1-1e-10> omega;   // SD of monthly arrival rate (fraction of U5 pop)
  array[D] real<lower=1e-10> P0;                     // Initial latent prevalence (count scale, population-agnostic)
  simplex[K_w] w_est;                                // used only when estimate_w = 1
}

transformed parameters {
  vector[K] w     = estimate_w == 1 ? w_est : rep_vector(1.0 / K, K);
  vector[K] log_w = log(w);

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
  if (estimate_w == 1) {
    w_est ~ dirichlet(rep_vector(1.0, K));
  }

  // District-specific mixture priors (no pooling across districts)
  for (d in 1:D) {
    if (joint == 1) {
      vector[K] lp;
      for (k in 1:K) {
        lp[k] = log_w[k]
              + beta_lpdf(p[d]     | a_p[k, d],     b_p[k, d])
              + beta_lpdf(gamma[d] | a_gamma[k, d], b_gamma[k, d])
              + beta_lpdf(omega[d] | a_omega[k, d], b_omega[k, d]);
      }
      target += log_sum_exp(lp);
    } else {
      vector[K] lp_p;
      vector[K] lp_g;
      vector[K] lp_o;
      for (k in 1:K) {
        lp_p[k] = log_w[k] + beta_lpdf(p[d]     | a_p[k, d],     b_p[k, d]);
        lp_g[k] = log_w[k] + beta_lpdf(gamma[d] | a_gamma[k, d], b_gamma[k, d]);
        lp_o[k] = log_w[k] + beta_lpdf(omega[d] | a_omega[k, d], b_omega[k, d]);
      }
      target += log_sum_exp(lp_p) + log_sum_exp(lp_g) + log_sum_exp(lp_o);
    }
    P0[d] ~ lognormal(5, 2.5);
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
