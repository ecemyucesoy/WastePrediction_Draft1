data {
  int<lower=1> N;                 // Number of observations
  int<lower=1> D;                 // Number of districts
  int<lower=1, upper=D> district_id[N]; // District ID per observation
  vector[N] y;                    // Observed SAM values
  vector<lower=0>[D] pop_u5_ref;   // One fixed U5 population per district
  int<lower=1, upper=N> start_index[D]; // Index of the first obs per district
  
  vector[2] p_hyperprior;
  vector[2] gamma_hyperprior;
  vector[2] alpha_hyperprior;
  
}

parameters {
  real<lower=1e-10, upper=1-1e-10> p[D];
  real<lower=1e-10, upper=1-1e-10> gamma[D];
  real<lower=1e-10> alpha[D];
  real<lower=1e-10> P0[D];

  real<lower=1e-10, upper=1-1e-10> p_global;
  real<lower=1e-10, upper=1-1e-10> gamma_global;

  real<lower=1e-10> kappa_p;
  real<lower=1e-10> kappa_gamma;

  real<lower=1e-10> alpha_global;
  real<lower=1e-10> sigma_alpha;
}

transformed parameters {
  real beta1[D];
  real beta0[D];

  for (d in 1:D) {
    beta1[d] = pow(1 - p[d], 6);
    real sum_terms = 0;
    for (i in 1:6) {
      sum_terms += pow(1 - p[d], 6 - i);
    }
    // gamma is a fraction; convert to counts by multiplying U5 population
    beta0[d] = (gamma[d] * pop_u5_ref[d]) * sum_terms;
  }
}

model {
  // Priors on global-level hyperparameters
  p_global ~ beta(p_hyperprior[1], p_hyperprior[2]);
  gamma_global ~ beta(gamma_hyperprior[1], gamma_hyperprior[2]);
  
  // ----- Pooling strengths (weak, heavy-tailed on positive scale) -----
  // Half-t via lower=0 constraint
  kappa_p     ~ student_t(3, 0, 10);
  kappa_gamma ~ student_t(3, 0, 10);
  
  
  // ----- Observation noise pooling -----
  alpha_global ~ lognormal(alpha_hyperprior[1], alpha_hyperprior[2]);
  sigma_alpha  ~ student_t(3, 0, 1);
  
  
  // Hierarchical priors
  for (d in 1:D) {
    p[d]     ~ beta(p_global     * kappa_p,      (1 - p_global)     * kappa_p);
    gamma[d] ~ beta(gamma_global * kappa_gamma,  (1 - gamma_global) * kappa_gamma);
    alpha[d] ~ lognormal(log(alpha_global), sigma_alpha);
    P0[d] ~ lognormal(5, 2.5);

  }
  
  // Likelihood
  for (n in 1:N) {
    int d = district_id[n];
    if (n == start_index[d]) {
      y[n] ~ normal(beta0[d] + beta1[d] * P0[d], alpha[d]);
    } else {
      y[n] ~ normal(beta0[d] + beta1[d] * y[n - 1], alpha[d]);
    }
  }
}

generated quantities {
  vector[N] log_lik;
  
  // Diagnostics / interpretable outputs
  vector[D] mu;
  vector[N] burden_6mo;
  vector[N] burden;
  vector[D] gamma_count;    // incidence as counts per month

  for (d in 1:D) {
    gamma_count[d] = gamma[d] * pop_u5_ref[d];
    mu[d]          = 1 / gamma_count[d];
  }

  for (n in 1:N) {
    int d = district_id[n];

    // Log-likelihood
    if (n == start_index[d]) {
      log_lik[n] = normal_lpdf(y[n] | beta0[d] + beta1[d] * P0[d], alpha[d]);
    } else {
      log_lik[n] = normal_lpdf(y[n] | beta0[d] + beta1[d] * y[n - 1], alpha[d]);
    }

    // Burden estimates
    burden[n] = y[n] + 12 * gamma_count[d];
    burden_6mo[n] = y[n] + 6 * gamma_count[d];

  }
}




