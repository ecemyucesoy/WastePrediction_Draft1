// No-pooling model with district-specific priors supplied from JSON.
// Each district's p, gamma, alpha have their own Beta / LogNormal prior
// hyperparameters; there is no shared hyperprior linking districts.

data {
  int<lower=1> N;                         // total observations
  int<lower=1> D;                         // number of districts
  int<lower=1, upper=D> district_id[N];   // district per observation
  vector[N] y;                            // observed SAM counts
  vector<lower=0>[D] pop_u5_ref;          // U5 population per district

  int<lower=1, upper=N> start_index[D];   // first obs index per district

  // District-specific prior hyperparameters for p ~ Beta(a_p, b_p)
  vector<lower=0>[D] a_p;
  vector<lower=0>[D] b_p;

  // District-specific prior hyperparameters for gamma ~ Beta(a_gamma, b_gamma)
  vector<lower=0>[D] a_gamma;
  vector<lower=0>[D] b_gamma;

  // District-specific prior hyperparameters for alpha ~ LogNormal(mu_alpha, sigma_alpha)
  // mu_alpha is on the log scale (as returned by the LLM JSON)
  vector[D] mu_alpha;
  vector<lower=0>[D] sigma_alpha;
}

parameters {
  real<lower=1e-10, upper=1-1e-10> p[D];
  real<lower=1e-10, upper=1-1e-10> gamma[D];
  real<lower=1e-10> alpha[D];
  real<lower=1e-10> P0[D];
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
    beta0[d] = (gamma[d] * pop_u5_ref[d]) * sum_terms;
  }
}

model {
  // District-specific independent priors (no pooling)
  for (d in 1:D) {
    p[d]     ~ beta(a_p[d],     b_p[d]);
    gamma[d] ~ beta(a_gamma[d], b_gamma[d]);
    alpha[d] ~ lognormal(mu_alpha[d], sigma_alpha[d]);
    P0[d]    ~ lognormal(5, 2.5);
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

  vector[D] mu;
  vector[N] burden_6mo;
  vector[N] burden;
  vector[D] gamma_count;

  for (d in 1:D) {
    gamma_count[d] = gamma[d] * pop_u5_ref[d];
    mu[d]          = 1 / gamma_count[d];
  }

  for (n in 1:N) {
    int d = district_id[n];

    if (n == start_index[d]) {
      log_lik[n] = normal_lpdf(y[n] | beta0[d] + beta1[d] * P0[d], alpha[d]);
    } else {
      log_lik[n] = normal_lpdf(y[n] | beta0[d] + beta1[d] * y[n - 1], alpha[d]);
    }

    burden[n]     = y[n] + 12 * gamma_count[d];
    burden_6mo[n] = y[n] +  6 * gamma_count[d];
  }
}
