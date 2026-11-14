data {
  int<lower=1> N;                 // Number of observations
  int<lower=1> D;                 // Number of districts
  int<lower=1, upper=D> district_id[N]; // District ID for each obs
  vector[N] y;                    // Observed SAM values
  int<lower=1, upper=N> start_index[D]; // Index of first obs per district
}

parameters {
  real<lower=0, upper=1> p[D];
  real<lower=0> gamma[D];
  real<lower=0> alpha[D];
  real<lower=0> P0[D];
}

transformed parameters {
  real beta0[D];
  real beta1[D];

  for (d in 1:D) {
    beta1[d] = pow(1 - p[d], 6);
    real sum_terms = 0;
    for (i in 1:6)
      sum_terms += pow(1 - p[d], 6 - i);
    beta0[d] = gamma[d] * sum_terms;
  }
}

model {
  for (d in 1:D) {
    p[d] ~ beta(3, 2);
    gamma[d] ~ lognormal(5, 2.5);
    alpha[d] ~ lognormal(5, 2.5);
    P0[d] ~ lognormal(5, 2.5);
  }

  for (n in 1:N) {
    int d = district_id[n];
    if (n == start_index[d])
      y[n] ~ normal(beta0[d] + beta1[d] * P0[d], alpha[d]);
    else
      y[n] ~ normal(beta0[d] + beta1[d] * y[n - 1], alpha[d]);
  }
}

generated quantities {
  vector[N] log_lik;
  vector[D] mu;
  vector[N] burden;
  vector[N] K;

  for (d in 1:D)
    mu[d] = 1 / gamma[d];

  for (n in 1:N) {
    int d = district_id[n];
    real pred = (n == start_index[d])
      ? beta0[d] + beta1[d] * P0[d]
      : beta0[d] + beta1[d] * y[n - 1];
    log_lik[n] = normal_lpdf(y[n] | pred, alpha[d]);
    burden[n] = y[n] + 12 * gamma[d];
    K[n] = (y[n] > 0) ? (burden[n] / y[n] - 1) : 0;
  }
}
