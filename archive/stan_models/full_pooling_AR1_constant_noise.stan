// Archived model (original file: full_pooling_model.stan). Converted to the current Stan array syntax;
// the model itself is unchanged.
data {
  int<lower=1> N;           // Total observations (across all districts)
  vector[N] y;              // Observed SAM values
  int<lower=1> D;           // Number of districts (for generated quantities)
  array[D] int<lower=1, upper=N> start_index; // First observation per district
}

parameters {
  real<lower=0, upper=1> p;
  real<lower=0> gamma;
  real<lower=0> alpha;
  real<lower=0> P0;
}

transformed parameters {
  real beta1 = pow(1 - p, 6);
  real beta0;

  {
    real sum_terms = 0;
    for (i in 1:6)
      sum_terms += pow(1 - p, 6 - i);
    beta0 = gamma * sum_terms;
  }
}

model {
  // Priors (same as hierarchical priors but global only)
  p ~ beta(3, 2);
  gamma ~ lognormal(5, 2.5);
  alpha ~ lognormal(5, 2.5);
  P0 ~ lognormal(5, 2.5);

  for (n in 1:N) {
    int d = 1;
    for (i in 1:D) {
      if (n == start_index[i]) d = i;
    }

    if (n == start_index[d])
      y[n] ~ normal(beta0 + beta1 * P0, alpha);
    else
      y[n] ~ normal(beta0 + beta1 * y[n - 1], alpha);
  }
}

generated quantities {
  vector[N] log_lik;
  vector[D] mu;
  vector[N] burden;
  vector[N] K;

  for (d in 1:D)
    mu[d] = 1 / gamma;

  for (n in 1:N) {
    int d = 1;
    for (i in 1:D) {
      if (n == start_index[i]) d = i;
    }

    real pred = (n == start_index[d])
      ? beta0 + beta1 * P0
      : beta0 + beta1 * y[n - 1];

    log_lik[n] = normal_lpdf(y[n] | pred, alpha);
    burden[n] = y[n] + 12 * gamma;
    K[n] = (y[n] > 0) ? (burden[n] / y[n] - 1) : 0;
  }
}
