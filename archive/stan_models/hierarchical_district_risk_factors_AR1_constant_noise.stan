// Archived model (original file: hierarchical_risk_model.stan). Converted to the current Stan array syntax;
// the model itself is unchanged.
// hierarchical_risk_model.stan

data {
  int<lower=1> N;
  int<lower=1> D;
  array[N] int<lower=1, upper=D> district_id;
  vector[N] y;

  vector[2] p_hyperprior;
  vector[2] gamma_hyperprior;
  vector[2] alpha_hyperprior;
  
  array[D] int<lower=1, upper=N> start_index;

  int<lower=1> K_p;
  int<lower=1> K_gamma;
  matrix[D, K_p] X_p;
  matrix[D, K_gamma] X_gamma;
}

parameters {
  vector[K_p] beta_p;
  vector[K_gamma] beta_gamma;

  vector[D] eta_p;
  vector[D] eta_gamma;

  real<lower=0, upper=1> p_global_raw;
  real<lower=0> gamma_global_raw;

  real<lower=0> sigma_p;
  real<lower=0> sigma_gamma;

  array[D] real<lower=0> alpha;
  real<lower=0> alpha_global;
  real<lower=0> sigma_alpha;

  array[D] real<lower=0> P0;
}

transformed parameters {
  real p_global = logit(p_global_raw);
  real gamma_global = log(gamma_global_raw);

  array[D] real<lower=0, upper=1> p;
  array[D] real<lower=0> gamma;
  array[D] real beta0;
  array[D] real beta1;

  for (d in 1:D) {
    p[d] = inv_logit(p_global + dot_product(X_p[d], beta_p) + eta_p[d]);
    gamma[d] = exp(gamma_global + dot_product(X_gamma[d], beta_gamma) + eta_gamma[d]);

    beta1[d] = pow(1 - p[d], 6);
    real sum_terms_d = 0;
    for (i in 1:6) {
      sum_terms_d += pow(1 - p[d], 6 - i);
    }
    beta0[d] = gamma[d] * sum_terms_d;
  }
}

model {
  beta_p ~ normal(0, 1);
  beta_gamma ~ normal(0, 1);

  p_global_raw ~ beta(p_hyperprior[1], p_hyperprior[2]);
  gamma_global_raw ~ lognormal(gamma_hyperprior[1], gamma_hyperprior[2]);

  eta_p ~ normal(0, sigma_p);
  eta_gamma ~ normal(0, sigma_gamma);

  sigma_p ~ student_t(3, 0, 2.5);
  sigma_gamma ~ student_t(3, 0, 2.5);

  alpha_global ~ lognormal(alpha_hyperprior[1], alpha_hyperprior[2]);
  sigma_alpha ~ student_t(3, 0, 2.5);
  alpha ~ lognormal(log(alpha_global), sigma_alpha);

  P0 ~ lognormal(5, 2.5);

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
  vector[N] burden;
  vector[N] burden_6mo;
  vector[N] K;

  for (d in 1:D) {
    mu[d] = 1 / gamma[d];
  }

  for (n in 1:N) {
    int d = district_id[n];
    if (n == start_index[d]) {
      log_lik[n] = normal_lpdf(y[n] | beta0[d] + beta1[d] * P0[d], alpha[d]);
    } else {
      log_lik[n] = normal_lpdf(y[n] | beta0[d] + beta1[d] * y[n - 1], alpha[d]);
    }

    burden[n] = y[n] + 12 * gamma[d];
    burden_6mo[n] = y[n] + 6 * gamma[d];
    K[n] = (y[n] > 0) ? (burden[n] / y[n] - 1) : 0;
  }
}
