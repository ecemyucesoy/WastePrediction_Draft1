data {
  int<lower=1> N;                     // number of observations
  int<lower=1> D;                     // number of districts
  int<lower=1, upper=D> district_id[N]; // district index
  vector[N] y;                        // observed SAM cases
  vector[2] p_hyperprior;            // hyperprior for p
  vector[2] gamma_hyperprior;
  vector[2] alpha_hyperprior;
  int<lower=1, upper=N> start_index[D]; // start of time series per district
}

parameters {
  real<lower=0, upper=1> p[D];        // DAILY exit probability
  real<lower=0> gamma[D];             // DAILY incidence rate
  real<lower=0> alpha[D];             // noise stddev
  real<lower=0> P0[D];                // initial SAM level

  real<lower=0, upper=1> p_global;
  real<lower=0> gamma_global;
  real<lower=0> alpha_global;
  real<lower=0> sigma_p;
  real<lower=0> sigma_gamma;
  real<lower=0> sigma_alpha;
}

transformed parameters {
  real beta1[D];
  real beta0[D];

  for (d in 1:D) {
    beta1[d] = pow(1 - p[d], 180);
    real sum_terms = 0;
    for (i in 1:180) {
      sum_terms += pow(1 - p[d], 180 - i);
    }
    beta0[d] = gamma[d] * sum_terms;
  }
}

model {
  // Priors
  p_global ~ beta(p_hyperprior[1], p_hyperprior[2]);
  gamma_global ~ lognormal(gamma_hyperprior[1], gamma_hyperprior[2]);
  alpha_global ~ lognormal(alpha_hyperprior[1], alpha_hyperprior[2]);

  sigma_p ~ student_t(3, 0, 2.5);
  sigma_gamma ~ student_t(3, 0, 2.5);
  sigma_alpha ~ student_t(3, 0, 2.5);

  for (d in 1:D) {
    p[d] ~ beta(p_global * (1 / sigma_p), (1 - p_global) * (1 / sigma_p));
    gamma[d] ~ lognormal(log(gamma_global), sigma_gamma);
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
  vector[D] mu;
  vector[N] burden;
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

    burden[n] = y[n] + 365 * gamma[d];  // Daily gamma, annual burden
    K[n] = (y[n] > 0) ? (burden[n] / y[n] - 1) : 0;
  }
}
