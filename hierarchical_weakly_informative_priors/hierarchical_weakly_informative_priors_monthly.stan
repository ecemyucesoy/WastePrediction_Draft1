data {
  int<lower=1> N;                         // Number of observations
  int<lower=1> D;                         // Number of districts
  array[N] int<lower=1, upper=D> district_id;   // District ID per observation
  vector[N] y;                            // Observed SAM prevalence
  vector<lower=0>[D] pop_u5_ref;          // Fixed U5 population per district
  array[D] int<lower=1, upper=N> start_index;   // First obs per district

  vector[2] p_hyperprior;                 // Beta hyperprior for p_global
  vector[2] gamma_hyperprior;             // Beta hyperprior for gamma_global
  vector[2] omega_hyperprior;             // Beta hyperprior for omega_global (fraction, same structure as gamma)
}

parameters {
  // District-level parameters
  array[D] real<lower=1e-10, upper=1-1e-10> p;          // Monthly departure probability
  array[D] real<lower=1e-10, upper=1-1e-10> gamma;      // Monthly incidence rate (fraction of U5 pop)
  array[D] real<lower=1e-10, upper=1-1e-10> omega;      // SD of monthly arrival rate (fraction of U5 pop)
  array[D] real<lower=1e-10> P0;                        // Initial latent prevalence

  // Global hyperparameters
  real<lower=1e-10, upper=1-1e-10> p_global;
  real<lower=1e-10, upper=1-1e-10> gamma_global;
  real<lower=1e-10, upper=1-1e-10> omega_global;  // Global mean of omega fraction
  real<lower=1e-10> kappa_p;
  real<lower=1e-10> kappa_gamma;
  real<lower=1e-10> kappa_omega;                  // Concentration for omega (mirrors kappa_gamma)
}

transformed parameters {
  // Quantities used in both mean and variance
  array[D] real beta1;                  // = pi_old
  array[D] real beta0;                  // = sum_s gamma_count * pi_s
  array[D] real c1;                     // = pi_old * (1 - pi_old)
  array[D] real c0_arrival_bernoulli;   // = sum_s gamma_count * pi_s * (1 - pi_s)
  array[D] real sum_pi_sq;              // = sum_s pi_s^2

  for (d in 1:D) {
    real q = 1 - p[d];                              // Monthly survival probability
    real gamma_count = gamma[d] * pop_u5_ref[d];    // Mean monthly arrivals in counts
    real omega_count = omega[d] * pop_u5_ref[d];    // SD of monthly arrivals in counts

    beta1[d] = pow(q, 6);                           // pi_old = (1-p)^6
    c1[d]    = beta1[d] * (1 - beta1[d]);           // pi_old * (1 - pi_old)

    real sum_pi_s             = 0;
    real sum_pi_s_one_minus   = 0;
    real sum_pi_s_sq_local    = 0;
    for (s in 1:6) {
      real pi_s = pow(q, 6 - s);                    // Survival prob for arrival in month s
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
  // Priors on global-level hyperparameters
  p_global     ~ beta(p_hyperprior[1],     p_hyperprior[2]);
  gamma_global ~ beta(gamma_hyperprior[1], gamma_hyperprior[2]);
  omega_global ~ beta(omega_hyperprior[1], omega_hyperprior[2]);

  kappa_p     ~ student_t(3, 0, 10);
  kappa_gamma ~ student_t(3, 0, 10);
  kappa_omega ~ student_t(3, 0, 10);

  // District-level priors
  for (d in 1:D) {
    p[d]     ~ beta(p_global     * kappa_p,     (1 - p_global)     * kappa_p);
    gamma[d] ~ beta(gamma_global * kappa_gamma, (1 - gamma_global) * kappa_gamma);
    omega[d] ~ beta(omega_global * kappa_omega, (1 - omega_global) * kappa_omega);
    P0[d]    ~ lognormal(5, 2.5);
  }

  // Likelihood: variance is affine in previous prevalence (from proof)
  for (n in 1:N) {
    int d = district_id[n];
    real P_prev;
    if (n == start_index[d]) {
      P_prev = P0[d];
    } else {
      P_prev = y[n - 1];
    }

    // alpha^2(P_prev) = c1 * P_prev  +  c0_arrival_bernoulli  +  omega_count^2 * sum_pi_sq
    // with omega_count = omega[d] * pop_u5_ref[d]
    real omega_count   = omega[d] * pop_u5_ref[d];
    real variance_term = c1[d] * P_prev
                       + c0_arrival_bernoulli[d]
                       + omega_count^2 * sum_pi_sq[d];
    real sd_term = sqrt(variance_term);
    y[n] ~ normal(beta0[d] + beta1[d] * P_prev, sd_term);
  }
}

generated quantities {
  vector[N] log_lik;
  vector[N] mu_pred;            // Conditional mean prediction at each obs
  vector[N] sd_pred;            // Conditional SD prediction at each obs
  vector[D] gamma_count_out;    // Monthly incidence in counts
  vector[D] omega_count_out;    // Monthly arrival SD in counts
  vector[D] mean_duration;      // 1/p in months
  vector[N] burden;             // Annual burden estimate
  vector[N] burden_6mo;         // 6-month burden estimate

  for (d in 1:D) {
    gamma_count_out[d] = gamma[d] * pop_u5_ref[d];
    omega_count_out[d] = omega[d] * pop_u5_ref[d];
    mean_duration[d]   = 1 / p[d];
  }

  for (n in 1:N) {
    int d = district_id[n];
    real P_prev;
    if (n == start_index[d]) {
      P_prev = P0[d];
    } else {
      P_prev = y[n - 1];
    }

    real omega_count   = omega[d] * pop_u5_ref[d];
    real variance_term = c1[d] * P_prev
                       + c0_arrival_bernoulli[d]
                       + omega_count^2 * sum_pi_sq[d];
    real sd_term = sqrt(variance_term);

    mu_pred[n]  = beta0[d] + beta1[d] * P_prev;
    sd_pred[n]  = sd_term;
    log_lik[n]  = normal_lpdf(y[n] | mu_pred[n], sd_term);

    burden[n]     = y[n] + 12 * gamma_count_out[d];
    burden_6mo[n] = y[n] + 6  * gamma_count_out[d];
  }
}
