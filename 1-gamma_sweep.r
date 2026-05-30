#!/usr/bin/env Rscript

# Load necessary libraries
if (!require("optparse")) install.packages("optparse", repos="http://cran.us.r-project.org")
library(optparse)

# Data Generation (Standard SEM from Kania & Wit)
generate_data <- function&î, shift = 0) {
  eps <- matrix(rnorm(n * 7), ncol = 7)
  x1 <- eps[, 1]
  x2 <- x1 + eps[, 2] # Parent
  x3 <- x1 + x2 + eps[, 3] # Parent
  y  <- x2 + x3 + eps[, 4] # Target: Causal parents are X2, X3
  x4 <- y + shift + eps[, 5] # Spurious child
  x5 <- y + shift + eps[, 6] # Spurious child
  x6 <- x5 + eps[, 7]
  X  <- cbind(x1, x2, x3, x4, x5, x6)
  return(list(X = X, y = y))
}

# ACR Objective Function
acr_loss <- function(beta, X, y, env_idx, gamma) {
  # Split by pseudo-environments
  r1 <- mean((y[env_idx==1] - X[env_idx==1,] %*% beta)^2)
  r2 <- mean((y[env_idx==2] - X[env_idx==2,] %*% beta)^2)
  return((r1 + r2) + gamma * abs(r1 - r2))
}

# --- RUN CONFIGURATION ---
gammas <- c(0, 0.5, 1, 5, 10, 20, 50, 100) # The "Sweep"
n_samples <- 500
test_shift <- 5.0 # Significant adversarial shift
results <- data.frame()

cat("Starting Gamma Sweep...\n")

for (g in gammas) {
  cat("Processing Gamma:", g, "\n")
  
  # 1. Setup Training (Internal split for pseudo-environments)
  train_data <- generate_data(n_samples, shift = 0)
  env_idx <- rep(1:2, length.out = n_samples)
  
  # 2. Optimization
  fit <- optim(par = rep(0, 6), fn = acr_loss, 
               X = train_data$X, y = train_data$y, 
               env_idx = env_idx, gamma = g, method = "BFGS")
  
  beta_hat <- fit$par
  
  # 3. Evaluation
  # In-Sample Risk (Train env)
  risk_in <- mean((train_data$y - train_data$X %*% beta_hat)^2)
  
  # Out-of-Sample Risk (Shifted env)
  test_data <- generate_data(n_samples, shift = test_shift)
  risk_out <- mean((test_data$y - test_data$X %*% beta_hat)^2)
  
  # 4. Store Results
  row <- data.frame(gamma = g, risk_in = risk_in, risk_out = risk_out,
                    b1=beta_hat[1], b2=beta_hat[2], b3=beta_hat[3], 
                    b4=beta_hat[4], b5=beta_hat[5], b6=beta_hat[6])
  results <- rbind(results, row)
}

# Save results to CSV
write.csv(results, "1-acr_experiment_results.csv", row.names = FALSE)
cat("Experiment complete. Results saved to 'acr_experiment_results.csv'.\n")