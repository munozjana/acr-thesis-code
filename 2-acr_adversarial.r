#!/usr/bin/env Rscript

# 1. Data Generation (Linear SEM)
generate_data <- function(n, shift = 0) {
  eps <- matrix(rnorm(n * 7), ncol = 7)
  x2 <- eps[, 1] + eps[, 2]
  x3 <- eps[, 1] + x2 + eps[, 3]
  y  <- x2 + x3 + eps[, 4]  # Causal: X2, X3
  x4 <- y + shift + eps[, 5] # Spurious child
  x5 <- y + shift + eps[, 6] # Spurious child
  X  <- cbind(eps[,1], x2, x3, x4, x5, eps[,7])
  return(list(X = X, y = y))
}

# 2. Adversarial ACR Logic
# We optimize Beta to minimize: Risk_plus + Gamma * |Risk_Delta|
# Where Risk_Delta is found by an adversary weights 'w'
run_acr_adversarial <- function(X, y, gamma, iterations = 10) {
  n <- nrow(X)
  beta <- rep(0, ncol(X))
  # Initialize adversary: random binary-ish weights to start
  w <- rbinom(n, 1, 0.5) 
  
  for (i in 1:iterations) {
    # --- LEARNER STEP: Minimize Risk given current Adversary weights 'w' ---
    learner_obj <- function(b) {
      err <- (y - X %*% b)^2
      r1 <- mean(err[w == 1])
      r2 <- mean(err[w == 0])
      return((r1 + r2) + gamma * abs(r1 - r2))
    }
    beta <- optim(par = beta, fn = learner_obj, method = "BFGS")$par
    
    # --- ADVERSARY STEP: Find weights 'w' that maximize Risk Difference ---
    # The adversary looks for the split where the model performs most inconsistently
    errors <- (y - X %*% beta)^2
    # Simple adversary: assign w=1 to high errors, w=0 to low errors (or vice versa)
    # to maximize the absolute difference |mean(err1) - mean(err2)|
    threshold <- median(errors)
    w <- ifelse(errors > threshold, 1, 0)
  }
  return(beta)
}

# --- EXPERIMENT ---
set.seed(42)
gammas <- c(0, 5, 20, 100)
train_data <- generate_data(500, shift = 0)
test_data  <- generate_data(500, shift = 5.0) # Distribution Shift

cat("Gamma | In-Sample MSE | Out-of-Sample MSE\n")
cat("------------------------------------------\n")

for (g in gammas) {
  beta_acr <- run_acr_adversarial(train_data$X, train_data$y, gamma = g)
  
  mse_in <- mean((train_data$y - train_data$X %*% beta_acr)^2)
  mse_out <- mean((test_data$y - test_data$X %*% beta_acr)^2)
  
  cat(sprintf("%5d | %12.4f | %17.4f\n", g, mse_in, mse_out))
}
