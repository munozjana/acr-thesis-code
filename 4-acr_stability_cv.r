#!/usr/bin/env Rscript

# Load data generator
source("2-acr_adversarial.r")

# Soft Adversary ACR Function (same as defined in 4-acr_cv.r)
fit_acr_soft <- function(X, y, gamma, iterations = 15) {
  n <- nrow(X)
  p <- ncol(X)
  beta <- rep(0, p)
  w <- rep(0.5, n) # Initial neutral weights

  for (i in 1:iterations) {
    learner_obj <- function(b) {
      err <- (y - X %*% b)^2
      r1 <- sum(w * err) / sum(w)
      r2 <- sum((1 - w) * err) / sum(1 - w)
      return((r1 + r2) + gamma * abs(r1 - r2))
    }
    beta <- optim(par = beta, fn = learner_obj, method = "BFGS")$par

    errors <- (y - X %*% beta)^2
    w <- 1 / (1 + exp(-(errors - mean(errors)) / sd(errors)))
  }
  return(beta)
}

run_stability_cv <- function(X, y, gamma_range, k = 5) {
  n <- nrow(X)
  folds <- sample(rep(1:k, length.out = n))
  cv_results <- data.frame()
  
  for (g in gamma_range) {
    fold_errors <- c()
    
    for (f in 1:k) {
      X_train <- X[folds != f, ]
      y_train <- y[folds != f]
      X_val   <- X[folds == f, ]
      y_val   <- y[folds == f]
      
      # Train using our adversarial learner
      beta_fold <- fit_acr_soft(X_train, y_train, gamma = g)
      
      # Validate
      val_mse <- mean((y_val - X_val %*% beta_fold)^2)
      fold_errors <- c(fold_errors, val_mse)
    }
    
    # CALCULATE METRICS
    avg_mse <- mean(fold_errors)
    st_dev  <- sd(fold_errors) # This is our 'Invariance' metric
    
    # The 'Stability Score' favors models that are consistent across folds
    # We penalize high variance (instability)
    stability_score <- avg_mse + 2 * st_dev 
    
    cv_results <- rbind(cv_results, data.frame(
      gamma = g, 
      avg_mse = avg_mse,
      instability = st_dev,
      score = stability_score
    ))
  }
  return(cv_results)
}

# --- THESIS SIMULATION ---
set.seed(42)
data <- generate_data(400, shift = 0)
gammas <- c(0, 1, 10, 20, 40, 80)

cat("Running Stability-Prioritized CV...\n")
results <- run_stability_cv(data$X, data$y, gamma_range = gammas)

# --- FIND THE OPTIMAL MODELS ---
best_by_accuracy  <- results$gamma[which.min(results$avg_mse)]
best_by_stability <- results$gamma[which.min(results$score)]

cat("\n--- RESULTS ---\n")
cat("Optimal Gamma (Accuracy-only): ", best_by_accuracy, "\n")
cat("Optimal Gamma (Stability-prioritized): ", best_by_stability, "\n")