#!/usr/bin/env Rscript

# 1. Soft Adversary ACR Function
# This uses a sigmoid-based weighting to find a "soft" split
fit_acr_soft <- function(X, y, gamma, iterations = 15) {
  n <- nrow(X)
  p <- ncol(X)
  beta <- rep(0, p)
  w <- rep(0.5, n) # Initial neutral weights
  
  for (i in 1:iterations) {
    # --- LEARNER STEP ---
    # Minimize: Weighted_Risk + Gamma * |Risk_Diff|
    learner_obj <- function(b) {
      err <- (y - X %*% b)^2
      r1 <- sum(w * err) / sum(w)
      r2 <- sum((1 - w) * err) / sum(1 - w)
      return((r1 + r2) + gamma * abs(r1 - r2))
    }
    beta <- optim(par = beta, fn = learner_obj, method = "BFGS")$par
    
    # --- ADVERSARY STEP ---
    # Maximize risk difference by shifting weights toward high-error points
    errors <- (y - X %*% beta)^2
    # Softmax-style update: weights become a function of the error magnitude
    w <- 1 / (1 + exp(-(errors - mean(errors)) / sd(errors)))
  }
  return(beta)
}

# 2. Cross-Validation Loop
run_acr_cv <- function(X, y, gamma_range, k = 5) {
  n <- nrow(X)
  folds <- sample(rep(1:k, length.out = n))
  cv_results <- data.frame()
  
  for (g in gamma_range) {
    fold_errors <- c()
    
    for (f in 1:k) {
      # Split into CV Train and CV Validation
      X_train <- X[folds != f, ]
      y_train <- y[folds != f]
      X_val   <- X[folds == f, ]
      y_val   <- y[folds == f]
      
      # Train with ACR
      beta_fold <- fit_acr_soft(X_train, y_train, gamma = g)
      
      # Evaluate on Validation fold
      val_mse <- mean((y_val - X_val %*% beta_fold)^2)
      fold_errors <- c(fold_errors, val_mse)
    }
    
    cv_results <- rbind(cv_results, data.frame(
      gamma = g, 
      avg_mse = mean(fold_errors),
      se_mse = sd(fold_errors) / sqrt(k)
    ))
  }
  return(cv_results)
}

# --- THESIS EXPERIMENT ---
set.seed(101)
# Generate a "small dataset" (e.g., n=300) to test stability
source("2-acr_adversarial.r") # Reuse your previous data generator
data <- generate_data(300, shift = 0)
gammas <- c(0, 1, 5, 10, 20, 50)

cat("Starting Adversarial Cross-Validation...\n")
cv_out <- run_acr_cv(data$X, data$y, gamma_range = gammas)

# --- VISUALIZATION ---
# Plot the CV Error Curve to find the "Causal Elbow"
pdf("4-acr_cv_plot.pdf")
plot(cv_out$gamma, cv_out$avg_mse, type="b", pch=19, col="blue",
     ylim = c(min(cv_out$avg_mse - cv_out$se_mse), max(cv_out$avg_mse + cv_out$se_mse)),
     xlab = "Regularization Parameter (Gamma)", ylab = "CV Mean Squared Error",
     main = "Adversarial CV for Optimal Gamma Selection")
arrows(cv_out$gamma, cv_out$avg_mse - cv_out$se_mse, 
       cv_out$gamma, cv_out$avg_mse + cv_out$se_mse, 
       code=3, angle=90, length=0.05, col="gray")
dev.off()

best_gamma <- cv_out$gamma[which.min(cv_out$avg_mse)]
cat("Optimal Gamma selected by CV:", best_gamma, "\n")

