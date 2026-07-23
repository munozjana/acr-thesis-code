#!/usr/bin/env Rscript

if (!require("optparse")) install.packages("optparse", repos="http://cran.us.r-project.org")
library(optparse)

# Command Line Arguments defined
option_list = list(
  make_option(c("-s", "--shiftTarget"), type="double", default=4.0, 
              help="Strength of the distribution shift in the test set", metavar="double"),
  make_option(c("-i", "--inSampleShift"), type="double", default=0.0, 
              help="Shift applied to the training data", metavar="double"),
  make_option(c("-n", "--n_samples"), type="integer", default=500, 
              help="Number of samples to generate", metavar="integer")
)

opt_parser = OptionParser(option_list=option_list)
opt = parse_args(opt_parser)

# SEM
generate_sem_data <- function(n, shift_strength = 0) {
  eps <- matrix(rnorm(n * 7), ncol = 7)
  x1 <- eps[, 1]
  x2 <- x1 + eps[, 2]
  x3 <- x1 + x2 + eps[, 3]
  y  <- x2 + x3 + eps[, 4] # Causal parents: X2, X3
  x4 <- y + shift_strength + eps[, 5] # Unstable child
  x5 <- y + shift_strength + eps[, 6] # Unstable child
  x6 <- x5 + eps[, 7]
  
  X <- cbind(x1, x2, x3, x4, x5, x6)
  colnames(X) <- paste0("X", 1:6)
  return(list(X = X, y = y))
}

cat("Running simulation with inSampleShift =", opt$inSampleShift, 
    "and shiftTarget =", opt$shiftTarget, "\n\n")

# Generate Environments
env_train <- generate_sem_data(opt$n_samples, shift_strength = opt$inSampleShift)
env_test  <- generate_sem_data(opt$n_samples, shift_strength = opt$shiftTarget)

# Fit a simple OLS for comparison
ols_model <- lm(env_train$y ~ env_train$X - 1)
preds <- env_test$X %*% coef(ols_model)
mse <- mean((env_test$y - preds)^2)

cat("--- RESULTS ---\n")
cat("OLS MSE on Test Set: ", mse, "\n")
cat("Coefficients used: \n")
print(round(coef(ols_model), 3))
