#!/usr/bin/env Rscript

# Load necessary library for the Adversarial function we built
source("2-acr_adversarial.r")

# Configuration
shifts <- seq(0, 50, by = 5)
results <- data.frame()

# 1. Train models on data with NO shift (Shift = 0)
set.seed(42)
train_data <- generate_data(1000, shift = 0)
ols_model <- lm(train_data$y ~ train_data$X - 1)
beta_ols <- coef(ols_model)
beta_acr <- run_acr_adversarial(train_data$X, train_data$y, gamma = 20)

# 2. Iterate through increasing OOD shifts
cat("Evaluating models across shifts...\n")
for (s in shifts) {
  test_data <- generate_data(500, shift = s)
  
  # Predict and Calculate MSE
  mse_ols <- mean((test_data$y - test_data$X %*% beta_ols)^2)
  mse_acr <- mean((test_data$y - test_data$X %*% beta_acr)^2)
  
  results <- rbind(results, data.frame(shift = s, mse_ols = mse_ols, mse_acr = mse_acr))
}

# 3. Save Results
write.csv(results, "3-shift_sensitivity_results.csv", row.names = FALSE)

# 4. Plotting the 'Error Explosion'
pdf("3-shift_analysis_plot.pdf")
plot(results$shift, results$mse_ols, type="b", col="red", pch=19, log="y",
     xlab="Distribution Shift (A)", ylab="MSE (Log Scale)",
     main="OLS Failure vs. ACR Stability")
lines(results$shift, results$mse_acr, type="b", col="green", pch=15)
legend("topleft", legend=c("Standard OLS", "Adversarial Causal Reg"), 
       col=c("red", "green"), pch=c(19, 15), lty=1)
dev.off()

cat("Analysis complete. Check '3-shift_sensitivity_results.csv' and the plot.\n")