# ============================================================
# SEM variations:
#   1  Chaging how much influence Y has on child: X4 = c*Y + delta + e (for c 0.1-1)
#   2  Adding noise to child variable: X4 = Y + delta + N(0,sigma) (sigma 0.1-10)
#   3  Noisy vs not child: env1 has precise child w/sigma=0.1, env2 sigma large
#   4  Noise on outcome variable change: Y = X2+X3 + N(0, sigma_env), sigma varies by env
#   5  shift on X1 and no child variables
#   6  ancestor shift + weak child
#   7  env2 is 20% of training 
#
# Metrics: adversary AUC vs true env labels, ACR % improvement over OLS
# Quick run: alpha=5, 10 seeds, 30 iterations
# ============================================================

library(ggplot2)
library(dplyr)

out_dir  <- "/Users/janamunoz/Desktop/Thesis/working folder/"
code_dir <- "/Users/janamunoz/Desktop/acr-thesis-code/"

# ── Core functions (same as all other experiments) ───────────────────────────

wls_learner <- function(X, y, w, gamma, ridge = 1e-6) {
  w  <- pmin(pmax(w, 1e-10), 1 - 1e-10)
  w1 <- w  / sum(w);  w2 <- (1 - w) / sum(1 - w)
  beta <- rep(0, ncol(X))
  for (iter in 1:3) {
    r  <- as.vector(y - X %*% beta)
    R1 <- sum(w1 * r^2);  R2 <- sum(w2 * r^2)
    s  <- sign(R1 - R2)
    v  <- pmax((1 + gamma * s) * w1 + (1 - gamma * s) * w2, 0)
    A  <- t(X * v) %*% X + ridge * diag(ncol(X))
    beta <- as.vector(solve(A, t(X * v) %*% y))
  }
  beta
}

adv_update <- function(X, y, beta) {
  r <- as.vector((y - X %*% beta)^2)
  plogis((r - mean(r)) / (sd(r) + 1e-8))
}

fit_acr_adv <- function(X, y, gamma, iterations = 30) {
  w <- rep(0.5, nrow(X)); beta <- rep(0, ncol(X))
  for (i in seq_len(iterations)) {
    beta <- wls_learner(X, y, w, gamma)
    w    <- adv_update(X, y, beta)
  }
  list(beta = beta, w = w)
}

cv_gamma <- function(X, y, gamma_grid, K = 5, seed = 1) {
  set.seed(seed)
  folds  <- sample(rep(seq_len(K), length.out = nrow(X)))
  scores <- sapply(gamma_grid, function(g) {
    mean(sapply(seq_len(K), function(k) {
      tr <- folds != k; va <- folds == k
      beta <- fit_acr_adv(X[tr,,drop=FALSE], y[tr], g)$beta
      mean((y[va] - X[va,,drop=FALSE] %*% beta)^2)
    }))
  })
  gamma_grid[which.min(scores)]
}

compute_auc <- function(w, true_env) {
  label <- as.integer(true_env == 2)
  n1 <- sum(label == 1); n0 <- sum(label == 0)
  if (n1 == 0 || n0 == 0) return(0.5)
  auc <- (sum(rank(w)[label == 1]) - n1*(n1+1)/2) / (n1*n0)
  max(auc, 1 - auc)
}

run_one <- function(X_tr, y_tr, env_tr, X_te, y_te,
                    gamma_grid = c(0.5,1,2,5,10,20), seed = 1) {
  g    <- cv_gamma(X_tr, y_tr, gamma_grid, seed = seed)
  ols  <- as.vector(coef(lm(y_tr ~ X_tr - 1)))
  fit  <- fit_acr_adv(X_tr, y_tr, g)
  list(
    auc     = compute_auc(fit$w, env_tr),
    ols_mse = mean((y_te - X_te %*% ols)^2),
    acr_mse = mean((y_te - X_te %*% fit$beta)^2),
    gamma   = g
  )
}

# ── Shared base: causal variables, no children (to be augmented) ─────────────

base_causal <- function(n, seed) {
  set.seed(seed)
  X1 <- rnorm(n); X2 <- X1 + rnorm(n); X3 <- X1 + X2 + rnorm(n)
  Y  <- X2 + X3 + rnorm(n)
  list(X1=X1, X2=X2, X3=X3, Y=Y)
}

# ── Experiment parameters ─────────────────────────────────────────────────────

alpha     <- 5        # fixed shift strength
seeds     <- 1:10
n_per_env <- 250
n_train   <- 2 * n_per_env
gamma_grid <- c(0.5, 1, 2, 5, 10, 20)

# Helper: standardise train, apply to test
std <- function(Xtr, Xte) {
  mu <- colMeans(Xtr); sg <- apply(Xtr, 2, sd)
  list(Xtr = scale(Xtr, mu, sg),
       Xte = scale(Xte, center = mu, scale = sg))
}

# ── 0: Baseline SEM-6 ───────────────────────────────────────

gen_v0 <- function(n_per_env, alpha, seed) {
  set.seed(seed); n <- 2*n_per_env
  env <- rep(c(1L,2L), each=n_per_env)
  b   <- base_causal(n, seed); Y <- b$Y
  A4  <- ifelse(env==1, rnorm(n,0,1), rnorm(n,alpha,1))
  A5  <- ifelse(env==1, rnorm(n,0,1), rnorm(n,alpha,1))
  X4  <- Y + A4 + rnorm(n,0,0.5); X5 <- Y + A5 + rnorm(n,0,0.5)
  X6  <- X5 + rnorm(n,0,0.5)
  Xr  <- cbind(b$X1,b$X2,b$X3,X4,X5,X6)
  # test: env2 at alpha_test = 1.5*alpha
  at <- 1.5*alpha; nt <- n_per_env
  set.seed(seed+1000)
  X1t <- rnorm(nt); X2t <- X1t+rnorm(nt); X3t <- X1t+X2t+rnorm(nt)
  Yt  <- X2t+X3t+rnorm(nt)
  X4t <- Yt+rnorm(nt,at,1)+rnorm(nt,0,0.5); X5t <- Yt+rnorm(nt,at,1)+rnorm(nt,0,0.5)
  X6t <- X5t+rnorm(nt,0,0.5)
  Xrt <- cbind(X1t,X2t,X3t,X4t,X5t,X6t)
  s   <- std(Xr, Xrt)
  list(X_tr=s$Xtr, y_tr=Y, env_tr=env, X_te=s$Xte, y_te=Yt)
}

# ── 1:  Chaging how much influence Y has on child ──────────────────────────
# For c < 1, OLS puts less weight on X4 → more residual left → delta creates
# asymmetric mean error in env2 that isn't fully absorbed

gen_v1 <- function(n_per_env, alpha, seed, c_coupling) {
  set.seed(seed); n <- 2*n_per_env
  env <- rep(c(1L,2L), each=n_per_env)
  b   <- base_causal(n, seed); Y <- b$Y
  A4  <- ifelse(env==1, rnorm(n,0,1), rnorm(n,alpha,1))
  X4  <- c_coupling*Y + A4 + rnorm(n,0,0.5)   # only ONE child, coupling c
  Xr  <- cbind(b$X1, b$X2, b$X3, X4)
  # test
  at <- 1.5*alpha; nt <- n_per_env
  set.seed(seed+1000)
  X1t <- rnorm(nt); X2t <- X1t+rnorm(nt); X3t <- X1t+X2t+rnorm(nt)
  Yt  <- X2t+X3t+rnorm(nt)
  X4t <- c_coupling*Yt+rnorm(nt,at,1)+rnorm(nt,0,0.5)
  Xrt <- cbind(X1t,X2t,X3t,X4t)
  s   <- std(Xr, Xrt)
  list(X_tr=s$Xtr, y_tr=Y, env_tr=env, X_te=s$Xte, y_te=Yt)
}

# ── 2:  Adding noise to child variable ─────────────────────
# Large sigma → OLS uses X4 less → shift in X4 has less effect on predictions
# → env2 residual mean shift smaller → adversary might still struggle

gen_v2 <- function(n_per_env, alpha, seed, sigma_child) {
  set.seed(seed); n <- 2*n_per_env
  env <- rep(c(1L,2L), each=n_per_env)
  b   <- base_causal(n, seed); Y <- b$Y
  A4  <- ifelse(env==1, rnorm(n,0,1), rnorm(n,alpha,1))
  X4  <- Y + A4 + rnorm(n, 0, sigma_child)
  Xr  <- cbind(b$X1, b$X2, b$X3, X4)
  at <- 1.5*alpha; nt <- n_per_env
  set.seed(seed+1000)
  X1t <- rnorm(nt); X2t <- X1t+rnorm(nt); X3t <- X1t+X2t+rnorm(nt)
  Yt  <- X2t+X3t+rnorm(nt)
  X4t <- Yt+rnorm(nt,at,1)+rnorm(nt,0,sigma_child)
  Xrt <- cbind(X1t,X2t,X3t,X4t)
  s   <- std(Xr, Xrt)
  list(X_tr=s$Xtr, y_tr=Y, env_tr=env, X_te=s$Xte, y_te=Yt)
}

# ── 3: Noisy vs not child ─────────────────────────────────────────────────
# env1: X4 = Y + N(0, sigma_small) → OLS exploits X4 well → tiny env1 residuals
# env2: X4 = Y + alpha + N(0, sigma_large) → OLS can't exploit X4 → large env2 residuals
# This breaks the OLS symmetry: env2 residuals are larger due to noisy child

gen_v3 <- function(n_per_env, alpha, seed, sigma_small, sigma_large) {
  set.seed(seed); n <- 2*n_per_env
  env <- rep(c(1L,2L), each=n_per_env)
  b   <- base_causal(n, seed); Y <- b$Y
  X4  <- ifelse(env==1,
                Y + rnorm(n, 0, sigma_small),
                Y + alpha + rnorm(n, 0, sigma_large))
  Xr  <- cbind(b$X1, b$X2, b$X3, X4)
  # test: env2 at alpha_test
  at <- 1.5*alpha; nt <- n_per_env
  set.seed(seed+1000)
  X1t <- rnorm(nt); X2t <- X1t+rnorm(nt); X3t <- X1t+X2t+rnorm(nt)
  Yt  <- X2t+X3t+rnorm(nt)
  X4t <- Yt + at + rnorm(nt, 0, sigma_large)  # test is env2-like
  Xrt <- cbind(X1t,X2t,X3t,X4t)
  s   <- std(Xr, Xrt)
  list(X_tr=s$Xtr, y_tr=Y, env_tr=env, X_te=s$Xte, y_te=Yt)
}

# ── 4: Noise on outcome variable change ───────────────────────────────────────────────────
# Y = X2+X3+N(0,1) in env1, Y = X2+X3+N(0,sigma2) in env2 --> gets higher noise
# ACR might not help (noise is irreducible) but adversary should work

gen_v4 <- function(n_per_env, alpha = NULL, alpha_noise, seed) {
  set.seed(seed); n <- 2*n_per_env
  env <- rep(c(1L,2L), each=n_per_env)
  b   <- base_causal(n, seed)
  # override Y with heteroscedastic noise
  Y   <- b$X2 + b$X3 + ifelse(env==1, rnorm(n,0,1), rnorm(n,0,alpha_noise))
  Xr  <- cbind(b$X1, b$X2, b$X3)   # no children; shift is in Y's noise
  # test: env2-like noise
  nt <- n_per_env; set.seed(seed+1000)
  X1t <- rnorm(nt); X2t <- X1t+rnorm(nt); X3t <- X1t+X2t+rnorm(nt)
  Yt  <- X2t + X3t + rnorm(nt, 0, alpha_noise)
  Xrt <- cbind(X1t, X2t, X3t)
  s   <- std(Xr, Xrt)
  list(X_tr=s$Xtr, y_tr=Y, env_tr=env, X_te=s$Xte, y_te=Yt)
}

# ── 5: shift on X1 and no child variables ─────────────────────────────────────────
# env2: X1 ~ N(alpha, 1) instead of N(0,1)
# Causal chain propagates shift: X2, X3, Y all higher in env2
# OLS should still fit well (correct causal structure), residuals similar

gen_v5 <- function(n_per_env, alpha, seed) {
  set.seed(seed); n <- 2*n_per_env
  env <- rep(c(1L,2L), each=n_per_env)
  X1  <- ifelse(env==1, rnorm(n,0,1), rnorm(n,alpha,1))
  X2  <- X1 + rnorm(n); X3 <- X1 + X2 + rnorm(n)
  Y   <- X2 + X3 + rnorm(n)
  Xr  <- cbind(X1, X2, X3)
  nt  <- n_per_env; at <- 1.5*alpha; set.seed(seed+1000)
  X1t <- rnorm(nt, at, 1); X2t <- X1t+rnorm(nt); X3t <- X1t+X2t+rnorm(nt)
  Yt  <- X2t+X3t+rnorm(nt)
  Xrt <- cbind(X1t, X2t, X3t)
  s   <- std(Xr, Xrt)
  list(X_tr=s$Xtr, y_tr=Y, env_tr=env, X_te=s$Xte, y_te=Yt)
}

# ── 6: ancestor shift + weak child ────────────────────────────────────────
# Combines V5 ancestor shift with a weak spurious child (c=0.3)
# Tests whether two simultaneous signals (ancestor shift + child) help adversary

gen_v6 <- function(n_per_env, alpha, seed, c_coupling = 0.3) {
  set.seed(seed); n <- 2*n_per_env
  env <- rep(c(1L,2L), each=n_per_env)
  X1  <- ifelse(env==1, rnorm(n,0,1), rnorm(n,alpha,1))
  X2  <- X1 + rnorm(n); X3 <- X1+X2+rnorm(n)
  Y   <- X2 + X3 + rnorm(n)
  X4  <- c_coupling*Y + ifelse(env==1, rnorm(n,0,1), rnorm(n,alpha,1)) + rnorm(n,0,0.5)
  Xr  <- cbind(X1, X2, X3, X4)
  nt  <- n_per_env; at <- 1.5*alpha; set.seed(seed+1000)
  X1t <- rnorm(nt,at,1); X2t <- X1t+rnorm(nt); X3t <- X1t+X2t+rnorm(nt)
  Yt  <- X2t+X3t+rnorm(nt)
  X4t <- c_coupling*Yt+rnorm(nt,at,1)+rnorm(nt,0,0.5)
  Xrt <- cbind(X1t, X2t, X3t, X4t)
  s   <- std(Xr, Xrt)
  list(X_tr=s$Xtr, y_tr=Y, env_tr=env, X_te=s$Xte, y_te=Yt)
}

# ── 7: env2 is 20% of training ──────────────────────────────────────────────────
# env1: 80% of training (n=400), env2: 20% (n=100)
# OLS dominated by env1 → env2 observations have larger residuals (leverage effect)
# Child structure same as SEM-6

gen_v7 <- function(n_per_env, alpha, seed, frac_env2 = 0.2) {
  set.seed(seed)
  n_total <- 2*n_per_env
  n2 <- round(n_total * frac_env2); n1 <- n_total - n2
  env <- c(rep(1L, n1), rep(2L, n2)); n <- n_total
  X1 <- rnorm(n); X2 <- X1+rnorm(n); X3 <- X1+X2+rnorm(n); Y <- X2+X3+rnorm(n)
  A4 <- c(rnorm(n1,0,1), rnorm(n2,alpha,1))
  A5 <- c(rnorm(n1,0,1), rnorm(n2,alpha,1))
  X4 <- Y+A4+rnorm(n,0,0.5); X5 <- Y+A5+rnorm(n,0,0.5); X6 <- X5+rnorm(n,0,0.5)
  Xr <- cbind(X1,X2,X3,X4,X5,X6)
  nt <- n_per_env; at <- 1.5*alpha; set.seed(seed+1000)
  X1t <- rnorm(nt); X2t <- X1t+rnorm(nt); X3t <- X1t+X2t+rnorm(nt); Yt <- X2t+X3t+rnorm(nt)
  X4t <- Yt+rnorm(nt,at,1)+rnorm(nt,0,0.5); X5t <- Yt+rnorm(nt,at,1)+rnorm(nt,0,0.5)
  X6t <- X5t+rnorm(nt,0,0.5)
  Xrt <- cbind(X1t,X2t,X3t,X4t,X5t,X6t)
  s   <- std(Xr, Xrt)
  list(X_tr=s$Xtr, y_tr=Y, env_tr=env, X_te=s$Xte, y_te=Yt)
}

# ── Run all experiments ───────────────────────────────────────────────────────

results <- list()

run_batch <- function(label, param_name, param_vals, gen_fn, extra_args = list()) {
  cat(sprintf("\n--- %s ---\n", label))
  for (pv in param_vals) {
    aucs <- mses_ols <- mses_acr <- numeric(length(seeds))
    for (si in seq_along(seeds)) {
      args <- c(list(n_per_env=n_per_env, alpha=alpha, seed=seeds[si]), extra_args)
      args[[param_name]] <- pv
      dat  <- do.call(gen_fn, args)
      res  <- run_one(dat$X_tr, dat$y_tr, dat$env_tr, dat$X_te, dat$y_te, seed=seeds[si])
      aucs[si]     <- res$auc
      mses_ols[si] <- res$ols_mse
      mses_acr[si] <- res$acr_mse
    }
    pct <- (mean(mses_acr) - mean(mses_ols)) / mean(mses_ols) * 100
    results[[length(results)+1]] <<- data.frame(
      variant = label, param = param_name,
      param_val = as.character(pv),
      auc      = mean(aucs),
      auc_sd   = sd(aucs),
      ols_mse  = mean(mses_ols),
      acr_mse  = mean(mses_acr),
      pct_vs_ols = pct,
      stringsAsFactors = FALSE)
    cat(sprintf("  %s=%-5s  AUC=%.3f±%.3f  ACR%%OLS=%+.1f%%\n",
                param_name, pv, mean(aucs), sd(aucs)/sqrt(length(seeds)), pct))
  }
}

# 0
cat("\n--- 0: Baseline SEM-6 ---\n")
aucs <- mses_ols <- mses_acr <- numeric(length(seeds))
for (si in seq_along(seeds)) {
  dat <- gen_v0(n_per_env, alpha, seeds[si])
  res <- run_one(dat$X_tr, dat$y_tr, dat$env_tr, dat$X_te, dat$y_te, seed=seeds[si])
  aucs[si] <- res$auc; mses_ols[si] <- res$ols_mse; mses_acr[si] <- res$acr_mse
}
pct <- (mean(mses_acr)-mean(mses_ols))/mean(mses_ols)*100
results[[1]] <- data.frame(variant="0: SEM-6 (baseline)", param="none",
  param_val="baseline", auc=mean(aucs), auc_sd=sd(aucs),
  ols_mse=mean(mses_ols), acr_mse=mean(mses_acr), pct_vs_ols=pct)
cat(sprintf("  AUC=%.3f±%.3f  ACR%%OLS=%+.1f%%\n",
            mean(aucs), sd(aucs)/sqrt(length(seeds)), pct))

# 1
run_batch("1: child's connection to Y sweep", "c_coupling",
          c(0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0), gen_v1)

# 2
run_batch("2: child with varying noise", "sigma_child",
          c(0.1, 0.3, 0.5, 1, 2, 5, 10), gen_v2)

# 3
cat("\n--- V3: Heteroscedastic child (env1 sigma=0.1, env2 sigma varies) ---\n")
for (sig_large in c(0.5, 1, 2, 5, 10)) {
  aucs <- mses_ols <- mses_acr <- numeric(length(seeds))
  for (si in seq_along(seeds)) {
    dat <- gen_v3(n_per_env, alpha, seeds[si], sigma_small=0.1, sigma_large=sig_large)
    res <- run_one(dat$X_tr, dat$y_tr, dat$env_tr, dat$X_te, dat$y_te, seed=seeds[si])
    aucs[si] <- res$auc; mses_ols[si] <- res$ols_mse; mses_acr[si] <- res$acr_mse
  }
  pct <- (mean(mses_acr)-mean(mses_ols))/mean(mses_ols)*100
  results[[length(results)+1]] <- data.frame(
    variant="V3: Heteroscedastic child", param="sigma_env2",
    param_val=as.character(sig_large), auc=mean(aucs), auc_sd=sd(aucs),
    ols_mse=mean(mses_ols), acr_mse=mean(mses_acr), pct_vs_ols=pct)
  cat(sprintf("  sigma_env2=%-5s  AUC=%.3f±%.3f  ACR%%OLS=%+.1f%%\n",
              sig_large, mean(aucs), sd(aucs)/sqrt(length(seeds)), pct))
}
#4
run_batch("4 shift on Y", "alpha_noise",
          c(1.5, 2, 3, 5, 10), gen_v4)
#5
cat("\n--- 5 shift on X1 and no children ---\n")
for (al in c(1, 3, 5, 10)) {
  aucs <- mses_ols <- mses_acr <- numeric(length(seeds))
  for (si in seq_along(seeds)) {
    dat <- gen_v5(n_per_env, al, seeds[si])
    res <- run_one(dat$X_tr, dat$y_tr, dat$env_tr, dat$X_te, dat$y_te, seed=seeds[si])
    aucs[si] <- res$auc; mses_ols[si] <- res$ols_mse; mses_acr[si] <- res$acr_mse
  }
  pct <- (mean(mses_acr)-mean(mses_ols))/mean(mses_ols)*100
  results[[length(results)+1]] <- data.frame(
    variant="5 shift on x1 and no children", param="alpha",
    param_val=as.character(al), auc=mean(aucs), auc_sd=sd(aucs),
    ols_mse=mean(mses_ols), acr_mse=mean(mses_acr), pct_vs_ols=pct)
  cat(sprintf("  alpha=%-5s  AUC=%.3f±%.3f  ACR%%OLS=%+.1f%%\n",
              al, mean(aucs), sd(aucs)/sqrt(length(seeds)), pct))
}
#6
run_batch("6 Ancestor shift + weak child", "c_coupling",
          c(0.1, 0.3, 0.5, 0.7, 1.0), gen_v6)
#7
cat("\n--- 7 env2 is 20% of training  ---\n")
for (frac in c(0.1, 0.2, 0.3, 0.4, 0.5)) {
  aucs <- mses_ols <- mses_acr <- numeric(length(seeds))
  for (si in seq_along(seeds)) {
    dat <- gen_v7(n_per_env, alpha, seeds[si], frac_env2=frac)
    res <- run_one(dat$X_tr, dat$y_tr, dat$env_tr, dat$X_te, dat$y_te, seed=seeds[si])
    aucs[si] <- res$auc; mses_ols[si] <- res$ols_mse; mses_acr[si] <- res$acr_mse
  }
  pct <- (mean(mses_acr)-mean(mses_ols))/mean(mses_ols)*100
  results[[length(results)+1]] <- data.frame(
    variant="7: Minority env2", param="frac_env2",
    param_val=as.character(frac), auc=mean(aucs), auc_sd=sd(aucs),
    ols_mse=mean(mses_ols), acr_mse=mean(mses_acr), pct_vs_ols=pct)
  cat(sprintf("  frac_env2=%-5s  AUC=%.3f±%.3f  ACR%%OLS=%+.1f%%\n",
              frac, mean(aucs), sd(aucs)/sqrt(length(seeds)), pct))
}

# ── Summary ───────────────────────────────────────────────────────────────────

res_df <- do.call(rbind, results)

cat("\n\n", strrep("=", 70), "\n")
cat("SUMMARY: Conditions ranked by AUC\n")
cat(strrep("=", 70), "\n")
res_sorted <- res_df[order(-res_df$auc), ]
cat(sprintf("%-40s %8s %8s %10s\n", "Variant / param_val", "AUC", "AUC_se", "ACR%OLS"))
for (i in seq_len(min(20, nrow(res_sorted)))) {
  r <- res_sorted[i,]
  cat(sprintf("%-40s %8.3f %8.3f %+10.1f%%\n",
              paste0(substr(r$variant,1,30), " [",r$param_val,"]"),
              r$auc, r$auc_sd/sqrt(10), r$pct_vs_ols))
}

cat("\n\nTop results where AUC > 0.65:\n")
top <- subset(res_df, auc > 0.65)
if (nrow(top) == 0) {
  cat("  None found above 0.65\n")
  cat("  Top 5 results:\n")
  top <- head(res_df[order(-res_df$auc),], 5)
  for (i in seq_len(nrow(top))) {
    r <- top[i,]
    cat(sprintf("  %-38s  AUC=%.3f  ACR%%OLS=%+.1f%%\n",
                paste0(r$variant," [",r$param_val,"]"), r$auc, r$pct_vs_ols))
  }
} else {
  for (i in seq_len(nrow(top))) {
    r <- top[i,]
    cat(sprintf("  %-38s  AUC=%.3f  ACR%%OLS=%+.1f%%\n",
                paste0(r$variant," [",r$param_val,"]"), r$auc, r$pct_vs_ols))
  }
}

# ── Plot ──────────────────────────────────────────────────────────────────────

p <- ggplot(res_df, aes(x = auc, y = pct_vs_ols,
                         colour = variant, label = param_val)) +
  geom_hline(yintercept = 0, linetype = "dashed", colour = "grey60") +
  geom_vline(xintercept = 0.5, linetype = "dashed", colour = "grey60") +
  geom_point(size = 3, alpha = 0.8) +
  geom_text(nudge_y = 0.5, size = 2.8, show.legend = FALSE) +
  annotate("rect", xmin = 0.65, xmax = Inf, ymin = -Inf, ymax = 0,
           fill = "green", alpha = 0.08) +
  annotate("text", x = 0.8, y = -8, label = "ACR wins + adversary works",
           colour = "darkgreen", size = 3) +
  labs(x = "Adversary AUC (higher = better environment discovery)",
       y = "ACR % improvement over OLS (negative = ACR wins)",
       title = "SEM design exploration: when does the adversary work?",
       subtitle = "Green region = adversary succeeds AND ACR helps",
       colour = NULL) +
  theme_bw(base_size = 11) +
  theme(legend.position = "bottom",
        legend.text = element_text(size = 8))

ggsave(paste0(code_dir, "12-sem_exploration.pdf"), p, width=11, height=7)
ggsave(paste0(code_dir, "12-sem_exploration.png"), p, width=11, height=7, dpi=160)

save(res_df, file=paste0(code_dir, "12-sem_exploration.RData"))
cat("\nPlot saved to", code_dir, "\n")
