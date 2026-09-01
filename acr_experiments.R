# ============================================================
# Systematic ACR Experiment
# Adversarial vs Random Split across:
#   - Synthetic SEM (known ground truth, two shift types)
#   - Airquality (every holdout month)
#   - Multiple seeds, multiple shift strengths
#
# Real data: ChickWeight, Boston housing, CO2 uptake, Colored MNIST
#
# Methods compared:
#   OLS         : pooled least squares (gamma = 0)
#   ACR-Adv     : adversarial soft-weight split (our method)
#   CR-Rand     : causal regularisation with fixed random split
#   Oracle-CR   : causal regularisation with true env labels (SEM only)
#
# Metrics:
#   OOD MSE     : mean squared error on held-out test set
#   AUC         : environment discovery AUC vs true labels (SEM only)
#   Coef error  : ||beta_hat / ||beta_hat|| - beta* / ||beta*||||  (SEM only)
# ============================================================

out_dir <- ""

# Closed-form weighted least squares learner
wls_learner <- function(X, y, w, gamma, ridge = 1e-6) {
  n <- nrow(X)
  w  <- pmin(pmax(w, 1e-10), 1 - 1e-10)
  sw  <- sum(w);   snw <- sum(1 - w)
  w1  <- w  / sw            # normalised env-1 weights
  w2  <- (1 - w) / snw      # normalised env-2 weights

  r   <- as.vector(y - X %*% rep(0, ncol(X)))  # init residuals at 0
  # sign of risk gap (we recompute after each beta update; start positive)
  # Use combined weight: v_i = (1+gamma*s)*w1_i + (1-gamma*s)*w2_i
  # We iterate once (one WLS step per adversary step is enough)
  beta <- rep(0, ncol(X))
  for (iter in 1:3) {
    r  <- as.vector(y - X %*% beta)
    R1 <- sum(w1 * r^2);  R2 <- sum(w2 * r^2)
    s  <- sign(R1 - R2)
    v  <- (1 + gamma * s) * w1 + (1 - gamma * s) * w2
    v  <- pmax(v, 0)                        # no negative weights in WLS
    XtV <- t(X * v)                         
    A   <- XtV %*% X + ridge * diag(ncol(X))
    b   <- XtV %*% y
    beta <- as.vector(solve(A, b))
  }
  beta
}

# Adversary: soft sigmoid update
adv_update <- function(X, y, beta) {
  r <- as.vector((y - X %*% beta)^2)
  plogis((r - mean(r)) / (sd(r) + 1e-8))
}

# Full ACR alternating loop (adversarial split)
fit_acr_adv <- function(X, y, gamma, iterations = 30) {
  w    <- rep(0.5, nrow(X))
  beta <- rep(0, ncol(X))
  for (i in seq_len(iterations)) {
    beta <- wls_learner(X, y, w, gamma)
    w    <- adv_update(X, y, beta)
  }
  list(beta = beta, w = w)
}

# CR with fixed split (random or oracle)
fit_cr_fixed <- function(X, y, gamma, w_fixed) {
  # w_fixed: binary or soft weights (fixed, not updated)
  wls_learner(X, y, w_fixed, gamma)
}

# Cross-validate gamma on training set
cv_gamma <- function(X, y, gamma_grid, K = 5, method = "adv", w_oracle = NULL,
                     seed = 1, iterations = 30) {
  set.seed(seed)
  n     <- nrow(X)
  if (is.null(groups)) {
    folds <- sample(rep(seq_len(K), length.out = n))
  } else {
    # grouped CV: all rows of a group land in the same fold (repeated measures)
    g      <- as.character(groups)
    ug     <- unique(g)
    gfold  <- sample(rep(seq_len(K), length.out = length(ug)))
    folds  <- gfold[match(g, ug)]
  }
  scores <- sapply(gamma_grid, function(g) {
    fold_mse <- sapply(seq_len(K), function(k) {
      tr <- folds != k;  va <- folds == k
      Xtr <- X[tr, , drop = FALSE];  ytr <- y[tr]
      Xva <- X[va, , drop = FALSE];  yva <- y[va]
      if (method == "adv") {
        beta <- fit_acr_adv(Xtr, ytr, g, iterations)$beta
      } else if (method == "rand") {
        set.seed(seed + k)
        wf <- as.numeric(sample(c(0, 1), sum(tr), replace = TRUE))
        beta <- fit_cr_fixed(Xtr, ytr, g, wf)
      } else if (method == "oracle" && !is.null(w_oracle)) {
        beta <- fit_cr_fixed(Xtr, ytr, g, w_oracle[tr])
      } else {
        beta <- fit_cr_fixed(Xtr, ytr, 0, rep(0.5, sum(tr)))
      }
      mean((yva - Xva %*% beta)^2)
    })
    mean(fold_mse)
  })
  gamma_grid[which.min(scores)]
}

# AUC between soft weights and true binary env labels
compute_auc <- function(w, true_env) {
  label <- as.integer(true_env == 2)  # env2 = 1, env1 = 0
  n1 <- sum(label == 1);  n0 <- sum(label == 0)
  if (n1 == 0 || n0 == 0) return(0.5)
  auc <- (sum(rank(w)[label == 1]) - n1 * (n1 + 1) / 2) / (n1 * n0)
  max(auc, 1 - auc)   # always report >= 0.5
}

# Normalised coefficient error vs true beta*
coef_error <- function(beta_hat, beta_true) {
  bh <- beta_hat  / (sqrt(sum(beta_hat^2))  + 1e-10)
  bt <- beta_true / (sqrt(sum(beta_true^2)) + 1e-10)
  sqrt(sum((bh - bt)^2))
}

# ============================================================
# 1. SEM DATASET
# ============================================================
# Structure (6 variables):
#   X1          = e1                    (exogenous)
#   X2          = X1 + e2               (causal parent of Y)
#   X3          = X1 + X2 + e3          (causal parent of Y)
#   Y           = X2 + X3 + eY          (response; beta* = (0,1,1,0,0,0))
#   X4          = Y + A4^(e) + e4        (child, shifted)
#   X5          = Y + A5^(e) + e5        (child, shifted)
#   X6          = X5 + e6               (grandchild)
#
# Shift:  mean   -> A^(2) ~ N(alpha, 1)
#         variance -> A^(2) ~ N(0, alpha)
#
# Train: env1 (n=250) + env2-train (n=250) = 500 observations
# Test:  env2 at test-shift alpha_test = 1.5 * alpha  (harder)

gen_sem <- function(n_per_env, alpha_train, alpha_test,
                    shift_type = "mean", seed = 1) {
  set.seed(seed)
  n_train <- 2 * n_per_env

  # --- training data ---
  env_tr <- rep(c(1L, 2L), each = n_per_env)
  X1 <- rnorm(n_train);  X2 <- X1 + rnorm(n_train);  X3 <- X1 + X2 + rnorm(n_train)
  Y  <- X2 + X3 + rnorm(n_train)

  if (shift_type == "mean") {
    A4 <- ifelse(env_tr == 1, rnorm(n_train, 0, 1), rnorm(n_train, alpha_train, 1))
    A5 <- ifelse(env_tr == 1, rnorm(n_train, 0, 1), rnorm(n_train, alpha_train, 1))
  } else {
    sd2 <- max(alpha_train, 0.1)
    A4  <- ifelse(env_tr == 1, rnorm(n_train, 0, 1), rnorm(n_train, 0, sd2))
    A5  <- ifelse(env_tr == 1, rnorm(n_train, 0, 1), rnorm(n_train, 0, sd2))
  }
  X4 <- Y + A4 + rnorm(n_train, 0, 0.5)
  X5 <- Y + A5 + rnorm(n_train, 0, 0.5)
  X6 <- X5 + rnorm(n_train, 0, 0.5)

  X_tr_raw <- cbind(X1, X2, X3, X4, X5, X6)

  # --- test data (stronger shift, env2 only) ---
  n_test <- n_per_env
  X1t <- rnorm(n_test);  X2t <- X1t + rnorm(n_test);  X3t <- X1t + X2t + rnorm(n_test)
  Yt  <- X2t + X3t + rnorm(n_test)

  if (shift_type == "mean") {
    A4t <- rnorm(n_test, alpha_test, 1);  A5t <- rnorm(n_test, alpha_test, 1)
  } else {
    sd2t <- max(alpha_test, 0.1)
    A4t  <- rnorm(n_test, 0, sd2t);  A5t <- rnorm(n_test, 0, sd2t)
  }
  X4t <- Yt + A4t + rnorm(n_test, 0, 0.5)
  X5t <- Yt + A5t + rnorm(n_test, 0, 0.5)
  X6t <- X5t + rnorm(n_test, 0, 0.5)
  X_te_raw <- cbind(X1t, X2t, X3t, X4t, X5t, X6t)

  # standardise using training means/sds
  mu  <- colMeans(X_tr_raw);  sg <- apply(X_tr_raw, 2, sd)
  X_tr <- scale(X_tr_raw, center = mu, scale = sg)
  X_te <- scale(X_te_raw, center = mu, scale = sg)

  list(X_tr = X_tr, y_tr = Y, env_tr = env_tr,
       X_te = X_te, y_te = Yt)
}

# Parameters
alpha_grid_sem  <- c(1, 3, 5, 10)
gamma_grid      <- c(0.5, 1, 2, 5, 10, 20)
seeds           <- 1:10
shift_types     <- c("mean", "variance")
n_per_env       <- 250
beta_true       <- c(0, 1, 1, 0, 0, 0)

# Results storage
sem_rows <- expand.grid(alpha = alpha_grid_sem, seed = seeds,
                        shift_type = shift_types, stringsAsFactors = FALSE)
metrics  <- c("ols_mse", "acr_adv_mse", "cr_rand_mse", "oracle_cr_mse",
              "acr_adv_auc", "cr_rand_auc", "oracle_auc",
              "ols_coef_err", "acr_adv_coef_err", "cr_rand_coef_err",
              "oracle_coef_err",
              "acr_adv_gamma", "cr_rand_gamma", "oracle_gamma")
sem_results <- cbind(sem_rows,
                     as.data.frame(matrix(NA_real_, nrow(sem_rows), length(metrics),
                                          dimnames = list(NULL, metrics))))

cat("=== Running SEM experiments (",nrow(sem_rows)," scenarios) ===\n",sep="")
pb <- txtProgressBar(min=0, max=nrow(sem_rows), style=3)

for (i in seq_len(nrow(sem_rows))) {
  al  <- sem_results$alpha[i]
  sid <- sem_results$seed[i]
  sht <- sem_results$shift_type[i]

  dat <- gen_sem(n_per_env, alpha_train = al, alpha_test = al * 1.5,
                 shift_type = sht, seed = sid)
  Xtr <- dat$X_tr;  ytr <- dat$y_tr;  env_tr <- dat$env_tr
  Xte <- dat$X_te;  yte <- dat$y_te

  # Oracle weights: env1 -> w=1, env2 -> w=0
  w_oracle <- as.numeric(env_tr == 1)
  # Random weights: 50/50 random binary
  set.seed(sid * 100)
  w_rand   <- as.numeric(sample(c(0,1), nrow(Xtr), replace = TRUE))

  # CV for each method
  g_adv    <- cv_gamma(Xtr, ytr, gamma_grid, method = "adv",    seed = sid)
  g_rand   <- cv_gamma(Xtr, ytr, gamma_grid, method = "rand",   seed = sid)
  g_oracle <- cv_gamma(Xtr, ytr, gamma_grid, method = "oracle",
                       w_oracle = w_oracle, seed = sid)

  # Fit models
  ols_beta    <- as.vector(coef(lm(ytr ~ Xtr - 1)))
  fit_adv     <- fit_acr_adv(Xtr, ytr, g_adv)
  beta_rand   <- fit_cr_fixed(Xtr, ytr, g_rand, w_rand)
  beta_oracle <- fit_cr_fixed(Xtr, ytr, g_oracle, w_oracle)

  # OOD MSE
  sem_results$ols_mse[i]       <- mean((yte - Xte %*% ols_beta)^2)
  sem_results$acr_adv_mse[i]  <- mean((yte - Xte %*% fit_adv$beta)^2)
  sem_results$cr_rand_mse[i]  <- mean((yte - Xte %*% beta_rand)^2)
  sem_results$oracle_cr_mse[i]<- mean((yte - Xte %*% beta_oracle)^2)

  # AUC (env discovery)
  sem_results$acr_adv_auc[i]  <- compute_auc(fit_adv$w, env_tr)
  sem_results$cr_rand_auc[i]  <- compute_auc(w_rand,    env_tr)
  sem_results$oracle_auc[i]   <- compute_auc(w_oracle,  env_tr)  # should be ~1

  # Coefficient recovery
  sem_results$ols_coef_err[i]      <- coef_error(ols_beta,     beta_true)
  sem_results$acr_adv_coef_err[i]  <- coef_error(fit_adv$beta, beta_true)
  sem_results$cr_rand_coef_err[i]  <- coef_error(beta_rand,    beta_true)
  sem_results$oracle_coef_err[i]   <- coef_error(beta_oracle,  beta_true)

  sem_results$acr_adv_gamma[i] <- g_adv
  sem_results$cr_rand_gamma[i] <- g_rand
  sem_results$oracle_gamma[i]  <- g_oracle

  setTxtProgressBar(pb, i)
}
close(pb)

# ============================================================
# 2. AIRQUALITY EXPERIMENT every holdout month, 10 seeds
# ============================================================

cat("\n=== Running airquality experiments ===\n")

data(airquality)
aq            <- na.omit(airquality)
aq$logOzone   <- log(aq$Ozone)
pred_vars     <- c("Solar.R","Wind","Temp")
aq[pred_vars] <- scale(aq[pred_vars])

holdout_months <- 5:9
aq_rows        <- expand.grid(holdout_month = holdout_months, seed = seeds,
                               stringsAsFactors = FALSE)
aq_metrics     <- c("ols_mse","acr_adv_mse","cr_rand_mse",
                    "acr_adv_auc", "cr_rand_auc",
                    "n_train","n_test",
                    "acr_adv_gamma","cr_rand_gamma")
aq_results <- cbind(aq_rows,
                    as.data.frame(matrix(NA_real_, nrow(aq_rows), length(aq_metrics),
                                         dimnames = list(NULL, aq_metrics))))

pb2 <- txtProgressBar(min=0, max=nrow(aq_results), style=3)

for (i in seq_len(nrow(aq_results))) {
  hm  <- aq_results$holdout_month[i]
  sid <- aq_results$seed[i]

  tr_idx <- which(aq$Month != hm)
  te_idx <- which(aq$Month == hm)
  if (length(te_idx) < 5 || length(tr_idx) < 20) {
    setTxtProgressBar(pb2, i); next
  }

  Xtr <- as.matrix(aq[tr_idx, pred_vars]);  ytr <- aq$logOzone[tr_idx]
  Xte <- as.matrix(aq[te_idx, pred_vars]);  yte <- aq$logOzone[te_idx]

  set.seed(sid * 100)
  w_rand <- as.numeric(sample(c(0,1), nrow(Xtr), replace = TRUE))

  g_adv  <- cv_gamma(Xtr, ytr, gamma_grid, method = "adv",  seed = sid)
  g_rand <- cv_gamma(Xtr, ytr, gamma_grid, method = "rand", seed = sid)

  ols_beta  <- as.vector(coef(lm(ytr ~ Xtr - 1)))
  fit_adv   <- fit_acr_adv(Xtr, ytr, g_adv)
  beta_rand <- fit_cr_fixed(Xtr, ytr, g_rand, w_rand)

  aq_results$ols_mse[i]      <- mean((yte - Xte %*% ols_beta)^2)
  aq_results$acr_adv_mse[i] <- mean((yte - Xte %*% fit_adv$beta)^2)
  aq_results$cr_rand_mse[i] <- mean((yte - Xte %*% beta_rand)^2)

  # AUC vs month (proxy for environment label)
  # treat holdout month as env2, others as env1
  month_label <- ifelse(aq$Month[tr_idx] == hm, 2L, 1L)  # all train = non-hm, so all 1
  # Use majority month as env2 proxy instead:
  # Within training, the month most different from median month = env2
  med_month   <- median(aq$Month[tr_idx])
  env_proxy   <- ifelse(aq$Month[tr_idx] > med_month, 2L, 1L)
  aq_results$acr_adv_auc[i] <- compute_auc(fit_adv$w, env_proxy)
  aq_results$cr_rand_auc[i] <- compute_auc(w_rand,    env_proxy)

  aq_results$n_train[i]        <- nrow(Xtr)
  aq_results$n_test[i]         <- nrow(Xte)
  aq_results$acr_adv_gamma[i] <- g_adv
  aq_results$cr_rand_gamma[i] <- g_rand

  setTxtProgressBar(pb2, i)
}
close(pb2)

# ============================================================
# 3. REST OF DATASETS 
#       ChickWeight  split: diets 1-3 (train) vs diet 4 (test)
#       Boston       split: crim < 1 (train) vs crim >= 1 (test)
#       CO2          split: Quebec (train) vs Mississippi (test)
# ============================================================

run_real_dataset <- function(Xtr, ytr, Xte, yte, env_tr, gamma_grid, seeds,
                             tag, groups = NULL) {
  cat("  ", tag, " (n_train = ", nrow(Xtr), ", n_test = ", nrow(Xte), ")\n", sep = "")
  ols_beta <- as.vector(coef(lm(ytr ~ Xtr - 1))) 
  per_seed <- lapply(seeds, function(sid) {
    set.seed(sid * 100)
    w_rand <- as.numeric(sample(c(0, 1), nrow(Xtr), replace = TRUE))
    g_adv  <- cv_gamma(Xtr, ytr, gamma_grid, method = "adv",
                       seed = sid, groups = groups)
    g_rand <- cv_gamma(Xtr, ytr, gamma_grid, method = "rand",
                       seed = sid, groups = groups)
    fit_adv   <- fit_acr_adv(Xtr, ytr, g_adv)
    beta_rand <- fit_cr_fixed(Xtr, ytr, g_rand, w_rand)
    mse <- function(b) mean((yte - Xte %*% b)^2)
    data.frame(seed        = sid,
               ols_mse     = mse(ols_beta),
               acr_adv_mse = mse(fit_adv$beta),
               cr_rand_mse = mse(beta_rand),
               acr_adv_auc = compute_auc(fit_adv$w, env_tr),
               cr_rand_auc = compute_auc(w_rand,    env_tr),
               acr_adv_gamma = g_adv,
               cr_rand_gamma = g_rand)
  })
  per_seed <- do.call(rbind, per_seed)
  per_seed$dataset <- tag
  per_seed$adv_vs_ols  <- (per_seed$acr_adv_mse - per_seed$ols_mse) / per_seed$ols_mse * 100
  per_seed$rand_vs_ols <- (per_seed$cr_rand_mse - per_seed$ols_mse) / per_seed$ols_mse * 100
  summ <- data.frame(
    dataset       = tag,
    n_train       = nrow(Xtr),
    n_test        = nrow(Xte),
    ols_mse       = mean(per_seed$ols_mse),
    acr_adv_mse   = mean(per_seed$acr_adv_mse),
    cr_rand_mse   = mean(per_seed$cr_rand_mse),
    acr_adv_auc   = mean(per_seed$acr_adv_auc),
    cr_rand_auc   = mean(per_seed$cr_rand_auc),
    auc_null      = auc_null_mean(sum(env_tr == 2), sum(env_tr == 1)),
    acr_adv_gamma = mean(per_seed$acr_adv_gamma),
    cr_rand_gamma = mean(per_seed$cr_rand_gamma),
    stringsAsFactors = FALSE)
  attr(summ, "per_seed") <- per_seed
  summ
}
 
# --- ChickWeight ---
data(ChickWeight)
cw     <- as.data.frame(ChickWeight)
cw_tr  <- cw[cw$Diet %in% c(1, 2, 3), ]
cw_te  <- cw[cw$Diet == 4, ]
mu_cw  <- mean(cw_tr$Time);  sd_cw <- sd(cw_tr$Time)
Xtr_cw <- add_int(scale(matrix(cw_tr$Time, ncol = 1), mu_cw, sd_cw))
Xte_cw <- add_int(scale(matrix(cw_te$Time, ncol = 1), mu_cw, sd_cw))
env_cw <- ifelse(cw_tr$Diet == 3, 2L, 1L)
 
cw_res <- run_real_dataset(Xtr_cw, cw_tr$weight, Xte_cw, cw_te$weight,
                           env_cw, gamma_grid, seeds, "ChickWeight",
                           groups = cw_tr$Chick)
 
# --- Boston housing ---
library(MASS)
bos      <- Boston
bos_tr   <- bos[bos$crim <  1, ]
bos_te   <- bos[bos$crim >= 1, ]
pred_bos <- setdiff(names(bos), c("medv", "crim"))
mu_bos   <- colMeans(as.matrix(bos_tr[, pred_bos]))
sd_bos   <- apply(as.matrix(bos_tr[, pred_bos]), 2, sd)
Xtr_bos  <- add_int(scale(as.matrix(bos_tr[, pred_bos]), mu_bos, sd_bos))
Xte_bos  <- add_int(scale(as.matrix(bos_te[, pred_bos]), mu_bos, sd_bos))
env_bos  <- ifelse(bos_tr$crim > median(bos_tr$crim), 2L, 1L)
 
bos_res <- run_real_dataset(Xtr_bos, bos_tr$medv, Xte_bos, bos_te$medv,
                            env_bos, gamma_grid, seeds, "Boston")
 
# --- CO2 uptake ---
data(CO2)
co2df    <- as.data.frame(CO2)
co2_tr   <- co2df[co2df$Type == "Quebec", ]
co2_te   <- co2df[co2df$Type == "Mississippi", ]
mu_co2   <- mean(co2_tr$conc);  sd_co2 <- sd(co2_tr$conc)
Xtr_co2  <- add_int(scale(matrix(co2_tr$conc, ncol = 1), mu_co2, sd_co2))
Xte_co2  <- add_int(scale(matrix(co2_te$conc, ncol = 1), mu_co2, sd_co2))
env_co2  <- ifelse(co2_tr$Treatment == "chilled", 2L, 1L)
 
co2_res <- run_real_dataset(Xtr_co2, co2_tr$uptake, Xte_co2, co2_te$uptake,
                            env_co2, gamma_grid, seeds, "CO2",
                            groups = co2_tr$Plant)
 
# keep the per-seed rows: the unified figure needs their spread
real_seed_results <- do.call(rbind, lapply(list(cw_res, bos_res, co2_res),
                                           function(d) attr(d, "per_seed")))
real_results <- rbind(cw_res, bos_res, co2_res)
real_results$adv_vs_ols  <- round((real_results$acr_adv_mse - real_results$ols_mse) /
                                    real_results$ols_mse * 100, 1)
real_results$rand_vs_ols <- round((real_results$cr_rand_mse - real_results$ols_mse) /
                                    real_results$ols_mse * 100, 1)
# --- CMNIST ---
gaps      <- c(0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30)
seeds     <- 1:5
n_per_env <- 2000
n_test    <- 2000

rd <- function(tag, k) as.matrix(read.csv(sprintf("cm_data/%s_%s.csv", tag, k), header = FALSE))

outfile <- "cmnist_results.csv"
out <- if (file.exists(outfile)) read.csv(outfile) else NULL

for (gp in gaps) for (sd_i in seeds) {
  if (!is.null(out) && any(abs(out$gap - gp) < 1e-9 & out$seed == sd_i)) next
  tag <- sprintf("g%03d_s%d", round(gp * 100), sd_i)
  system(sprintf("python3 make_cmnist.py %f %d %d %d %s > /dev/null",
                 gp, sd_i, n_per_env, n_test, tag))

  Xtr <- cbind(1, rd(tag, "Xtr")); ytr <- as.vector(rd(tag, "ytr"))
  Xte <- cbind(1, rd(tag, "Xte")); yte <- as.vector(rd(tag, "yte"))
  Gtr <- cbind(1, rd(tag, "Gtr")); Gte <- cbind(1, rd(tag, "Gte"))
  env <- as.vector(rd(tag, "env"))

  t0  <- Sys.time()
  fit <- fit_acr_two_phase(Xtr, ytr, seed = sd_i)

  b_erm  <- ridge_ols(Xtr, ytr)
  w_orac <- as.numeric(env == 1)
  b_orcr <- acr_learner(Xtr, ytr, w_orac, fit$gamma)

  set.seed(1000 + sd_i)                                   # random balanced split
  w_rand <- ifelse(sample(rep(c(0, 1), length.out = nrow(Xtr))) == 1,
                   1 - HP$eps, HP$eps)
  b_crrd <- acr_learner(Xtr, ytr, w_rand, fit$gamma)

  b_gray <- ridge_ols(Gtr, ytr)                           # colour-free oracle

  mse <- function(X, b) mean((yte - X %*% b)^2)
  acc <- function(X, b) mean((as.vector(X %*% b) > 0.5) == (yte > 0.5))

  row <- data.frame(
    gap = gp, seed = sd_i, corr1 = 0.85 + gp / 2, corr2 = 0.85 - gp / 2,
    erm_mse    = mse(Xte, b_erm),  erm_acc    = acc(Xte, b_erm),
    acr_mse    = mse(Xte, fit$beta), acr_acc  = acc(Xte, fit$beta),
    crrand_mse = mse(Xte, b_crrd), crrand_acc = acc(Xte, b_crrd),
    oracr_mse  = mse(Xte, b_orcr), oracr_acc  = acc(Xte, b_orcr),
    gray_mse   = mse(Gte, b_gray), gray_acc   = acc(Gte, b_gray),
    adv_auc    = compute_auc(fit$w, env),
    auc_null   = auc_null_mean(n_per_env, n_per_env),
    gamma      = fit$gamma, gamma_max = fit$gamma_max,
    secs       = as.numeric(Sys.time() - t0, units = "secs"))

  out <- rbind(out, row)
  write.csv(out, outfile, row.names = FALSE)
  unlink(list.files("cm_data", pattern = paste0("^", tag, "_"), full.names = TRUE))
  cat(sprintf("gap %.2f seed %d | ERM %.4f  ACR %.4f  OrCR %.4f  gray %.4f | AUC %.3f  G* %.2f | %.0fs\n",
              gp, sd_i, row$erm_mse, row$acr_mse, row$oracr_mse, row$gray_mse,
              row$adv_auc, row$gamma, row$secs))
  flush.console()
}

cat("\n== aggregated over seeds ==\n")
agg <- aggregate(cbind(erm_mse, acr_mse, crrand_mse, oracr_mse, gray_mse,
                       erm_acc, acr_acc, oracr_acc, gray_acc,
                       adv_auc, gamma) ~ gap, data = out, FUN = mean)
agg$acr_vs_erm_pct <- 100 * (agg$erm_mse - agg$acr_mse) / agg$erm_mse
agg$crrand_vs_erm_pct <- 100 * (agg$erm_mse -
                                aggregate(crrand_mse ~ gap, out, mean)$crrand_mse) / agg$erm_mse
print(round(agg, 4), row.names = FALSE)
write.csv(agg, "cmnist_aggregated.csv", row.names = FALSE)

                                           
# ============================================================
# 4. SUMMARY TABLES
# ============================================================

cat("\n\n============================================================\n")
cat("SEM RESULTS: MEAN SHIFT (averaged over 10 seeds)\n")
cat("============================================================\n")
d  <- subset(sem_results, shift_type == "mean")
ag <- aggregate(cbind(ols_mse, acr_adv_mse, cr_rand_mse, oracle_cr_mse,
                      acr_adv_auc, cr_rand_auc,
                      ols_coef_err, acr_adv_coef_err, cr_rand_coef_err,
                      oracle_coef_err) ~ alpha, data = d, FUN = mean)
print(round(ag, 3))

cat("\n============================================================\n")
cat("SEM RESULTS: VARIANCE SHIFT (averaged over 10 seeds)\n")
cat("============================================================\n")
d  <- subset(sem_results, shift_type == "variance")
ag <- aggregate(cbind(ols_mse, acr_adv_mse, cr_rand_mse, oracle_cr_mse,
                      acr_adv_auc, cr_rand_auc) ~ alpha, data = d, FUN = mean)
print(round(ag, 3))

cat("\n============================================================\n")
cat("AIRQUALITY RESULTS BY HOLDOUT MONTH (averaged over 10 seeds)\n")
cat("============================================================\n")
ag_aq <- aggregate(cbind(ols_mse, acr_adv_mse, cr_rand_mse) ~ holdout_month,
                   data = aq_results, FUN = mean, na.rm = TRUE)
ag_aq$adv_vs_ols  <- round((ag_aq$acr_adv_mse  - ag_aq$ols_mse) / ag_aq$ols_mse * 100, 1)
ag_aq$rand_vs_ols <- round((ag_aq$cr_rand_mse  - ag_aq$ols_mse) / ag_aq$ols_mse * 100, 1)
ag_aq$adv_vs_rand <- round((ag_aq$acr_adv_mse  - ag_aq$cr_rand_mse) / ag_aq$cr_rand_mse * 100, 1)
month_names        <- c("5"="May","6"="Jun","7"="Jul","8"="Aug","9"="Sep")
ag_aq$month        <- month_names[as.character(ag_aq$holdout_month)]
num_cols <- c("ols_mse","acr_adv_mse","cr_rand_mse","adv_vs_ols","rand_vs_ols","adv_vs_rand")
out_tbl  <- ag_aq[, c("month", num_cols)]
out_tbl[num_cols] <- round(out_tbl[num_cols], 3)
print(out_tbl)

# ============================================================
# 5. PLOTS
# ============================================================

col_ols    <- "#2980B9"
col_adv    <- "#C0392B"
col_rand   <- "#27AE60"
col_oracle <- "#8E44AD"

make_ci_band <- function(x_vals, y_vals, y_sd, col, alpha = 0.15) {
  polygon(c(x_vals, rev(x_vals)),
          c(y_vals + y_sd, rev(y_vals - y_sd)),
          col = adjustcolor(col, alpha.f = alpha), border = NA)
}

# --- Plot 1: SEM OOD MSE vs alpha w/MEAN SHIFT ---
pdf(paste0(out_dir,"6-sem_ood_mse_mean.pdf"), width=8, height=5)
par(mar=c(5,5,3,2))
d   <- subset(sem_results, shift_type=="mean")
ag  <- aggregate(cbind(ols_mse,acr_adv_mse,cr_rand_mse,oracle_cr_mse) ~ alpha, d, mean)
agsd<- aggregate(cbind(ols_mse,acr_adv_mse,cr_rand_mse,oracle_cr_mse) ~ alpha, d, sd)

ylim <- range(ag[,-1]) * c(0.8, 1.2)
plot(ag$alpha, ag$ols_mse, type="b", pch=16, col=col_ols, lwd=2,
     ylim=ylim, xlab="Training shift strength (alpha)",
     ylab="OOD MSE", main="SEM w/Mean Shift: OOD MSE vs Shift Strength")
make_ci_band(ag$alpha, ag$ols_mse, agsd$ols_mse, col_ols)
lines(ag$alpha, ag$acr_adv_mse,  type="b", pch=17, col=col_adv,    lwd=2)
make_ci_band(ag$alpha, ag$acr_adv_mse,  agsd$acr_adv_mse,  col_adv)
lines(ag$alpha, ag$cr_rand_mse,  type="b", pch=15, col=col_rand,   lwd=2, lty=2)
lines(ag$alpha, ag$oracle_cr_mse,type="b", pch=18, col=col_oracle, lwd=2, lty=3)
legend("topleft",
       legend=c("OLS","ACR (adversarial)","CR (random split)","Oracle CR (true labels)"),
       col=c(col_ols,col_adv,col_rand,col_oracle),
       lwd=2, pch=c(16,17,15,18), lty=c(1,1,2,3), bty="n", cex=0.85)
dev.off()

# --- Plot 2: SEM OOD MSE vs alpha w/VARIANCE SHIFT ---
pdf(paste0(out_dir,"6-sem_ood_mse_variance.pdf"), width=8, height=5)
par(mar=c(5,5,3,2))
d   <- subset(sem_results, shift_type=="variance")
ag  <- aggregate(cbind(ols_mse,acr_adv_mse,cr_rand_mse,oracle_cr_mse) ~ alpha, d, mean)

ylim <- range(ag[,-1]) * c(0.8, 1.2)
plot(ag$alpha, ag$ols_mse, type="b", pch=16, col=col_ols, lwd=2,
     ylim=ylim, xlab="Training shift strength (alpha)",
     ylab="OOD MSE", main="SEM â Variance Shift: OOD MSE vs Shift Strength")
lines(ag$alpha, ag$acr_adv_mse,  type="b", pch=17, col=col_adv,    lwd=2)
lines(ag$alpha, ag$cr_rand_mse,  type="b", pch=15, col=col_rand,   lwd=2, lty=2)
lines(ag$alpha, ag$oracle_cr_mse,type="b", pch=18, col=col_oracle, lwd=2, lty=3)
abline(h = ag$ols_mse[1], lty=3, col="grey60")
legend("topleft",
       legend=c("OLS","ACR (adversarial)","CR (random split)","Oracle CR (true labels)"),
       col=c(col_ols,col_adv,col_rand,col_oracle),
       lwd=2, pch=c(16,17,15,18), lty=c(1,1,2,3), bty="n", cex=0.85)
dev.off()

# --- Plot 3: Environment discovery AUC vs alpha ---
pdf(paste0(out_dir,"6-sem_auc_vs_alpha.pdf"), width=8, height=5)
par(mar=c(5,5,3,2))
d_m <- subset(sem_results, shift_type=="mean")
d_v <- subset(sem_results, shift_type=="variance")
ag_m <- aggregate(cbind(acr_adv_auc,cr_rand_auc) ~ alpha, d_m, mean)
ag_v <- aggregate(acr_adv_auc ~ alpha, d_v, mean)

plot(ag_m$alpha, ag_m$acr_adv_auc, type="b", pch=17, col=col_adv, lwd=2,
     ylim=c(0.45,1.05), xlab="Training shift strength (alpha)",
     ylab="Environment discovery AUC",
     main="Adversary: How Well Does It Recover the True Split?")
lines(ag_v$alpha, ag_v$acr_adv_auc, type="b", pch=17, col=col_adv, lwd=2, lty=2)
lines(ag_m$alpha, ag_m$cr_rand_auc, type="b", pch=15, col=col_rand, lwd=2)
abline(h=0.5, lty=3, col="grey50"); abline(h=1.0, lty=3, col="grey80")
text(max(ag_m$alpha)*0.6, 0.52, "Chance (0.5)", col="grey50", cex=0.8)
legend("bottomright",
       legend=c("ACR-Adv (mean shift)","ACR-Adv (variance shift)","Random split (mean shift)"),
       col=c(col_adv,col_adv,col_rand), lwd=2,
       pch=c(17,17,15), lty=c(1,2,1), bty="n", cex=0.85)
dev.off()

# --- Plot 4: Coefficient recovery error vs alpha (mean shift) ---
pdf(paste0(out_dir,"6-sem_coef_recovery.pdf"), width=8, height=5)
par(mar=c(5,5,3,2))
d  <- subset(sem_results, shift_type=="mean")
ag <- aggregate(cbind(ols_coef_err,acr_adv_coef_err,cr_rand_coef_err,
                      oracle_coef_err) ~ alpha, d, mean)

ylim <- c(0, max(ag$ols_coef_err)*1.1)
plot(ag$alpha, ag$ols_coef_err, type="b", pch=16, col=col_ols, lwd=2,
     ylim=ylim, xlab="Training shift strength (alpha)",
     ylab="Normalised coefficient error vs beta*",
     main="Causal Coefficient Recovery w/Mean Shift")
lines(ag$alpha, ag$acr_adv_coef_err,  type="b", pch=17, col=col_adv,    lwd=2)
lines(ag$alpha, ag$cr_rand_coef_err,  type="b", pch=15, col=col_rand,   lwd=2, lty=2)
lines(ag$alpha, ag$oracle_coef_err,   type="b", pch=18, col=col_oracle, lwd=2, lty=3)
legend("topright",
       legend=c("OLS","ACR (adversarial)","CR (random split)","Oracle CR"),
       col=c(col_ols,col_adv,col_rand,col_oracle),
       lwd=2, pch=c(16,17,15,18), lty=c(1,1,2,3), bty="n", cex=0.85)
dev.off()

# --- Plot 5: Airquality OOD MSE by holdout month ---
df(paste0(out_dir,"6-aq_mse_by_month.pdf"), width=8, height=5)
par(mar=c(5,5,3,2))
ag_aq  <- aggregate(cbind(ols_mse,acr_adv_mse,cr_rand_mse) ~ holdout_month,
                    aq_results, mean, na.rm=TRUE)
ag_aqsd<- aggregate(cbind(ols_mse,acr_adv_mse,cr_rand_mse) ~ holdout_month,
                    aq_results, sd, na.rm=TRUE)
mnms   <- c("5"="May","6"="Jun","7"="Jul","8"="Aug","9"="Sep")

bp <- barplot(t(as.matrix(ag_aq[, c("ols_mse","acr_adv_mse","cr_rand_mse")])),
              beside=TRUE, names.arg=mnms[as.character(ag_aq$holdout_month)],
              col=c(col_ols,col_adv,col_rand),
              ylab="OOD MSE (log-Ozone)", main="Airquality: OOD MSE by Holdout Month",
              legend.text=c("OLS","ACR (adversarial)","CR (random)"),
              args.legend=list(x="topright", bty="n", cex=0.85))
dev.off()

# --- Plot 6: ACR-Adv vs CR-Rand head-to-head scatter (all experiments) ---
pdf(paste0(out_dir,"6-adv_vs_rand_scatter.pdf"), width=6, height=6)
par(mar=c(5,5,3,2))
x_rand <- c(sem_results$cr_rand_mse[sem_results$shift_type=="mean"],
             aq_results$cr_rand_mse)
x_adv  <- c(sem_results$acr_adv_mse[sem_results$shift_type=="mean"],
             aq_results$acr_adv_mse)
ok     <- !is.na(x_rand) & !is.na(x_adv)
xlim   <- ylim2 <- range(c(x_rand[ok], x_adv[ok]))
plot(x_rand[ok], x_adv[ok], pch=16, col=adjustcolor(col_adv, 0.4),
     xlim=xlim, ylim=ylim2,
     xlab="CR (random split) OOD MSE",
     ylab="ACR (adversarial) OOD MSE",
     main="Head-to-Head: Adversarial vs Random Split")
abline(0, 1, lty=2, col="grey40", lwd=2)
pct_adv_wins <- mean(x_adv[ok] < x_rand[ok]) * 100
legend("topleft", bty="n", cex=0.9,
       legend=sprintf("Adversarial wins in %.0f%% of scenarios", pct_adv_wins))
dev.off()

save(sem_results, aq_results,
     file=paste0(out_dir,"6-systematic_results.RData"))
cat("\nAll done. Plots and results saved to:\n", out_dir, "\n")
