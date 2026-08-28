out_dir <- "/Users/janamunoz/Desktop/Literatura tesis/working folder/"

HP <- list(
  T1      = 500,    # Phase 1 gradient-ascent steps
  eta     = 0.001,  # max per-step weight change (gradient is normalised)
  eta_ent = 0.02,   # entropy coefficient
  kappa   = 5.00,   # balance penalty coefficient
  mu      = 0.90,   # momentum coefficient
  rho     = 0.50,   # balance target
  eps     = 0.05,   # weight clipping, w in [eps, 1-eps]
  n_grid  = 10,     # size of the adaptive Gamma grid
  n_path  = 201,    # resolution of the learner path scan
  K       = 5,      # cross-validation folds
  lambda  = 1e-8    # ridge, for invertibility only
)

# sd_y = 0.5 makes the Causal Oracle risk equal Var(eps_Y) = 0.25, matching
# Richter & Wit Table 1. rescale divides by training sd but does NOT centre —
# centering breaks the WRD closed form.
gen_sem <- function(n_per_env, alpha_train, alpha_test,
                    shift_type = "mean", seed = 1,
                    sd_y = 0.5, sd_child = 0.5, rescale = FALSE) {
  set.seed(seed)
  n_train <- 2 * n_per_env
  env_tr  <- rep(c(1L, 2L), each = n_per_env)

  draw_shift <- function(n, env, alpha) {
    if (shift_type == "mean")
      ifelse(env == 1, rnorm(n, 0, 1), rnorm(n, alpha, 1))
    else
      ifelse(env == 1, rnorm(n, 0, 1), rnorm(n, 0, max(alpha, 0.1)))
  }

  X1 <- rnorm(n_train)
  X2 <- X1 + rnorm(n_train)
  X3 <- X1 + X2 + rnorm(n_train)
  Y  <- X2 + X3 + rnorm(n_train, 0, sd_y)
  A4 <- draw_shift(n_train, env_tr, alpha_train)
  A5 <- draw_shift(n_train, env_tr, alpha_train)
  X4 <- Y  + A4 + rnorm(n_train, 0, sd_child)
  X5 <- Y  + A5 + rnorm(n_train, 0, sd_child)
  X6 <- X5 + rnorm(n_train, 0, sd_child)
  Xtr <- cbind(X1, X2, X3, X4, X5, X6)

  n_test <- n_per_env
  env_te <- rep(2L, n_test)
  X1t <- rnorm(n_test)
  X2t <- X1t + rnorm(n_test)
  X3t <- X1t + X2t + rnorm(n_test)
  Yt  <- X2t + X3t + rnorm(n_test, 0, sd_y)
  A4t <- draw_shift(n_test, env_te, alpha_test)
  A5t <- draw_shift(n_test, env_te, alpha_test)
  X4t <- Yt  + A4t + rnorm(n_test, 0, sd_child)
  X5t <- Yt  + A5t + rnorm(n_test, 0, sd_child)
  X6t <- X5t + rnorm(n_test, 0, sd_child)
  Xte <- cbind(X1t, X2t, X3t, X4t, X5t, X6t)

  sg <- if (rescale) apply(Xtr, 2, sd) else rep(1, ncol(Xtr))
  Xtr <- sweep(Xtr, 2, sg, "/")     # scale only; NEVER centre
  Xte <- sweep(Xte, 2, sg, "/")
  colnames(Xtr) <- colnames(Xte) <- paste0("X", 1:6)

  list(X_tr = Xtr, y_tr = Y, env_tr = env_tr,
       X_te = Xte, y_te = Yt, sd_X = sg)
}

env_risks <- function(r2, w) {
  w <- pmin(pmax(w, 1e-12), 1 - 1e-12)
  c(R1 = sum(w * r2) / sum(w), R2 = sum((1 - w) * r2) / sum(1 - w))
}
acr_objective <- function(r2, w, gamma) {
  R <- env_risks(r2, w)
  unname(R["R1"] + R["R2"] + gamma * abs(R["R1"] - R["R2"]))
}
binary_entropy <- function(w) {
  w <- pmin(pmax(w, 1e-12), 1 - 1e-12)
  -mean(w * log(w) + (1 - w) * log(1 - w))
}
ginv_solve <- function(A, b, tol = 1e-10) {
  sv <- svd(A)
  d  <- ifelse(abs(sv$d) > tol * max(abs(sv$d)), 1 / sv$d, 0)
  as.vector(sv$v %*% (d * (t(sv$u) %*% b)))
}
ridge_ols <- function(X, y, lambda = HP$lambda)
  as.vector(solve(crossprod(X) + lambda * diag(ncol(X)), crossprod(X, y)))

# Every stationary point of L = R_+ + Gamma|R_A| lies on the path
# beta(theta) = (G_+ + theta*G_A + lambda*I)^{-1} (Z_+ + theta*Z_A)
# for theta in [-Gamma, Gamma]. Scanning this path finds the exact global min.
make_path <- function(X, y, w, ridge = HP$lambda) {
  w  <- pmin(pmax(w, 1e-12), 1 - 1e-12)
  w1 <- w / sum(w); w2 <- (1 - w) / sum(1 - w)
  G1 <- t(X * w1) %*% X; G2 <- t(X * w2) %*% X
  Z1 <- t(X * w1) %*% y; Z2 <- t(X * w2) %*% y
  Gp <- G1 + G2; Ga <- G1 - G2; Zp <- Z1 + Z2; Za <- Z1 - Z2
  Ip <- diag(ncol(X))
  function(theta) {
    A <- Gp + theta * Ga + ridge * Ip
    b <- Zp + theta * Za
    beta <- tryCatch(as.vector(solve(A, b)), error = function(e) ginv_solve(A, b))
    if (!all(is.finite(beta))) beta <- ginv_solve(A, b)
    beta
  }
}

acr_learner <- function(X, y, w, gamma, n_path = HP$n_path, ridge = HP$lambda) {
  beta_of <- make_path(X, y, w, ridge)
  if (gamma <= 0) return(beta_of(0))
  thetas <- seq(-gamma, gamma, length.out = n_path)
  L <- vapply(thetas, function(th) {
    b <- beta_of(th)
    acr_objective(as.vector(y - X %*% b)^2, w, gamma)
  }, numeric(1))
  beta_of(thetas[which.min(L)])
}

# Initialise from signed residuals (Richter & Wit eq. 27), not squared —
# signed residuals carry the direction of the environment shift.
init_weights_signed <- function(r) {
  l <- (r - median(r)) / (sd(r) + 1e-12)
  0.1 + 0.8 * plogis(l)
}

phase1_discover <- function(X, y, hp = HP, trace = FALSE) {
  n  <- nrow(X)
  b0 <- ridge_ols(X, y, hp$lambda)      # beta held FIXED throughout Phase 1
  r  <- as.vector(y - X %*% b0)
  r2 <- r^2
  w  <- init_weights_signed(r)
  m  <- rep(0, n)
  hist <- NULL

  if (hp$T1 > 0) for (t in seq_len(hp$T1)) {
    S  <- sum(w); Sb <- n - S
    N1 <- sum(w * r2); N2 <- sum((1 - w) * r2)
    R1 <- N1 / S; R2 <- N2 / Sb
    s  <- sign(R1 - R2); if (s == 0) s <- 1

    # Gradient normalised so hp$eta is the max per-step weight change;
    # without this, 500 steps move weights by < 0.05 in total.
    dR1 <-  (r2 * S  - N1) / S^2
    dR2 <- -(r2 * Sb - N2) / Sb^2
    g <- s * (dR1 - dR2)
    g <- g / (max(abs(g)) + 1e-12)

    g <- g - hp$kappa * 2 * (mean(w) - hp$rho) -
             hp$eta_ent * (log(w) - log(1 - w))

    m <- hp$mu * m + (1 - hp$mu) * g
    w <- pmin(pmax(w + hp$eta * m, hp$eps), 1 - hp$eps)

    if (trace && (t == 1 || t %% 10 == 0)) {
      RR <- env_risks(r2, w)
      hist <- rbind(hist, data.frame(t = t,
        contrast = unname(abs(RR[1] - RR[2])),
        entropy  = binary_entropy(w), wbar = mean(w)))
    }
  }
  list(w = w, r2 = r2, beta_ols = b0, hist = hist)
}

gamma_max_of <- function(r2, w) {
  R  <- env_risks(r2, w)
  gm <- unname((R["R1"] + R["R2"]) / max(abs(R["R1"] - R["R2"]), 1e-10))
  min(max(gm, 0.5), 1e4)
}
adaptive_grid <- function(gamma_max, n_grid = HP$n_grid)
  c(0, exp(seq(log(0.1), log(gamma_max), length.out = n_grid)))

# CV uses the robust ACR criterion (Richter & Wit eq. 41), not plain MSE.
select_gamma <- function(X, y, w, gamma_grid, gamma_cv, hp = HP, seed = 1) {
  set.seed(seed)
  folds <- sample(rep(seq_len(hp$K), length.out = nrow(X)))
  scores <- vapply(gamma_grid, function(g) {
    mean(vapply(seq_len(hp$K), function(k) {
      tr <- folds != k; va <- folds == k
      b  <- acr_learner(X[tr, , drop = FALSE], y[tr], w[tr], g, hp$n_path)
      r2v <- as.vector(y[va] - X[va, , drop = FALSE] %*% b)^2
      acr_objective(r2v, w[va], gamma_cv)
    }, numeric(1)))
  }, numeric(1))
  list(gamma = gamma_grid[which.min(scores)], scores = scores)
}

fit_acr_two_phase <- function(X, y, hp = HP, seed = 1, trace = FALSE) {
  ph  <- phase1_discover(X, y, hp, trace)
  gm  <- gamma_max_of(ph$r2, ph$w)
  grd <- adaptive_grid(gm, hp$n_grid)
  sel <- select_gamma(X, y, ph$w, grd, gamma_cv = gm, hp = hp, seed = seed)
  list(beta = acr_learner(X, y, ph$w, sel$gamma, hp$n_path),
       w = ph$w, gamma = sel$gamma, gamma_max = gm,
       grid = grd, scores = sel$scores, hist = ph$hist, beta_ols = ph$beta_ols)
}

compute_auc <- function(w, true_env) {
  label <- as.integer(true_env == 2)
  n1 <- sum(label == 1); n0 <- sum(label == 0)
  if (n1 == 0 || n0 == 0) return(0.5)
  auc <- (sum(rank(w)[label == 1]) - n1 * (n1 + 1) / 2) / (n1 * n0)
  max(auc, 1 - auc)
}
# Symmetrised AUC >= 0.5 by construction; null mean is NOT 0.5.
auc_null_mean <- function(n1, n0)
  0.5 + sqrt((n1 + n0 + 1) / (12 * n1 * n0)) * sqrt(2 / pi)

coef_dir_error <- function(beta_hat, beta_true, sd_X) {
  bt <- beta_true * sd_X
  u  <- function(v) v / (sqrt(sum(v^2)) + 1e-12)
  sqrt(sum((u(beta_hat) - u(bt))^2))
}

# ── Experiment ────────────────────────────────────────────────────────────────

alpha_grid  <- c(1, 3, 5, 10)
seeds       <- 1:10
shift_types <- c("mean", "variance")
n_per_env   <- 250
alpha_test  <- 20
beta_true   <- c(0, 1, 1, 0, 0, 0)

rows <- expand.grid(alpha = alpha_grid, seed = seeds,
                    shift_type = shift_types, stringsAsFactors = FALSE)
cols <- c("ols_mse", "acr_mse", "oracle_cr_mse", "causal_mse",
          "acr_auc", "acr_gamma", "gamma_max",
          "ols_coef", "acr_coef", "alpha_cover")
res  <- cbind(rows, as.data.frame(matrix(NA_real_, nrow(rows), length(cols),
                                         dimnames = list(NULL, cols))))

pb <- txtProgressBar(min = 0, max = nrow(rows), style = 3)

for (i in seq_len(nrow(rows))) {
  al <- res$alpha[i]; sid <- res$seed[i]; sht <- res$shift_type[i]
  d  <- gen_sem(n_per_env, al, alpha_test, sht, sid)
  Xtr <- d$X_tr; ytr <- d$y_tr; Xte <- d$X_te; yte <- d$y_te

  fit    <- fit_acr_two_phase(Xtr, ytr, seed = sid)
  b_ols  <- ridge_ols(Xtr, ytr)
  w_orac <- as.numeric(d$env_tr == 1)
  b_orcr <- acr_learner(Xtr, ytr, w_orac, fit$gamma)
  b_caus <- rep(0, 6); b_caus[2:3] <- ridge_ols(Xtr[, 2:3, drop = FALSE], ytr)

  mse <- function(b) mean((yte - Xte %*% b)^2)
  res$ols_mse[i]       <- mse(b_ols)
  res$acr_mse[i]       <- mse(fit$beta)
  res$oracle_cr_mse[i] <- mse(b_orcr)
  res$causal_mse[i]    <- mse(b_caus)
  res$acr_auc[i]       <- compute_auc(fit$w, d$env_tr)
  res$acr_gamma[i]     <- fit$gamma
  res$gamma_max[i]     <- fit$gamma_max
  res$ols_coef[i]      <- coef_dir_error(b_ols,    beta_true, d$sd_X)
  res$acr_coef[i]      <- coef_dir_error(fit$beta, beta_true, d$sd_X)
  res$alpha_cover[i]   <- al * sqrt((fit$gamma + 1) / 2)

  setTxtProgressBar(pb, i)
}
close(pb)

agg <- aggregate(cbind(ols_mse, acr_mse, oracle_cr_mse, causal_mse,
                       acr_auc, acr_gamma, alpha_cover) ~ alpha + shift_type,
                 data = res, FUN = mean)
agg <- agg[order(agg$shift_type, agg$alpha), ]
agg[, 3:9] <- round(agg[, 3:9], 3)
print(agg, row.names = FALSE)

write.csv(res, paste0(out_dir, "9-sem_gradient_results.csv"), row.names = FALSE)
capture.output(print(agg, row.names = FALSE),
               file = paste0(out_dir, "9-sem_gradient_table.txt"))

# ── Phase 1 diagnostics ───────────────────────────────────────────────────────

d_m <- gen_sem(n_per_env, 1, alpha_test, "mean",     seed = 1)
d_v <- gen_sem(n_per_env, 1, alpha_test, "variance", seed = 1)
h_m <- phase1_discover(d_m$X_tr, d_m$y_tr, HP, trace = TRUE)
h_v <- phase1_discover(d_v$X_tr, d_v$y_tr, HP, trace = TRUE)

pdf(paste0(out_dir, "9-phase1_diagnostics.pdf"), width = 10, height = 3.6)
par(mfrow = c(1, 3), mar = c(4.2, 4.2, 2.4, 1))
plot(h_m$hist$t, h_m$hist$contrast, type = "l", col = "#C0392B", lwd = 2,
     xlab = "Phase 1 iteration", ylab = expression("|" * R[1] - R[2] * "|"),
     main = "Risk contrast",
     ylim = range(0, h_m$hist$contrast, h_v$hist$contrast))
lines(h_v$hist$t, h_v$hist$contrast, col = "#2980B9", lwd = 2, lty = 2)
legend("topleft", c("mean shift", "variance shift"),
       col = c("#C0392B", "#2980B9"), lty = c(1, 2), lwd = 2, bty = "n")
plot(h_m$hist$t, h_m$hist$entropy, type = "l", col = "#C0392B", lwd = 2,
     xlab = "Phase 1 iteration", ylab = "H(w)", main = "Weight entropy",
     ylim = range(h_m$hist$entropy, h_v$hist$entropy, log(2)))
lines(h_v$hist$t, h_v$hist$entropy, col = "#2980B9", lwd = 2, lty = 2)
abline(h = log(2), lty = 3, col = "grey40")
plot(h_m$hist$t, h_m$hist$wbar, type = "l", col = "#C0392B", lwd = 2,
     xlab = "Phase 1 iteration", ylab = expression(bar(w)), main = "Balance",
     ylim = c(0, 1))
lines(h_v$hist$t, h_v$hist$wbar, col = "#2980B9", lwd = 2, lty = 2)
abline(h = 0.5, lty = 3, col = "grey40")
dev.off()

# Gradient ascent can erode the signal from signed-residual initialisation
# (finite-sample analogue of Richter & Wit Remark 4); check AUC vs T1.
null_auc <- auc_null_mean(n_per_env, n_per_env)
sens <- do.call(rbind, lapply(c(0, 50, 200, 500, 2000), function(T1) {
  hp <- HP; hp$T1 <- T1
  pm <- phase1_discover(d_m$X_tr, d_m$y_tr, hp)
  pv <- phase1_discover(d_v$X_tr, d_v$y_tr, hp)
  data.frame(T1 = T1,
             auc_mean_shift = round(compute_auc(pm$w, d_m$env_tr), 3),
             auc_var_shift  = round(compute_auc(pv$w, d_v$env_tr), 3),
             gamma_max_mean = round(gamma_max_of(pm$r2, pm$w), 2),
             entropy_mean   = round(binary_entropy(pm$w), 3))
}))
print(sens, row.names = FALSE)
capture.output(print(sens, row.names = FALSE),
               file = paste0(out_dir, "9-phase1_sensitivity.txt"))
