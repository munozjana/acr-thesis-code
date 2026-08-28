# gen_sem with scale() (so the design IS centred), sd_y = 1, children noise 0.5, alpha_test = 1.5 * alpha
# the sigmoid adversary adv_update()
# the truncated learner wls_learner() with M = 3 sign iterations, ridge = 1e-6, and pmax(v, 0)

out <- file("11-methodology_diagnostics.txt", open = "wt")
say <- function(...) { cat(..., sep = ""); cat(..., sep = "", file = out) }

wls_learner <- function(X, y, w, gamma, ridge = 1e-6) {
  w  <- pmin(pmax(w, 1e-10), 1 - 1e-10)
  w1 <- w / sum(w); w2 <- (1 - w) / sum(1 - w)
  beta <- rep(0, ncol(X))
  for (iter in 1:3) {
    r  <- as.vector(y - X %*% beta)
    R1 <- sum(w1 * r^2); R2 <- sum(w2 * r^2)
    s  <- sign(R1 - R2)
    v  <- (1 + gamma * s) * w1 + (1 - gamma * s) * w2
    v  <- pmax(v, 0)
    XtV <- t(X * v)
    beta <- as.vector(solve(XtV %*% X + ridge * diag(ncol(X)), XtV %*% y))
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
    beta <- wls_learner(X, y, w, gamma); w <- adv_update(X, y, beta)
  }
  list(beta = beta, w = w)
}

gen_sem <- function(n_per_env, alpha_train, alpha_test, shift_type = "mean",
                    seed = 1, CENTRE = TRUE) {
  set.seed(seed)
  n <- 2 * n_per_env
  env <- rep(c(1L, 2L), each = n_per_env)
  X1 <- rnorm(n); X2 <- X1 + rnorm(n); X3 <- X1 + X2 + rnorm(n)
  Y  <- X2 + X3 + rnorm(n)                       # sd 1, as in the code
  dr <- function(a) if (shift_type == "mean")
      ifelse(env == 1, rnorm(n, 0, 1), rnorm(n, a, 1))
    else ifelse(env == 1, rnorm(n, 0, 1), rnorm(n, 0, max(a, 0.1)))
  A4 <- dr(alpha_train); A5 <- dr(alpha_train)
  X4 <- Y + A4 + rnorm(n, 0, .5); X5 <- Y + A5 + rnorm(n, 0, .5)
  X6 <- X5 + rnorm(n, 0, .5)
  Xr <- cbind(X1, X2, X3, X4, X5, X6)

  nt <- n_per_env
  X1t <- rnorm(nt); X2t <- X1t + rnorm(nt); X3t <- X1t + X2t + rnorm(nt)
  Yt  <- X2t + X3t + rnorm(nt)
  drt <- function(a) if (shift_type == "mean") rnorm(nt, a, 1)
                     else rnorm(nt, 0, max(a, 0.1))
  A4t <- drt(alpha_test); A5t <- drt(alpha_test)
  X4t <- Yt + A4t + rnorm(nt, 0, .5); X5t <- Yt + A5t + rnorm(nt, 0, .5)
  X6t <- X5t + rnorm(nt, 0, .5)
  Xe <- cbind(X1t, X2t, X3t, X4t, X5t, X6t)

  mu <- if (CENTRE) colMeans(Xr) else rep(0, 6)
  sg <- apply(Xr, 2, sd)
  list(X_tr = scale(Xr, mu, sg), y_tr = Y,
       X_te = scale(Xe, mu, sg), y_te = Yt, env_tr = env, sd_X = sg)
}

risks <- function(r2, w) { w <- pmin(pmax(w, 1e-12), 1 - 1e-12)
  c(sum(w * r2) / sum(w), sum((1 - w) * r2) / sum(1 - w)) }
Lobj <- function(X, y, b, w, g) { R <- risks(as.vector(y - X %*% b)^2, w)
  R[1] + R[2] + g * abs(R[1] - R[2]) }
gsolve <- function(A, b) { sv <- svd(A)
  d <- ifelse(abs(sv$d) > 1e-10 * max(abs(sv$d)), 1/sv$d, 0)
  as.vector(sv$v %*% (d * (t(sv$u) %*% b))) }
moments <- function(X, y, w) { w <- pmin(pmax(w, 1e-12), 1 - 1e-12)
  a <- w/sum(w); b <- (1-w)/sum(1-w)
  list(G1 = t(X*a)%*%X, G2 = t(X*b)%*%X, Z1 = t(X*a)%*%y, Z2 = t(X*b)%*%y,
       a = a, b = b) }
## minimizer of L
path_min <- function(X, y, w, gamma, ng = 401, ridge = 1e-6) {
  M <- moments(X, y, w)
  Gp <- M$G1 + M$G2; Ga <- M$G1 - M$G2
  Zp <- M$Z1 + M$Z2; Za <- M$Z1 - M$Z2
  bt <- function(th) { A <- Gp + th*Ga + ridge*diag(ncol(X))
    tryCatch(as.vector(solve(A, Zp + th*Za)), error = function(e) gsolve(A, Zp + th*Za)) }
  if (gamma <= 0) return(bt(0))
  th <- seq(-gamma, gamma, length.out = ng) ##theta in [-G, G]
  bt(th[which.min(sapply(th, function(t) Lobj(X, y, bt(t), w, gamma)))])
}
gamma_conv <- function(X, w) { M <- moments(X, rep(0, nrow(X)), w)
  Gp <- M$G1 + M$G2; Ga <- M$G1 - M$G2
  Rc <- chol(Gp + 1e-12*diag(ncol(X)))
  1 / max(abs(eigen(solve(t(Rc)) %*% Ga %*% solve(Rc), only.values = TRUE)$values)) }

GRID <- c(0.5, 1, 2, 5, 10, 20)

# ============================================================
say("================================================================\n")
say(" Methodology diagnostics, computed with the setup of 6-systematic_experiment.R\n")
#  centred, sd_y = 1 alpha_test = 1.5*alpha, sigmoid adversary, truncated learner
say("================================================================\n")

say("mean shift, alpha = 3, seed 1, Gamma = 5, at t = T = 30\n\n")
d <- gen_sem(250, 3, 4.5, "mean", 1)
f <- fit_acr_adv(d$X_tr, d$y_tr, 5)
w <- f$w
say(sprintf("  mean w        = %.3f   (Gaussian prediction 0.480)\n", mean(w)))
say(sprintf("  min  w        = %.3f   (Gaussian prediction 0.330)\n", min(w)))
say(sprintf("  max  w        = %.3f\n", max(w)))
say(sprintf("  P(w < 0.5)    = %.3f   (Gaussian prediction 0.683)\n", mean(w < 0.5)))
say(sprintf("  share in [0.33,0.60] = %.3f\n", mean(w >= 0.33 & w <= 0.60)))

# ---- convergence  ----
w2 <- rep(0.5, nrow(d$X_tr)); b2 <- rep(0, 6); prev <- NULL
tr <- data.frame()
for (t in 1:30) {
  b2 <- wls_learner(d$X_tr, d$y_tr, w2, 5); w2 <- adv_update(d$X_tr, d$y_tr, b2)
  tr <- rbind(tr, data.frame(t = t, L = Lobj(d$X_tr, d$y_tr, b2, w2, 5),
    step = if (is.null(prev)) NA else sqrt(sum((b2 - prev)^2)),
    s = sign(diff(rev(risks(as.vector(d$y_tr - d$X_tr %*% b2)^2, w2))))))
  prev <- b2
}
print(round(tr[tr$t %in% c(1,2,5,25,28,30), ], 4), row.names = FALSE)
capture.output(print(round(tr[tr$t %in% c(1,2,5,25,28,30), ], 4), row.names = FALSE), file = out)
say(sprintf("\n  mean step, iterations 2-10  = %.3f\n", mean(tr$step[2:10])))
say(sprintf("  mean step, iterations 21-30 = %.3f\n", mean(tr$step[21:30])))
say(sprintf("  L range over the last 10    = [%.3f, %.3f]\n",
            min(tr$L[21:30]), max(tr$L[21:30])))

say(sprintf("%6s %5s %6s | %10s | %s\n", "alpha", "seed", "shift", "Gamma_conv",
            "first Gamma where closed form != true min"))
gc_all <- c(); fail_all <- c()
for (sh in c("mean", "variance")) for (al in c(1, 3, 10)) for (sd_ in 1:4) {
  dd <- gen_sem(250, al, al*1.5, sh, sd_)
  ww <- fit_acr_adv(dd$X_tr, dd$y_tr, 5)$w
  gc_ <- gamma_conv(dd$X_tr, ww); gc_all <- c(gc_all, gc_)
  fail <- NA
  for (g in GRID) {
    bb <- wls_learner(dd$X_tr, dd$y_tr, ww, g)
    be <- path_min(dd$X_tr, dd$y_tr, ww, g)
    if (Lobj(dd$X_tr, dd$y_tr, bb, ww, g) > Lobj(dd$X_tr, dd$y_tr, be, ww, g) * 1.001) {
      fail <- g; break }
  }
  fail_all <- c(fail_all, fail)
  say(sprintf("%6d %5d %6s | %10.2f | %s\n", al, sd_, substr(sh,1,4), gc_,
              ifelse(is.na(fail), "none", fail)))
}
say(sprintf("\n  Gamma_conv range over the %d configurations: %.2f to %.2f\n",
            length(gc_all), min(gc_all), max(gc_all)))
say(sprintf("  first failing Gamma: %s\n",
            paste(sort(unique(na.omit(fail_all))), collapse = ", ")))
say(sprintf("  configurations with no failure on the grid: %d of %d\n",
            sum(is.na(fail_all)), length(fail_all)))

# worst gap between the closed form and the true minimum, at Gamma = 20
say("\n  Worst case on the grid (alpha = 1, mean shift, seed 1):\n")
d1 <- gen_sem(250, 1, 1.5, "mean", 1); w1 <- fit_acr_adv(d1$X_tr, d1$y_tr, 5)$w
for (g in GRID) {
  bb <- wls_learner(d1$X_tr, d1$y_tr, w1, g); be <- path_min(d1$X_tr, d1$y_tr, w1, g)
  say(sprintf("    Gamma=%5.1f  L(closed form)=%9.3f  L(true min)=%9.3f\n",
              g, Lobj(d1$X_tr, d1$y_tr, bb, w1, g), Lobj(d1$X_tr, d1$y_tr, be, w1, g)))
}
# ---- truncation  ----
say("mean shift, alpha = 1, seed 1, weights from fit_acr_adv\n\n")
M <- moments(d1$X_tr, d1$y_tr, w1); a <- M$a; b <- M$b; s <- 1
tf <- function(g) { v <- pmax((1 + g*s)*a + (1 - g*s)*b, 0)
  as.vector(solve(t(d1$X_tr * v) %*% d1$X_tr + 1e-6*diag(6), t(d1$X_tr * v) %*% d1$y_tr)) }
u <- pmax(s*(a - b), 0)
blim <- gsolve(t(d1$X_tr * u) %*% d1$X_tr, t(d1$X_tr * u) %*% d1$y_tr)
say(sprintf("%8s %10s %14s  %s\n", "Gamma", "frac v=0", "dist to limit", "beta"))
for (g in c(0, 1, 2, 5, 10, 20, 100, 1000)) {
  bt <- tf(g); v <- pmax((1 + g*s)*a + (1 - g*s)*b, 0)
  say(sprintf("%8g %10.3f %14.4f  %s\n", g, mean(v == 0),
              sqrt(sum((bt - blim)^2)), paste(sprintf("%6.2f", bt), collapse = " ")))
}
say(sprintf("%8s %10s %14.4f  %s\n", "limit", "-", 0, paste(sprintf("%6.2f", blim), collapse = " ")))

# ---- cen tering ----
say("mean shift, alpha = 1, alpha_test = 1.5, 10 seeds, Gamma = 5\n\n")
for (CEN in c(TRUE, FALSE)) {
  om <- am <- c()
  for (sd_ in 1:10) {
    dd <- gen_sem(250, 1, 1.5, "mean", sd_, CENTRE = CEN)
    ob <- as.vector(solve(crossprod(dd$X_tr) + 1e-8*diag(6), crossprod(dd$X_tr, dd$y_tr)))
    ab <- fit_acr_adv(dd$X_tr, dd$y_tr, 5)$beta
    om <- c(om, mean((dd$y_te - dd$X_te %*% ob)^2))
    am <- c(am, mean((dd$y_te - dd$X_te %*% ab)^2))
  }
  say(sprintf("  centred = %-5s : OLS = %7.3f   ACR = %7.3f   change = %+6.1f%%\n",
              CEN, mean(om), mean(am), 100*(mean(am) - mean(om))/mean(om)))
}
say("\nDone!!!\n")
close(out)
