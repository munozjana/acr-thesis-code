# ============================================================
#   Fig 1: DAG 
#   Fig 2: ICP over combinations of variables in 4 variable SEM
#   Fig 3: CR lambda path (coefficient path and rick gap and OOD risk)
#   Fig 4: Spurious variables (scatter plot, MSE plot)
#   Fig 5: Types of shifts
# ============================================================


## ── FIG 1 DAG PLOT ──────────────────────────────────────────────────────────
draw_node <- function(x, y, lab, fill = "white", r = 0.16, shaded = FALSE) {
  symbols(x, y, circles = r, inches = FALSE, add = TRUE,
          bg = fill, fg = if (shaded) "grey40" else "black",
          lwd = if (shaded) 1.2 else 1.6)
  text(x, y, lab, cex = 0.95, col = if (shaded) "grey25" else "black")
}
draw_edge <- function(x1, y1, x2, y2, r = 0.16, dashed = FALSE) {
  d <- sqrt((x2-x1)^2 + (y2-y1)^2)
  sx <- x1 + (x2-x1)*r/d; sy <- y1 + (y2-y1)*r/d
  ex <- x2 - (x2-x1)*r/d; ey <- y2 - (y2-y1)*r/d
  arrows(sx, sy, ex, ey, length = 0.09, lwd = 1.5,
         lty = if (dashed) 2 else 1, col = if (dashed) "grey40" else "black")
}
figC <- function() {
  par(mar = c(0.5, 0.5, 0.5, 0.5))
  plot(NA, xlim = c(-1.35, 2.6), ylim = c(-0.55, 2.45), axes = FALSE,
       xlab = "", ylab = "", asp = 1)
  cP <- "#DDEEDC"; cC <- "#F9DEDC"; cN <- "white"; cA <- "grey88"
  # edges first
  draw_edge(0,2, 1,2); draw_edge(1,2, 2,2)
  # X1 -> X3 as a curved arc above the top row (avoids passing through X2)
  tt <- seq(0, 1, length.out = 60)
  bx <- (1-tt)^2*0.10 + 2*tt*(1-tt)*1.0 + tt^2*1.90
  by <- (1-tt)^2*2.12 + 2*tt*(1-tt)*2.62 + tt^2*2.12
  lines(bx, by, lwd = 1.5)
  arrows(bx[58], by[58], bx[60], by[60], length = 0.09, lwd = 1.5)
  draw_edge(1,2, 1,1); draw_edge(2,2, 1,1)
  draw_edge(1,1, 0,0); draw_edge(1,1, 1,0)
  draw_edge(1,0, 2,0)
  draw_edge(-0.9,0.35, 0,0, dashed = TRUE)
  draw_edge(0.1,-0.5, 1,0, dashed = TRUE)   # wait, A5 placed below-left of X5
  # nodes
  draw_node(0,2, expression(X[1]), cN)
  draw_node(1,2, expression(X[2]), cP)
  draw_node(2,2, expression(X[3]), cP)
  draw_node(1,1, expression(Y),    cN)
  draw_node(0,0, expression(X[4]), cC)
  draw_node(1,0, expression(X[5]), cC)
  draw_node(2,0, expression(X[6]), cN)
  draw_node(-0.9,0.35, expression(A[4]^{(e)}), cA, shaded = TRUE)
  draw_node(0.1,-0.5,  expression(A[5]^{(e)}), cA, shaded = TRUE)
  legend("topright",
         legend = c("parents of Y", "children of Y", "environment shift"),
         fill = c(cP, cC, cA), bty = "n", cex = 0.85, border = "grey40")
}
pdf("fig_dag.pdf", width = 6.4, height = 4.9); figC(); dev.off()



set.seed(1)

## ── small SEM (ancestor X1, parents X2 X3, child X4) ──
gen_small <- function(n, shift_on = c("parents", "child"), alpha, seed) {
  # env with alpha = 0 is observational; alpha > 0 applies noise
  # interventions (mean + variance) to the parents' equations, or a mean
  # shift to the child, depending on shift_on.
  set.seed(seed)
  shift_on <- match.arg(shift_on)
  int_par <- (shift_on == "parents" && alpha > 0)
  a4 <- if (shift_on == "child") alpha else 0
  X1 <- rnorm(n)
  X2 <- X1 + (if (int_par) rnorm(n, alpha, 3) else rnorm(n))
  X3 <- X1 + X2 + (if (int_par) rnorm(n, alpha, 3) else rnorm(n))
  Y  <- X2 + X3 + rnorm(n)
  X4 <- Y + a4 + rnorm(n, 0, 1.5)
  data.frame(X1, X2, X3, X4, Y)
}

## ── FIG 2 PLOT ICP over all 16 subsets ─────────────────────────────────────────
icp_pvalue <- function(S, d1, d2) {
  d  <- rbind(d1, d2); env <- rep(1:2, c(nrow(d1), nrow(d2)))
  if (length(S) == 0) r <- d$Y - mean(d$Y)
  else r <- resid(lm(Y ~ ., data = d[, c(S, "Y"), drop = FALSE]))
  p_t <- t.test(r[env == 1], r[env == 2])$p.value
  p_f <- var.test(r[env == 1], r[env == 2])$p.value
  min(1, 2 * min(p_t, p_f))          # Bonferroni combination
}
vars <- c("X1", "X2", "X3", "X4")
subsets <- unlist(lapply(0:4, function(k) combn(vars, k, simplify = FALSE)),
                  recursive = FALSE)
sub_lab <- sapply(subsets, function(S)
  if (length(S) == 0) "{ }" else paste(sub("X", "", S), collapse = ""))

run_icp <- function(shift_on, alpha = 2, n = 2000, seed = 11) {
  d1 <- gen_small(n, shift_on, 0,     seed)
  d2 <- gen_small(n, shift_on, alpha, seed + 1)
  sapply(subsets, icp_pvalue, d1 = d1, d2 = d2)
}
p_anc  <- run_icp("parents")
p_chld <- run_icp("child", alpha = 4)
acc_anc  <- p_anc  > 0.05
acc_chld <- p_chld > 0.05
inter <- function(acc) {
  A <- subsets[acc]
  if (!length(A)) return(character(0))
  Reduce(intersect, A)
}
cat("parent-intervention accepted:", paste(sub_lab[acc_anc], collapse = " "), "\n")
cat("  intersection:", paste(inter(acc_anc), collapse = ","), "\n")
cat("child-shift accepted:", paste(sub_lab[acc_chld], collapse = " "), "\n")
cat("  intersection:", if (length(inter(acc_chld))) paste(inter(acc_chld), collapse=",") else "(empty)", "\n")

figE <- function() {
  par(mfrow = c(2, 1), mar = c(3.6, 4.2, 2.4, 1), mgp = c(2.3, 0.7, 0))
  for (panel in 1:2) {
    pv  <- if (panel == 1) p_anc else p_chld
    acc <- pv > 0.05
    ttl <- if (panel == 1)
      "(a) Intervention on the parents: ICP returns {X2, X3}"
    else
      "(b) Shift on the child X4 only: ICP returns the empty set"
    cols <- ifelse(acc, "#2E8B57", "grey75")
    bp <- barplot(pmax(pv, 1e-4), log = "y", col = cols, border = NA,
                  names.arg = NA, axes = FALSE,
                  ylab = "", main = ttl, cex.main = 1.0,
                  ylim = c(2e-5, 1.5))

    axis(2, at = 10^(0:-4), labels = c("1e+00", "1e-01", "1e-02", "1e-03", "1e-04"),
         las = 1, cex.axis = 0.8)
    mtext("invariance p-value", side = 2, line = 4.0, cex = 0.9)

    axis(1, at = bp, labels = sub_lab, las = 1, cex.axis = 0.72,
         tick = FALSE, line = -0.6)
    mtext("candidate subset of {X1, X2, X3, X4}", side = 1, line = 1.9, cex = 0.8)

    abline(h = 0.05, lty = 2, col = "grey40")
    mtext(expression(alpha ~ "=" ~ 0.05), side = 4, at = 0.05, las = 1,
          line = 0.4, cex = 0.78, col = "grey30")
    box(col = "grey60")
  }
  par(fig = c(0, 1, 0, 1), oma = c(0, 0, 0, 0), mar = c(0, 0, 0, 0), new = TRUE)
  plot(0, 0, type = "n", axes = FALSE, xlab = "", ylab = "")
  legend("bottom", horiz = TRUE, inset = c(0, 0.012),
         legend = c("accepted (p > 0.05)", "rejected (p-value floored at 1e-4)"),
         fill = c("#2E8B57", "grey75"), border = NA, bty = "n", cex = 0.82,
         xpd = NA)
}
pdf("fig_13_icp.pdf", width = 9.5, height = 6.6); figE(); dev.off()

## ── FIG 3 causal regularization lambda path ───────────────────────────────
# Fixed-environment CR, WRD solve WITHOUT clipping, allowing negative weights.
cr_fit <- function(X, y, env, lambda) {
  # Worst-Risk Decomposition
  # solve both sign branches in closed form, return the sign-consistent solution (or one with the lower CR objective).
  n1 <- sum(env == 1); n2 <- sum(env == 2)
  w1 <- ifelse(env == 1, 1/n1, 0); w2 <- ifelse(env == 2, 1/n2, 0)
  obj <- function(beta) {
    r <- as.vector(y - X %*% beta)
    R1 <- sum(w1 * r^2); R2 <- sum(w2 * r^2)
    list(val = (R1 + R2) + lambda * abs(R1 - R2), s = sign(R1 - R2))
  }
  cand <- lapply(c(-1, 1), function(s) {
    v <- (1 + lambda * s) * w1 + (1 - lambda * s) * w2
    A <- t(X * v) %*% X + 1e-8 * diag(ncol(X))
    beta <- tryCatch(as.vector(solve(A, t(X * v) %*% y)),
                     error = function(e) rep(NA, ncol(X)))
    if (anyNA(beta)) return(NULL)
    o <- obj(beta); list(beta = beta, consistent = (o$s == s), val = o$val)
  })
  cand <- Filter(Negate(is.null), cand)
  cons <- Filter(function(c) c$consistent, cand)
  pick <- if (length(cons) == 1) cons[[1]]
          else cand[[which.min(sapply(cand, `[[`, "val"))]]
  pick$beta
}
# Full thesis SEM, mean shift on children between the two training envs
gen_env6 <- function(n, alpha, seed) {
  set.seed(seed)
  X1 <- rnorm(n); X2 <- X1 + rnorm(n); X3 <- X1 + X2 + rnorm(n)
  Y  <- X2 + X3 + rnorm(n)
  X4 <- Y + rnorm(n, alpha, 1) + rnorm(n, 0, 0.5)
  X5 <- Y + rnorm(n, alpha, 1) + rnorm(n, 0, 0.5)
  X6 <- X5 + rnorm(n, 0, 0.5)
  list(X = cbind(X1, X2, X3, X4, X5, X6), Y = Y)
}
lambdas <- c(0, 0.25, 0.5, 0.75, 1, 1.5, 2)
seeds <- 1:20
coef_arr <- array(NA, c(length(lambdas), 6, length(seeds)))
gap  <- matrix(NA, length(lambdas), length(seeds))
ood  <- matrix(NA, length(lambdas), length(seeds))
for (si in seq_along(seeds)) {
  e1 <- gen_env6(500, 0, seeds[si]); e2 <- gen_env6(500, 3, seeds[si] + 500)
  X <- rbind(e1$X, e2$X); y <- c(e1$Y, e2$Y); env <- rep(1:2, each = 500)
  te <- gen_env6(2000, 6, seeds[si] + 9000)
  for (li in seq_along(lambdas)) {
    b <- cr_fit(X, y, env, lambdas[li])
    coef_arr[li, , si] <- b
    r <- as.vector(y - X %*% b)
    gap[li, si] <- abs(mean(r[env == 1]^2) - mean(r[env == 2]^2))
    ood[li, si] <- mean((te$Y - te$X %*% b)^2)
  }
}
cm <- apply(coef_arr, 1:2, mean)
cat("\nlambda-path (mean over 20 seeds):\n")
for (li in seq_along(lambdas))
  cat(sprintf("lam=%5.2f  b2=%.2f b3=%.2f b4=%.2f b5=%.2f  gap=%.3f  ood=%.2f\n",
      lambdas[li], cm[li,2], cm[li,3], cm[li,4], cm[li,5],
      mean(gap[li,]), mean(ood[li,])))

figF <- function() {
  par(mfrow = c(1, 2), mar = c(4.2, 4.2, 2.8, 1), mgp = c(2.4, 0.7, 0))
  lx <- lambdas + 0.05                       # shift for log axis (lambda=0)
  # (a) coefficient paths
  matplot(lx, cm[, c(2,3,4,5)], type = "b", log = "x", lwd = 2, lty = 1,
          pch = c(17,17,19,19), col = c("#2E8B57","#7BC47F","#3B7DD8","#9AC1EA"),
          xlab = expression(lambda ~ "(log scale," ~ lambda==0 ~ "at left edge)"),
          ylab = expression(hat(beta)[j]), main = "(a) Coefficient path",
          xaxt = "n")
  axis(1, at = lambdas + 0.05, labels = lambdas)
  abline(h = c(0, 1), lty = 3, col = "grey55")
  legend("right", c(expression(X[2] ~ "(parent)"), expression(X[3] ~ "(parent)"),
                    expression(X[4] ~ "(child)"),  expression(X[5] ~ "(child)")),
         col = c("#2E8B57","#7BC47F","#3B7DD8","#9AC1EA"),
         pch = c(17,17,19,19), lwd = 2, bty = "n", cex = 0.82)
  # (b) risk gap and OOD risk
  plot(lx, apply(gap, 1, mean), type = "b", log = "xy", pch = 19, lwd = 2,
       col = "#B23A48", xaxt = "n",
       ylim = range(c(gap, ood)) * c(0.8, 1.2),
       xlab = expression(lambda ~ "(log scale)"),
       ylab = "value (log scale)", main = "(b) Risk gap and OOD risk")
  lines(lx, apply(ood, 1, mean), type = "b", pch = 15, lwd = 2, col = "#E2711D")
  axis(1, at = lambdas + 0.05, labels = lambdas)
  legend("bottomleft",
         c(expression(Delta * R ~ "(training risk gap)"),
           expression("OOD MSE (" * alpha[test] == 6 * ")")),
         col = c("#B23A48", "#E2711D"), pch = c(19, 15), lwd = 2,
         bty = "n", cex = 0.85)
}
pdf("fig_13_crpath.pdf", width = 9.5, height = 4); figF(); dev.off()


## ── FIG 4 causal vs spurious ───────────────────────────────

set.seed(1)

# x1 causes y. x2 is a noisy copy of y in TRAINING only (spurious, like a
# child of Y in the SEM). At test time x2 is independent noise.
run_once <- function(seed, n = 200) {
  set.seed(seed)
  x1 <- rnorm(n); y <- x1 + rnorm(n)
  x2 <- y + rnorm(n, 0, 0.5)                 # spurious: correlated via y
  x1t <- rnorm(n); yt <- x1t + rnorm(n)
  x2t <- rnorm(n, 0, sd(x2))                 # test: same scale, independent
  f_both   <- lm(y ~ x1 + x2)
  f_causal <- lm(y ~ x1)
  c(b1 = unname(coef(f_both)["x1"]), b2 = unname(coef(f_both)["x2"]),
    tr_both   = mean(resid(f_both)^2),
    te_both   = mean((yt - predict(f_both,   data.frame(x1=x1t, x2=x2t)))^2),
    tr_causal = mean(resid(f_causal)^2),
    te_causal = mean((yt - predict(f_causal, data.frame(x1=x1t)))^2))
}
res <- t(sapply(1:100, run_once))
m <- colMeans(res)
cat(sprintf("coef x1 = %.2f | coef x2 = %.2f\n", m["b1"], m["b2"]))
cat(sprintf("ERM both   : train %.2f  test %.2f\n", m["tr_both"], m["te_both"]))
cat(sprintf("causal only: train %.2f  test %.2f\n", m["tr_causal"], m["te_causal"]))

## ── spurious experiment ──────────────────────────────────────────
set.seed(7)
n <- 200
x1 <- rnorm(n); y <- x1 + rnorm(n); x2 <- y + rnorm(n, 0, 0.5)
x1t <- rnorm(n); yt <- x1t + rnorm(n); x2t <- rnorm(n, 0, sd(x2))

figB <- function() {
  par(mfrow = c(1, 2), mar = c(4.2, 4.2, 2.6, 1), mgp = c(2.4, 0.7, 0))
  # left: x2 vs y, train vs test
  plot(x2, y, pch = 19, col = adjustcolor("#3B7DD8", 0.55), cex = 0.7,
       xlab = expression(x[2]), ylab = "y",
       main = expression("The spurious feature " * x[2]))
  points(x2t, yt, pch = 17, col = adjustcolor("#E2711D", 0.55), cex = 0.7)
  legend("topleft", c("training (correlated)", "test (independent)"),
         pch = c(19, 17), col = c("#3B7DD8", "#E2711D"), bty = "n", cex = 0.85)
  # right: MSE bars
  bars <- matrix(c(m["tr_both"], m["te_both"], m["tr_causal"], m["te_causal"]),
                 nrow = 2)
  bp <- barplot(bars, beside = TRUE, names.arg = c("ERM (x1 + x2)", "causal (x1 only)"),
                col = c("#3B7DD8", "#E2711D"), ylim = c(0, max(bars) * 1.25),
                ylab = "MSE", main = "Average over 100 repetitions")
  text(bp, bars + max(bars) * 0.05, sprintf("%.2f", bars), cex = 0.85)
  legend("topleft", c("train", "test"), fill = c("#3B7DD8", "#E2711D"),
         bty = "n", cex = 0.85)
}
pdf("fig_spurious.pdf", width = 9.5, height = 4); figB(); dev.off()


## ── FIG 5 shift taxonomy ───────────────────────────────────────────────
figA <- function() {
  par(mfrow = c(1, 3), mar = c(4.2, 4.2, 2.8, 1), mgp = c(2.4, 0.7, 0))
  set.seed(3)
  # (a) covariate shift: same line, different x regions
  xa <- rnorm(120, -1, 0.6); ya <- 1 + 0.8 * xa + rnorm(120, 0, 0.4)
  xb <- rnorm(120, 1.6, 0.6); yb <- 1 + 0.8 * xb + rnorm(120, 0, 0.4)
  plot(xa, ya, xlim = range(c(xa, xb)), ylim = range(c(ya, yb)),
       pch = 19, col = adjustcolor("#3B7DD8", 0.5), cex = 0.7,
       xlab = "x", ylab = "y", main = "(a) Covariate shift")
  points(xb, yb, pch = 17, col = adjustcolor("#E2711D", 0.5), cex = 0.7)
  abline(1, 0.8, lwd = 2)
  legend("topleft", c("train", "test"), pch = c(19, 17),
         col = c("#3B7DD8", "#E2711D"), bty = "n", cex = 0.9)
  # (b) label shift: P(y) changes, P(x|y) fixed
  set.seed(4)
  p_tr <- c(0.8, 0.2); p_te <- c(0.2, 0.8)
  bp <- barplot(rbind(p_tr, p_te), beside = TRUE,
                names.arg = c("y = 0", "y = 1"), ylim = c(0, 1),
                col = c("#3B7DD8", "#E2711D"), ylab = "P(y)",
                main = "(b) Label shift")
  legend("topright", c("train", "test"), fill = c("#3B7DD8", "#E2711D"),
         bty = "n", cex = 0.9)
  text(mean(bp), 0.95, expression(P(x ~ "|" ~ y) ~ "unchanged"), cex = 0.9)
  # (c) mechanism shift: the conditional itself changes
  set.seed(5)
  xc <- rnorm(120, 0, 0.9); yc <- 1 + 0.8 * xc + rnorm(120, 0, 0.4)
  xd <- rnorm(120, 0, 0.9); yd <- 1 - 0.8 * xd + rnorm(120, 0, 0.4)
  plot(xc, yc, xlim = range(c(xc, xd)), ylim = range(c(yc, yd)),
       pch = 19, col = adjustcolor("#3B7DD8", 0.5), cex = 0.7,
       xlab = "x", ylab = "y", main = "(c) Mechanism shift")
  points(xd, yd, pch = 17, col = adjustcolor("#E2711D", 0.5), cex = 0.7)
  abline(1, 0.8, lwd = 2, col = "#3B7DD8")
  abline(1, -0.8, lwd = 2, col = "#E2711D")
}
pdf("fig_taxonomy.pdf", width = 12, height = 3.9); figA(); dev.off()
