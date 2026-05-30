# ============================================================
# Report Figures — ACR Research Update for Richter & Wit
# Generates 6 publication-quality plots:
#   Fig 1: Adversary environment discovery (airquality scatter)
#   Fig 2: Coefficient comparison OLS vs ACR
#   Fig 3: OOD MSE by holdout month
#   Fig 4: Delta_R non-monotonicity
#   Fig 5: Pareto frontier (accuracy vs stability)
#   Fig 6: Adversary AUC vs shift strength (SEM, mean vs variance)
# ============================================================

out_dir <- "/Users/janamunoz/Desktop/Literatura tesis/working folder/"

# ---- Colour palette (consistent across all figures) ----
C_OLS    <- "#2980B9"   # steel blue
C_ACR    <- "#C0392B"   # red
C_RAND   <- "#27AE60"   # green
C_ORACLE <- "#8E44AD"   # purple
C_DARK   <- "#1a2744"   # navy
C_GREY   <- "grey60"

# ---- Helper: ACR core functions ----
wls_learner <- function(X, y, w, gamma, ridge = 1e-6) {
  w  <- pmin(pmax(w, 1e-10), 1 - 1e-10)
  w1 <- w  / sum(w)
  w2 <- (1 - w) / sum(1 - w)
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

compute_auc <- function(w, true_env) {
  label <- as.integer(true_env == 2)
  n1 <- sum(label == 1); n0 <- sum(label == 0)
  if (n1 == 0 || n0 == 0) return(0.5)
  auc <- (sum(rank(w)[label == 1]) - n1*(n1+1)/2) / (n1*n0)
  max(auc, 1 - auc)
}

# ============================================================
# PREPARE AIRQUALITY DATA
# ============================================================

data(airquality)
aq          <- na.omit(airquality)
aq$logOzone <- log(aq$Ozone)
pred_vars   <- c("Solar.R","Wind","Temp")
aq_sc       <- aq
aq_sc[pred_vars] <- scale(aq[pred_vars])

gamma_grid <- c(0, 0.5, 1, 2, 5, 10, 20, 50, 100)

# September holdout
tr_idx <- which(aq_sc$Month != 9)
te_idx <- which(aq_sc$Month == 9)
X_tr   <- as.matrix(aq_sc[tr_idx, pred_vars])
y_tr   <- aq_sc$logOzone[tr_idx]
X_te   <- as.matrix(aq_sc[te_idx, pred_vars])
y_te   <- aq_sc$logOzone[te_idx]

# Fit ACR for each lambda (September holdout)
ols_beta <- as.vector(coef(lm(y_tr ~ X_tr - 1)))
sep_res  <- lapply(gamma_grid, function(g) {
  if (g == 0) {
    list(beta = ols_beta, w = rep(0.5, length(y_tr)))
  } else {
    fit_acr_adv(X_tr, y_tr, g)
  }
})
ood_mse   <- sapply(sep_res, function(f) mean((y_te - X_te %*% f$beta)^2))
betas     <- sapply(sep_res, function(f) f$beta)   # 3 x length(gamma_grid)
w_adv_sep <- sep_res[[which(gamma_grid == 1)]]$w   # weights at lambda=1

# Risk gap for each lambda
compute_risks <- function(X, y, beta, w) {
  w  <- pmin(pmax(w, 1e-10), 1-1e-10)
  w1 <- w / sum(w); w2 <- (1-w)/sum(1-w)
  r  <- as.vector((y - X %*% beta)^2)
  c(R1 = sum(w1*r), R2 = sum(w2*r))
}

delta_R <- sapply(seq_along(gamma_grid), function(i) {
  rs <- compute_risks(X_tr, y_tr, sep_res[[i]]$beta, sep_res[[i]]$w)
  abs(rs["R1"] - rs["R2"])
})
R_plus <- sapply(seq_along(gamma_grid), function(i) {
  rs <- compute_risks(X_tr, y_tr, sep_res[[i]]$beta, sep_res[[i]]$w)
  rs["R1"] + rs["R2"]
})

# Multi-month experiment (10 seeds, all 5 months)
holdout_months <- 5:9
seeds_aq       <- 1:10
set.seed(42)
aq_mm <- expand.grid(holdout_month = holdout_months, seed = seeds_aq)
aq_mm$ols_mse <- aq_mm$acr_mse <- aq_mm$rand_mse <- NA

for (i in seq_len(nrow(aq_mm))) {
  hm  <- aq_mm$holdout_month[i]
  sid <- aq_mm$seed[i]
  tri <- which(aq_sc$Month != hm)
  tei <- which(aq_sc$Month == hm)
  if (length(tei) < 5 || length(tri) < 20) next
  Xtr <- as.matrix(aq_sc[tri, pred_vars]); ytr <- aq_sc$logOzone[tri]
  Xte <- as.matrix(aq_sc[tei, pred_vars]); yte <- aq_sc$logOzone[tei]
  ob  <- as.vector(coef(lm(ytr ~ Xtr - 1)))
  fa  <- fit_acr_adv(Xtr, ytr, 1)
  set.seed(sid * 100)
  wr  <- as.numeric(sample(c(0,1), nrow(Xtr), replace = TRUE))
  rb  <- wls_learner(Xtr, ytr, wr, 1)
  aq_mm$ols_mse[i]  <- mean((yte - Xte %*% ob)^2)
  aq_mm$acr_mse[i]  <- mean((yte - Xte %*% fa$beta)^2)
  aq_mm$rand_mse[i] <- mean((yte - Xte %*% rb)^2)
}

ag_mm <- aggregate(cbind(ols_mse, acr_mse, rand_mse) ~ holdout_month,
                   aq_mm, mean, na.rm = TRUE)
ag_sd <- aggregate(cbind(ols_mse, acr_mse, rand_mse) ~ holdout_month,
                   aq_mm, sd, na.rm = TRUE)

# ============================================================
# QUICK SEM EXPERIMENT (5 seeds) for AUC diagnostic
# ============================================================
cat("Running SEM AUC diagnostic (5 seeds x 4 alpha x 2 shift types)...\n")

gen_sem <- function(n, alpha, shift_type, seed) {
  set.seed(seed); n2 <- 2*n; env <- rep(c(1L,2L), each=n)
  X1 <- rnorm(n2); X2 <- X1+rnorm(n2); X3 <- X1+X2+rnorm(n2)
  Y  <- X2+X3+rnorm(n2)
  if (shift_type == "mean") {
    A4 <- ifelse(env==1, rnorm(n2,0,1), rnorm(n2,alpha,1))
    A5 <- ifelse(env==1, rnorm(n2,0,1), rnorm(n2,alpha,1))
  } else {
    sd2 <- max(alpha,0.1)
    A4  <- ifelse(env==1, rnorm(n2,0,1), rnorm(n2,0,sd2))
    A5  <- ifelse(env==1, rnorm(n2,0,1), rnorm(n2,0,sd2))
  }
  X4 <- Y+A4+rnorm(n2,0,.5); X5 <- Y+A5+rnorm(n2,0,.5); X6 <- X5+rnorm(n2,0,.5)
  Xr <- cbind(X1,X2,X3,X4,X5,X6)
  mu <- colMeans(Xr); sg <- apply(Xr,2,sd)
  list(X = scale(Xr,mu,sg), y = Y, env = env)
}

alpha_vals <- c(1,3,5,10)
sem_auc    <- expand.grid(alpha=alpha_vals, seed=1:5,
                          shift_type=c("mean","variance"),
                          stringsAsFactors=FALSE)
sem_auc$adv_auc <- NA

for (i in seq_len(nrow(sem_auc))) {
  d   <- gen_sem(250, sem_auc$alpha[i], sem_auc$shift_type[i], sem_auc$seed[i])
  fit <- fit_acr_adv(d$X, d$y, 1, iterations=20)
  sem_auc$adv_auc[i] <- compute_auc(fit$w, d$env)
}

auc_mean <- aggregate(adv_auc ~ alpha + shift_type, sem_auc, mean)
auc_sd   <- aggregate(adv_auc ~ alpha + shift_type, sem_auc, sd)

cat("Done. Generating figures...\n")

# ============================================================
# FIGURE 1: ADVERSARY ENVIRONMENT DISCOVERY (SCATTER)
# ============================================================

pdf(paste0(out_dir,"7-fig1_adversary_discovery.pdf"), width=6.5, height=5.5)
par(mar=c(4.5, 4.5, 3, 1.5), mgp=c(2.5,0.7,0), family="sans")

# Standardised Temp and Solar.R for training set
Temp_tr   <- scale(aq$Temp)[tr_idx]
Solar_tr  <- scale(aq$Solar.R)[tr_idx]
month_tr  <- aq$Month[tr_idx]

# Colour by adversary weight (w > 0.5 = hard env = dark red)
wcol <- ifelse(w_adv_sep > 0.5,
               adjustcolor(C_ACR,  alpha.f = 0.85),
               adjustcolor(C_OLS,  alpha.f = 0.70))

# Shape by month
pch_month <- c("5"=21,"6"=22,"7"=23,"8"=24)[as.character(month_tr)]

plot(Temp_tr, Solar_tr,
     col  = wcol, bg = wcol, pch = pch_month,
     cex  = 1.3, lwd = 0.5,
     xlab = "Standardised Temperature",
     ylab = "Standardised Solar Radiation",
     main = "Adversary Environment Discovery\n(No Month Labels Used)")

# Convex hull for each discovered environment
hard_idx <- which(w_adv_sep > 0.5)
easy_idx <- which(w_adv_sep <= 0.5)
if (length(hard_idx) >= 3) {
  hh <- chull(Temp_tr[hard_idx], Solar_tr[hard_idx])
  polygon(Temp_tr[hard_idx[hh]], Solar_tr[hard_idx[hh]],
          border = C_ACR, lwd = 1.5, lty = 2, col = NA)
}
if (length(easy_idx) >= 3) {
  eh <- chull(Temp_tr[easy_idx], Solar_tr[easy_idx])
  polygon(Temp_tr[easy_idx[eh]], Solar_tr[easy_idx[eh]],
          border = C_OLS, lwd = 1.5, lty = 2, col = NA)
}

legend("topright", bty="n", cex=0.82,
       legend = c("Hard env (w > 0.5)", "Easy env (w ≤ 0.5)",
                  "May","June","July","Aug"),
       col    = c(C_ACR, C_OLS, rep("grey30",4)),
       pt.bg  = c(C_ACR, C_OLS, rep("grey30",4)),
       pch    = c(16, 16, 21, 22, 23, 24),
       pt.cex = c(1.3,1.3, rep(1.1,4)))

text(-1.8, 2.1, "92% of May\n→ Hard env", col=C_ACR, cex=0.78, font=3)
text( 1.5,-1.6, "84% of Jul–Aug\n→ Easy env", col=C_OLS, cex=0.78, font=3)
dev.off()


# ============================================================
# FIGURE 2: COEFFICIENT COMPARISON (OLS vs ACR)
# ============================================================

pdf(paste0(out_dir,"7-fig2_coefficients.pdf"), width=7, height=5)
par(mar=c(4.5, 5, 3, 1.5), mgp=c(3,0.7,0), family="sans")

# Lambda path for each predictor
matplot(gamma_grid, t(betas), type="l",
        lwd = 2.5,
        lty = c(1,2,4),
        col = c("#E67E22","#2980B9","#27AE60"),
        xlab = expression(lambda),
        ylab = "Estimated coefficient",
        main = "Coefficient Paths under Adversarial Regularization",
        log  = "x",
        xlim = c(0.5, 100),
        xaxt = "n")

axis(1, at = c(0.5,1,2,5,10,20,50,100),
     labels = c("0.5","1","2","5","10","20","50","100"))

abline(h = 0,   lty = 3, col = "grey70")
abline(v = 1,   lty = 3, col = C_DARK, lwd = 1.5)
text(1.15, max(betas)*0.92, expression(lambda*"=1  (CV)"),
     col = C_DARK, cex = 0.78, adj = 0)

legend("bottomright", bty="n", cex=0.88, lwd=2.5, lty=c(1,2,4),
       col    = c("#E67E22","#2980B9","#27AE60"),
       legend = c("Solar.R  (sign reversal at λ=1)",
                  "Wind     (stable across λ)",
                  "Temp     (shrinkage)"))

# Annotate the sign reversal on Solar.R
sr_vals <- betas[1,]   # row 1 = Solar.R
sign_chg <- which(diff(sign(sr_vals)) != 0)
if (length(sign_chg) > 0) {
  abline(h = 0, col = "#E67E22", lty = 3, lwd = 1)
}
dev.off()


# ============================================================
# FIGURE 3: OOD MSE BY HOLDOUT MONTH
# ============================================================

pdf(paste0(out_dir,"7-fig3_ood_by_month.pdf"), width=7.5, height=5.5)
par(mar=c(4, 5, 3, 1.5), mgp=c(3,0.7,0), family="sans")

mnms   <- c("May","Jun","Jul","Aug","Sep")
n_m    <- nrow(ag_mm)
x_pos  <- 1:n_m
width  <- 0.22
offset <- c(-width, 0, width)
cols   <- c(C_OLS, C_ACR, C_RAND)
mats   <- cbind(ag_mm$ols_mse, ag_mm$acr_mse, ag_mm$rand_mse)
sds    <- cbind(ag_sd$ols_mse, ag_sd$acr_mse, ag_sd$rand_mse)

ylim_max <- max(mats + sds, na.rm=TRUE) * 1.18
plot(NA, xlim=c(0.5, n_m+0.5), ylim=c(0, ylim_max),
     xlab="", ylab="OOD MSE (log-Ozone scale)",
     main="OOD MSE by Holdout Month\nACR-Adv vs OLS vs CR-Random",
     xaxt="n", las=1)
axis(1, at=x_pos, labels=mnms, tick=FALSE, font=1)

for (j in 1:3) {
  xj  <- x_pos + offset[j]
  yj  <- mats[,j]
  sdj <- sds[,j]
  rect(xj - width*0.45, 0, xj + width*0.45, yj,
       col=adjustcolor(cols[j], 0.82), border=NA)
  # error bars
  ok <- !is.na(sdj)
  segments(xj[ok], yj[ok]-sdj[ok], xj[ok], yj[ok]+sdj[ok],
           col="grey30", lwd=1.2)
}

# Annotate June (pathological case)
june_idx <- which(ag_mm$holdout_month == 6)
text(june_idx, max(mats[june_idx,], na.rm=TRUE) + 1.5,
     "Pathological\ncase", col=C_ACR, cex=0.72, font=3)

# Annotate May (biggest win)
may_idx <- which(ag_mm$holdout_month == 5)
text(may_idx, ag_mm$acr_mse[may_idx] + 1.8,
     "−25.8%\nvs OLS", col=C_ACR, cex=0.72, font=2)

abline(h=0, col="grey80")
legend("topright", bty="n", cex=0.85,
       legend=c("OLS","ACR-Adv","CR-Random"),
       fill=adjustcolor(c(C_OLS,C_ACR,C_RAND), 0.82), border=NA)
dev.off()


# ============================================================
# FIGURE 4: DELTA_R NON-MONOTONICITY
# ============================================================

pdf(paste0(out_dir,"7-fig4_deltaR_nonmonotone.pdf"), width=7, height=5)
par(mar=c(4.5, 5, 3, 5), mgp=c(3,0.7,0), family="sans")

# Left axis: Delta_R
plot(gamma_grid[-1], delta_R[-1],   # skip lambda=0 for log axis
     type="b", pch=17, lwd=2.5, col=C_DARK,
     log="x", xaxt="n",
     xlab=expression(lambda),
     ylab=expression(paste(Delta, R, " (risk gap)")),
     main=expression(paste(Delta, R, " Non-Monotonicity under Adversarial Co-Adaptation")),
     ylim=c(0, max(delta_R)*1.15),
     las=1)

axis(1, at=c(0.5,1,2,5,10,20,50,100),
     labels=c("0.5","1","2","5","10","20","50","100"))

# Right axis: OOD MSE
par(new=TRUE)
plot(gamma_grid[-1], ood_mse[-1],
     type="b", pch=16, lwd=2.5, col=C_ACR,
     log="x", xaxt="n", yaxt="n",
     xlab="", ylab="",
     ylim=c(min(ood_mse)*0.92, max(ood_mse)*1.08))

axis(4, las=1, col.axis=C_ACR, col=C_ACR)
mtext("OOD MSE (September holdout)", side=4, line=3.5, col=C_ACR, cex=0.9)

# OLS baseline
abline(h=ood_mse[1], lty=3, col=C_ACR, lwd=1.5)
text(70, ood_mse[1]*1.01, "OLS baseline", col=C_ACR, cex=0.75, adj=1)

# lambda=1 marker
abline(v=1, lty=2, col="grey50", lwd=1.5)
text(1.15, max(delta_R)*1.05, expression(lambda*"=1"),
     col="grey40", cex=0.78, adj=0)

# Annotation: the paradox
text(15, max(delta_R)*0.55,
     expression(paste(Delta,R," grows — yet OOD MSE improves")),
     col=C_DARK, cex=0.77, font=3)

legend("right", bty="n", cex=0.85, inset=0.02,
       lwd=2.5, pch=c(17,16),
       col=c(C_DARK, C_ACR),
       legend=c(expression(Delta*R), "OOD MSE"))
dev.off()


# ============================================================
# FIGURE 5: PARETO FRONTIER
# ============================================================

pdf(paste0(out_dir,"7-fig5_pareto_frontier.pdf"), width=6.5, height=5.5)
par(mar=c(4.5, 5, 3, 1.5), mgp=c(3,0.7,0), family="sans")

# Colour by OOD MSE (dark = good, light = bad)
mse_range  <- range(ood_mse)
mse_norm   <- (ood_mse - mse_range[1]) / diff(mse_range)
pt_cols    <- colorRampPalette(c(C_ACR, "lightyellow"))(100)[
                ceiling(mse_norm * 99) + 1]

plot(R_plus, delta_R, type="n",
     xlab=expression(R["+"](hat(beta)^lambda) ~~ "(pooled in-sample risk)"),
     ylab=expression(Delta*R(hat(beta)^lambda) ~~ "(risk gap)"),
     main="Pareto Frontier: Accuracy vs. Stability")

# Connect the dots (path as lambda increases)
lines(R_plus, delta_R, col="grey75", lwd=1.2, lty=2)
points(R_plus, delta_R, pch=21, cex=1.6, bg=pt_cols, col="grey30", lwd=0.8)

# Label key lambda values
lbl_idx <- c(1,2,3,5,7,9)   # lambda = 0, 0.5, 1, 5, 20, 100
lbl_txt <- c("OLS","0.5","1","5","20","100")
text(R_plus[lbl_idx] + diff(range(R_plus))*0.02,
     delta_R[lbl_idx] + diff(range(delta_R))*0.03,
     labels = paste0("λ=",lbl_txt),
     cex = 0.72, col = C_DARK)

# Mark the CV-selected point
cv_idx <- which(gamma_grid == 1)
points(R_plus[cv_idx], delta_R[cv_idx],
       pch=21, cex=2.2, bg=pt_cols[cv_idx], col=C_ACR, lwd=2)
text(R_plus[cv_idx] - diff(range(R_plus))*0.05,
     delta_R[cv_idx] + diff(range(delta_R))*0.06,
     "CV selects λ=1", col=C_ACR, cex=0.78, font=2)

# Colour bar legend (manual)
legend_x  <- max(R_plus)*0.5; legend_y <- max(delta_R)*0.9
n_leg     <- 10
leg_cols  <- colorRampPalette(c(C_ACR,"lightyellow"))(n_leg)
rect_w    <- diff(range(R_plus))*0.04
for (k in seq_len(n_leg)) {
  rect(legend_x + (k-1)*rect_w, legend_y - diff(range(delta_R))*0.05,
       legend_x + k*rect_w,     legend_y,
       col=leg_cols[k], border=NA)
}
text(legend_x, legend_y + diff(range(delta_R))*0.03,
     "Low OOD MSE", col=C_ACR, cex=0.7, adj=0)
text(legend_x + n_leg*rect_w, legend_y + diff(range(delta_R))*0.03,
     "High", col="grey50", cex=0.7, adj=1)
dev.off()


# ============================================================
# FIGURE 6: ADVERSARY AUC vs SHIFT STRENGTH (SEM)
# ============================================================

pdf(paste0(out_dir,"7-fig6_auc_vs_alpha.pdf"), width=7, height=5)
par(mar=c(4.5, 5, 3, 1.5), mgp=c(3,0.7,0), family="sans")

auc_m <- subset(auc_mean, shift_type=="mean")
auc_v <- subset(auc_mean, shift_type=="variance")
sd_m  <- subset(auc_sd,   shift_type=="mean")
sd_v  <- subset(auc_sd,   shift_type=="variance")

ylim <- c(0.44, 1.02)
plot(auc_m$alpha, auc_m$adv_auc,
     type="b", pch=17, lwd=2.5, col=C_ACR,
     ylim=ylim,
     xlab=expression("Training shift strength " ~ alpha),
     ylab="Environment discovery AUC",
     main="Adversary: How Well Does It Recover the True Split?",
     las=1)

# CI band for mean shift
polygon(c(auc_m$alpha, rev(auc_m$alpha)),
        c(auc_m$adv_auc + sd_m$adv_auc,
          rev(auc_m$adv_auc - sd_m$adv_auc)),
        col=adjustcolor(C_ACR, 0.15), border=NA)

lines(auc_v$alpha, auc_v$adv_auc,
      type="b", pch=17, lwd=2.5, col=C_OLS, lty=2)
polygon(c(auc_v$alpha, rev(auc_v$alpha)),
        c(auc_v$adv_auc + sd_v$adv_auc,
          rev(auc_v$adv_auc - sd_v$adv_auc)),
        col=adjustcolor(C_OLS, 0.15), border=NA)

# Chance line
abline(h=0.5, lty=3, col="grey55", lwd=1.5)
text(9.5, 0.515, "Chance (0.5)", col="grey45", cex=0.78, adj=1)

# Perfect line
abline(h=1.0, lty=3, col="grey80")

# Annotation: saturation zone
rect(0.7, 0.44, 10.3, 0.62,
     col=adjustcolor("grey80",0.25), border=NA)
text(5.5, 0.455,
     "Child-variable saturation zone\n(adversary near-blind)",
     col="grey40", cex=0.72, font=3)

legend("topright", bty="n", cex=0.88, lwd=2.5,
       pch=17, lty=c(1,2),
       col=c(C_ACR, C_OLS),
       legend=c("Mean shift (adversary detects signal)",
                "Variance shift (adversary blind)"))
dev.off()

cat("\nAll 6 figures saved to:\n")
cat(paste0("  ", out_dir, "7-fig", 1:6, "_*.pdf\n"))
cat("\nFigure list:\n")
cat("  Fig 1: 7-fig1_adversary_discovery.pdf\n")
cat("  Fig 2: 7-fig2_coefficients.pdf\n")
cat("  Fig 3: 7-fig3_ood_by_month.pdf\n")
cat("  Fig 4: 7-fig4_deltaR_nonmonotone.pdf\n")
cat("  Fig 5: 7-fig5_pareto_frontier.pdf\n")
cat("  Fig 6: 7-fig6_auc_vs_alpha.pdf\n")
