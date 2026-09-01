# ============================================================
#   Fig 1: Adversary environment discovery (airquality scatter points separated by environment)
#   Fig 2: OOD MSE by holdout month (graph bar comparing methods for airquality data)
#   Fig 3: Pareto frontier (accuracy vs stability)
#   Fig 4: sem grid heatmap plot
#   Fig 5: OOD test MSE vs test shift strength for 2nd sem implementation
#   Fig 6: unified datsets 
# ============================================================

library(ggplot2)
library(dplyr)
library(tidyr)

out_dir <- ""
code_dir <- ""

# ---- Colour palette ----
C_OLS    <- "#2980B9"   # steel blue
C_ACR    <- "#C0392B"   # red
C_RAND   <- "#27AE60"   # green
C_ORACLE <- "#8E44AD"   # purple
C_DARK   <- "#1a2744"   # navy
C_GREY   <- "grey60"

# ---- ACR core functions ----
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
# AIRQUALITY DATA
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

# ============================================================
# FIGURE 1: ADVERSARY ENVIRONMENT DISCOVERY (SCATTER)
# ============================================================

pdf(paste0(out_dir,"fig_adversary_discovery.pdf"), width=6.5, height=5.5)
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
# FIGURE 3: OOD MSE BY HOLDOUT MONTH
# ============================================================

pdf(paste0(out_dir,"fig_ood_by_month.pdf"), width=7.5, height=5.5)
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
# FIGURE 5: PARETO FRONTIER
# ============================================================

pdf(paste0(out_dir,"fig_pareto_frontier.pdf"), width=6.5, height=5.5)
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



### 14
load(paste0(code_dir, "sem_grid_big.RData"))

results$x_label <- factor(
  paste0(results$n_children, ifelse(results$n_children == 1, " child", " children")),
  levels = paste0(0:4, ifelse(0:4 == 1, " child", " children")))

results$y_label <- factor(
  paste0(results$n_causal, ifelse(results$n_causal == 1, " parent", " parents")),
  levels = paste0(1:10, ifelse(1:10 == 1, " parent", " parents")))

p <- ggplot(results, aes(x = x_label, y = y_label,
                          fill = auc, label = sprintf("%.2f", auc))) +
  geom_tile(colour = "white", linewidth = 0.7) +
  geom_text(size = 3.2, colour = "black", fontface = "bold") +
  scale_fill_gradientn(
    colours = c("#4575b4", "#91bfdb", "#ffffbf", "#fc8d59", "#d73027"),
    values  = scales::rescale(c(0.50, 0.55, 0.60, 0.65, 0.75)),
    limits  = c(0.50, NA),
    name    = "Adversary\nAUC"
  ) +
  labs(title = NULL, subtitle = NULL, x = NULL, y = NULL) +
  theme_bw(base_size = 12) +
  theme(
    axis.text.x      = element_text(size = 10),
    axis.text.y      = element_text(size = 10),
    legend.position  = "right",
    panel.grid       = element_blank()
  )

for (path in c(code_dir, out_dir)) {
  ggsave(paste0(path, "auc_heatmap.pdf"), p, width = 8, height = 6.5)}




######### FIG 5 NEW SEM IMPLEMENTATION:

alpha_test_grid <- c(0, 1, 2, 3, 5, 7, 10, 15, 20, 30, 50)
alpha_rows      <- c(1, 3, 5, 10)
seeds_fig       <- 1:10
sd_y_fig        <- 0.5       

fit_list <- list(); curve_list <- list(); fi <- 1; ci <- 1

for (sht in c("mean", "variance")) {
  for (al in alpha_rows) {
    for (sid in seeds_fig) {

      # alpha_test does not touch the training draws, so the fit is done once
      d <- gen_sem(n_per_env, al, alpha_test_grid[1], sht, sid)
      fit    <- fit_acr_two_phase(d$X_tr, d$y_tr, seed = sid)
      b_ols  <- ridge_ols(d$X_tr, d$y_tr)
      b_orcr <- acr_learner(d$X_tr, d$y_tr,
                            as.numeric(d$env_tr == 1), fit$gamma)
      b_caus <- rep(0, 6)
      b_caus[2:3] <- ridge_ols(d$X_tr[, 2:3, drop = FALSE], d$y_tr)

      fit_list[[fi]] <- data.frame(
        shift_type  = sht, alpha = al, seed = sid,
        gamma       = fit$gamma,
        gamma_max   = fit$gamma_max,
        auc         = compute_auc(fit$w, d$env_tr),
        alpha_cover = al * sqrt((fit$gamma + 1) / 2))   # eq. 4.29
      fi <- fi + 1

      for (at in alpha_test_grid) {
        dt <- gen_sem(n_per_env, al, at, sht, sid)
        stopifnot(identical(dt$X_tr, d$X_tr))   # training set must be unchanged
        Xe <- dt$X_te; ye <- dt$y_te
        mse <- function(b) mean((ye - Xe %*% b)^2)

        curve_list[[ci]] <- data.frame(
          shift_type = sht, alpha = al, seed = sid, alpha_test = at,
          OLS = mse(b_ols), ACR = mse(fit$beta),
          `Oracle-CR` = mse(b_orcr), `Causal oracle` = mse(b_caus),
          check.names = FALSE)
        ci <- ci + 1
      }
    }
  }
}

fits   <- bind_rows(fit_list)
curves <- bind_rows(curve_list)

summ <- curves %>%
  pivot_longer(c(OLS, ACR, `Oracle-CR`, `Causal oracle`),
               names_to = "method", values_to = "mse") %>%
  group_by(shift_type, alpha, alpha_test, method) %>%
  summarise(m = mean(mse), s = sd(mse), n = n(), .groups = "drop") %>%
  mutate(lo = pmax(m - qt(0.975, n - 1) * s / sqrt(n), 1e-3),
         hi = m + qt(0.975, n - 1) * s / sqrt(n),
         method = factor(method,
                    levels = c("OLS", "ACR", "Oracle-CR", "Causal oracle")))

cover <- fits %>%
  group_by(shift_type, alpha) %>%
  summarise(alpha_cover = mean(alpha_cover),
            gamma_star  = mean(gamma), .groups = "drop")

cols <- c("OLS" = C_OLS, "ACR" = C_ACR,
          "Oracle-CR" = C_ORACLE, "Causal oracle" = "grey45")

p <- ggplot(summ, aes(alpha_test, m, colour = method, fill = method)) +
  geom_hline(yintercept = sd_y_fig^2, linetype = "dotted", colour = "grey55") +
  geom_vline(data = cover, aes(xintercept = alpha_cover),
             linetype = "dashed", colour = "grey35", inherit.aes = FALSE) +
  geom_ribbon(aes(ymin = lo, ymax = hi), alpha = 0.18, colour = NA) +
  geom_line(aes(linetype = method), linewidth = 0.75) +
  facet_grid(alpha ~ shift_type, scales = "free_y",
             labeller = labeller(
               alpha      = function(a) paste0("alpha == ", a),
               shift_type = c(mean = "mean shift", variance = "variance shift")),
             switch = "y") +
  scale_y_log10() +
  scale_colour_manual(values = cols, name = NULL) +
  scale_fill_manual(values = cols, guide = "none") +
  scale_linetype_manual(values = c("solid","solid","22","11"), guide = "none") +
  labs(x = expression(alpha[test]), y = "OOD test MSE (log scale)") +
  theme_bw(base_size = 11) +
  theme(legend.position = "top", panel.grid.minor = element_blank(),
        strip.background = element_rect(fill = "grey95", colour = NA))

ggsave(paste0(out_dir, "9-fig_mse_vs_alphatest.pdf"), p, width = 8, height = 9)
write.csv(curves, paste0(out_dir, "9-mse_vs_alphatest.csv"), row.names = FALSE)
print(cover)

# ============================================================
# FIGURE 6: UNIFIED DATASETS PLOT
# ============================================================

load(paste0(out_dir, "6-systematic_results.RData"))
cm <- read.csv(paste0(base_dir, "cmnist_results.csv"))
g  <- sort(unique(cm$gap))
mm  <- function(v) sapply(g, function(x) mean(cm[[v]][abs(cm$gap - x) < 1e-9]))
sdv <- function(v) sapply(g, function(x) sd(cm[[v]][abs(cm$gap - x) < 1e-9]))

cm_acr <- 100 * (mm("erm_mse") - mm("acr_mse"))    / mm("erm_mse")
cm_crr <- 100 * (mm("erm_mse") - mm("crrand_mse")) / mm("erm_mse")
cm$acr_p <- 100 * (cm$erm_mse - cm$acr_mse)    / cm$erm_mse
cm$crr_p <- 100 * (cm$erm_mse - cm$crrand_mse) / cm$erm_mse
cm_acr_sd <- sapply(g, function(x) sd(cm$acr_p[abs(cm$gap - x) < 1e-9]))
cm_crr_sd <- sapply(g, function(x) sd(cm$crr_p[abs(cm$gap - x) < 1e-9]))

# 1st dataset: ny airquality
aq_ok <- aq_results[!is.na(aq_results$ols_mse), ]
aq_ok$adv_p  <- 100 * (aq_ok$ols_mse - aq_ok$acr_adv_mse) / aq_ok$ols_mse
aq_ok$rand_p <- 100 * (aq_ok$ols_mse - aq_ok$cr_rand_mse) / aq_ok$ols_mse
aq_m  <- aggregate(cbind(ols_mse, acr_adv_mse, cr_rand_mse) ~ holdout_month,
                   aq_ok, mean)
aq_acr <- 100 * (aq_m$ols_mse - aq_m$acr_adv_mse) / aq_m$ols_mse
aq_crr <- 100 * (aq_m$ols_mse - aq_m$cr_rand_mse) / aq_m$ols_mse
aq_acr_sd <- aggregate(adv_p  ~ holdout_month, aq_ok, sd)$adv_p
aq_crr_sd <- aggregate(rand_p ~ holdout_month, aq_ok, sd)$rand_p

# rest of datasets
rs <- real_seed_results
rs_order <- c("CO2", "ChickWeight", "Boston")
rs$dataset <- factor(rs$dataset, levels = rs_order)
rs_m <- aggregate(cbind(ols_mse, acr_adv_mse, cr_rand_mse) ~ dataset, rs, mean)
rs_m <- rs_m[match(rs_order, as.character(rs_m$dataset)), ]
rl_acr <- 100 * (rs_m$ols_mse - rs_m$acr_adv_mse) / rs_m$ols_mse
rl_crr <- 100 * (rs_m$ols_mse - rs_m$cr_rand_mse) / rs_m$ols_mse
sd_by <- function(col) {
  a <- aggregate(as.formula(paste(col, "~ dataset")), rs, sd)
  a[match(rs_order, as.character(a$dataset)), 2]
}
rl_acr_sd <- sd_by("adv_vs_ols")   # sd is unaffected by the sign flip
rl_crr_sd <- sd_by("rand_vs_ols")


lab <- c(sprintf("MNIST\n%.2f/%.2f", 0.85 + g / 2, 0.85 - g / 2),
         "AQ\nMay", "AQ\nJun", "AQ\nJul", "AQ\nAug", "AQ\nSep", 
         "CO2\nMississippi", "ChickWt\nDiet 4", "Boston\nHigh crime")
acr    <- c(cm_acr,    aq_acr,    rl_acr)
crr    <- c(cm_crr,    aq_crr,    rl_crr)
acr_sd <- c(cm_acr_sd, aq_acr_sd, rl_acr_sd)
crr_sd <- c(cm_crr_sd, aq_crr_sd, rl_crr_sd)
ncm <- length(g); n <- length(lab)
stopifnot(length(acr) == n, length(crr) == n)

err_scale <- 1      
grp   <- c(rep(1, ncm), rep(2, 5), 3, 4, 5)
gname <- c("Colored MNIST", "NY Airquality", "CO2", "ChickWt", "Boston")
gcol  <- c("#2980B9", "#E67E22", "#1ABC9C", "#9B59B6", "#E74C3C")
col_acr <- C_ACR; col_crr <- C_RAND
 
pdf(paste0(out_dir, "7-fig_unified_comparison.pdf"), width = 13.2, height = 6.3)
par(mar = c(5.2, 5.0, 3.4, 1.2), family = "sans")
 
x <- seq_len(n); off <- 0.17; hw <- 0.15
rng  <- range(c(acr - acr_sd, acr + acr_sd, crr - crr_sd, crr + crr_sd, 0),
              na.rm = TRUE)
pad  <- 0.12 * diff(rng)
ylim <- c(rng[1] - pad, rng[2] + 2.5 * pad)   # headroom for the group labels
ticks <- pretty(rng, n = 7)
 
plot(NA, xlim = c(0.4, n + 0.6), ylim = ylim, xaxs = "i",
     xaxt = "n", yaxt = "n", xlab = "", ylab = "")
 
bs <- tapply(x, grp, min); be <- tapply(x, grp, max)
for (i in seq_along(bs))
  rect(bs[i] - 0.5, ylim[1], be[i] + 0.5, ylim[2],
       col = adjustcolor(gcol[i], alpha.f = 0.051), border = NA)
for (i in seq_along(bs)[-1])
  abline(v = bs[i] - 0.5, col = "grey65", lty = 2)
abline(h = 0, col = "grey45", lwd = 1)
 
axis(2, at = ticks, las = 1, cex.axis = 0.95)
mtext("% improvement over baseline  (OLS / ERM)", side = 2, line = 3.1, cex = 1.0)
 
rect(x - off - hw, 0, x - off + hw, crr, col = col_crr, border = NA)
rect(x + off - hw, 0, x + off + hw, acr, col = col_acr, border = NA)
 
## variation whiskers
e <- crr_sd * err_scale; ok <- !is.na(e)
segments(x[ok] - off, crr[ok] - e[ok], x[ok] - off, crr[ok] + e[ok],
         col = "black", lwd = 0.9)
e <- acr_sd * err_scale; ok <- !is.na(e)
segments(x[ok] + off, acr[ok] - e[ok], x[ok] + off, acr[ok] + e[ok],
         col = "black", lwd = 0.9)
 
axis(1, at = x, labels = FALSE, tick = FALSE)
mtext(lab, side = 1, at = x, line = 1.5, cex = 0.72)
text((bs + be) / 2, ylim[2] - 0.6 * pad, gname, col = gcol, font = 2, cex = 0.82)
 
legend("topleft", c("ACR-Adv", "CR-Random"), fill = c(col_acr, col_crr),
       border = NA, bty = "n", cex = 0.92, inset = c(0.005, 0.005))
mtext("Positive = better than OLS/ERM baseline   |   Negative = worse",
      side = 1, line = 3.7, cex = 0.8, font = 3, col = "grey40")
title("ACR vs Baseline on a Unified Scale Across Datasets and Benchmarks",
      cex.main = 1.25, font.main = 2)
box()
dev.off()
 
unified_tbl <- data.frame(panel = lab, acr_pct = round(acr, 2),
                          acr_sd = round(acr_sd, 2),
                          crrand_pct = round(crr, 2),
                          crrand_sd = round(crr_sd, 2))
print(unified_tbl, row.names = FALSE)
write.csv(unified_tbl, paste0(out_dir, "7-unified_comparison.csv"),
          row.names = FALSE)
