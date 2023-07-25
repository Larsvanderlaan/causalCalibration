# causalCalibration: Causal Isotonic Calibration for Heterogeneous Treatment Effects

Provides R code implementing causal isotonic calibration and cross-calibration as described 
in the manuscript "Causal Isotonic Calibration for Heterogeneous Treatment Effects" by Lars van der Laan, Ernesto Ulloa, Marco Carone, and Alex Luedtke.

Preprint describing methods and algorithms can be found at this link: https://arxiv.org/abs/2302.14011.

See vignette for code examples.


```
install.packages(“devtools”)
library(devtools)
install_github(“larsvanderlaan/causalCalibration”)
library(causalCalibration)
install_github(“tlverse/sl3”)
library(sl3)

set.seed(123)
n <- 1000
W <- runif(n, -1 , 1)
pA1 <- plogis(2*W)
A <- rbinom(n, size = 1, pA1)
CATE <- 1 + W
EY0 <- W
EY1 <- W + CATE
Y <- rnorm(n, W + A * CATE, 0.3)

# Initial cross-fitted learner of CATE

data <- data.frame(W, A, Y)
K <- 10
task_outcome <- sl3_Task$new(data, covariates = c("W"), outcome = "Y", folds = K)
folds <- task_outcome$folds
# crossfit gam learner using sl3
lrnr <- Lrnr_cv$new(Lrnr_gam$new())

# Compute T-learner for CATE.
lrnr_trained_A0 <- lrnr$train(task[A==0])
lrnr_trained_A1 <- lrnr$train(task[A==1])

# get pooled out-of-fold predictions
tau_preds_pooled <- lrnr_trained_A1$predict(task) - lrnr_trained_A0$predict(task)

calibrator <- causalCalibrate(tau, A, Y, EY1, EY0, pA1, tau_pred = tau_preds_pooled)


#generate new dataset
n <- 200
W <- runif(n, -1 , 1)
pA1 <- plogis(2*W)
A <- rbinom(n, size = 1, pA1)
CATE <- 1 + W
EY0 <- W
EY1 <- W + CATE
data <- data.frame(W, A, Y)
new_task <- sl3_Task$new(data, covariates = c("W"), outcome = "Y", folds = 10)


# get matrix of fold-specific CATE predictions from each cross-fitted CATE learner.
new_preds_mat <- do.call(cbind, lapply(1:K, function(k) {
  # get predictions using learner trained on kth training set
   tau_k <- lrnr_trained_A1$predict_fold(new_task, k) - lrnr_trained_A0$predict_fold(new_task, k)
  return(tau_k)
}))
new_preds_crosscalibrated <- cross_calibrate(calibrator, new_tau_mat)
plot(W,  new_preds_crosscalibrated)

 



```
