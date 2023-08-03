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
K <- 10 # 10-fold cross-fitting
task <- sl3_Task$new(data, covariates = c("W"), outcome = "Y", folds = K) # sl3_Task for T-learner
tmp <- task$folds # used to generate folds internally

# crossfit gam learner using sl3
lrnr <- Lrnr_cv$new(Lrnr_gam$new())

# Compute T-learner for CATE.
lrnr_trained_A0 <- lrnr$train(task[A==0])
lrnr_trained_A1 <- lrnr$train(task[A==1])

# get pooled out-of-fold predictions
initial_CATE_preds_pooled <- lrnr_trained_A1$predict(task) - lrnr_trained_A0$predict(task)

# train calibrator
calibrator <- causalCalibrate(tau, A, Y, EY1, EY0, pA1, tau_pred = initial_CATE_preds_pooled)


#generate new dataset to obtain calibrated predictions
n <- 200
W <- runif(n, -1 , 1)
new_data <- data.frame(W)

# new task
new_task <- sl3_Task$new(new_data, covariates = c("W"), outcome = c(), folds = 10)

# get matrix of fold-specific initial CATE predictions from new task
initial_CATE_cfpreds_mat <- do.call(cbind, lapply(1:K, function(k) {
  # get predictions using learner trained on kth training set
  tau_k <- lrnr_trained_A1$predict_fold(new_task, k) - lrnr_trained_A0$predict_fold(new_task, k)
  return(tau_k)
}))
# use cross_calibrate to collapse the predictions from the K initial cross-fitted CATE predictors into a single calibrated prediction. 
new_CATE_preds_crosscalibrated <- cross_calibrate(calibrator, initial_CATE_cfpreds_mat)
plot(W,  new_CATE_preds_crosscalibrated)




```

## Citation

To reference this work, please use the following bibtex citation:

`
@inproceedings{van2023causal,
  title={Causal isotonic calibration for heterogeneous treatment effects},
  author={{van der Laan}, Lars and Ulloa-P{\'e}rez, Ernesto and Carone, Marco and Luedtke, Alex},
  booktitle={Proceedings of the 40th International Conference on Machine Learning (ICML)},
  year={2023},
  address={Honolulu, Hawaii, USA},
  publisher={PMLR},
  volume={202},
    pdf={https://proceedings.mlr.press/v202/van-der-laan23a/van-der-laan23a.pdf}
}

`
