# causalCalibration: Causal Isotonic Calibration for Heterogeneous Treatment Effects

Provides R code implementing causal isotonic calibration and cross-calibration as described 
in the manuscript "Causal Isotonic Calibration for Heterogeneous Treatment Effects" by Lars van der Laan, Ernesto Ulloa, Marco Carone, and Alex Luedtke.

Preprint describing methods and algorithms can be found at this link: https://arxiv.org/abs/2302.14011.

See vignette for code examples.


```

set.seed(123)
n <- 1000
W <- runif(n, -1 , 1)
pA1 <- plogis(2*W)
A <- rbinom(n, size = 1, pA1)
CATE <- 1 + W
EY0 <- W
EY1 <- W + CATE
Y <- rnorm(n, W + A * CATE, 0.3)

# Initial uncalibrated predictor 
tau <- exp(CATE) - 1
 
calibrator <- causalCalibrate(tau, A, Y, EY1, EY0, pA1, tau_pred = tau)
plot(W,  calibrator$tau_calibrated)

```
