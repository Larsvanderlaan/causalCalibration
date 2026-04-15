# causalCalibration

Production-ready causal calibration tooling for heterogeneous treatment effect models, with native R and Python packages, cross-calibration support, and doubly robust calibration diagnostics.

[Package Website](https://larsvanderlaan.github.io/causalCalibration/)

## Legacy branch

The original paper-era implementation has been preserved on the `main_deprecated` branch. If you need the historical ICML 2023 code exactly as it previously lived in this repository, use that branch directly.

## Repository layout

- `r/causalCalibration`: R package
- `python`: Python package
- `docs`: static package website and supporting notes
- `examples`: runnable end-to-end Python workflow materials
- `shared/fixtures`: shared inputs and expected outputs for parity tests

## Supported calibration losses

- `dr`: doubly robust/AIPW pseudo-outcome calibration
- `r`: residualized R-loss calibration

## Supported calibration methods

- `isotonic`
- `smooth_isotonic`
- `linear`
- `histogram`

## Install

Python:

```bash
python3 -m pip install -e python
```

R:

```r
install.packages("r/causalCalibration", repos = NULL, type = "source")
```

## Quick start

User-facing website:

- [Package website](./docs/index.html)

Source examples:

- Python notebook: [examples/python-workflow.ipynb](./examples/python-workflow.ipynb)
- R vignette: [getting-started.Rmd](./r/causalCalibration/vignettes/getting-started.Rmd)
- Method notes: [docs/getting-started.qmd](./docs/getting-started.qmd), [docs/standard-calibration.qmd](./docs/standard-calibration.qmd), [docs/cross-calibration.qmd](./docs/cross-calibration.qmd), [docs/diagnostics.qmd](./docs/diagnostics.qmd), [docs/choosing-losses-and-methods.qmd](./docs/choosing-losses-and-methods.qmd), [docs/reference.qmd](./docs/reference.qmd)

Python:

```python
from causal_calibration import fit_calibrator, diagnose_calibration

calibrator = fit_calibrator(
    predictions=tau_hat,
    treatment=a,
    outcome=y,
    mu0=mu0_hat,
    mu1=mu1_hat,
    propensity=e_hat,
    loss="dr",
    method="isotonic",
)

tau_calibrated = calibrator.predict(tau_new)

diagnostics = diagnose_calibration(
    predictions=tau_calibrated,
    comparison_predictions=tau_hat,
    treatment=a,
    outcome=y,
    mu0=mu0_hat,
    mu1=mu1_hat,
    propensity=e_hat,
)
```

R:

```r
library(causalCalibration)

calibrator <- fit_calibrator(
  predictions = tau_hat,
  treatment = a,
  outcome = y,
  mu0 = mu0_hat,
  mu1 = mu1_hat,
  propensity = e_hat,
  loss = "dr",
  method = "isotonic"
)

tau_calibrated <- predict(calibrator, tau_new)
```

## Method references

- ICML 2023 paper: [Causal Isotonic Calibration for Heterogeneous Treatment Effects](https://proceedings.mlr.press/v202/van-der-laan23a.html)
- AISTATS 2022 paper: [Calibration Error for Heterogeneous Treatment Effects](https://arxiv.org/abs/2203.13364)
- R-learner target weighting: [Nie and Wager (2021)](https://academic.oup.com/biomet/article/108/2/299/5911092)
- DR-learner theory: [Kennedy (2020)](https://doi.org/10.48550/arXiv.2004.14497)
