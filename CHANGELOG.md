# Changelog

## 1.2.0

- Added BLP-style diagnostics based on weighted pseudo-outcome regression with jackknife confidence intervals, p-values, and a zero-exclusion flag for the CATE slope.
- Aligned scalar calibration-error estimation with the Xu-Yadlowsky product-form estimator while keeping practical out-of-fold curve fitting.
- Hardened cross-calibration tests and shared fixtures around true pooled OOF predictions.
- Made R isotonic support optional through `reticulate` rather than a hard package load dependency.
- Tightened CI, docs checks, and release infrastructure, including explicit website-source link handling and a reproducible Pages deployment workflow.

## 1.1.0

- Replaced the isotonic backend with a weighted LightGBM monotone-tree calibrator in both languages.
- Removed `smooth_isotonic` and added `monotone_spline` as the package’s smooth monotone calibration method.
- Added overlap diagnostics, automatic overlap warnings, and explicit loss recommendations.
- Added overlap-targeted diagnostics for `loss="r"` workflows alongside the original-population DR diagnostic.
- Added stronger cross-fit bundle validation utilities and lightweight workflow bundle helpers.
- Added native Python plotting helpers for calibrators and diagnostics objects.

## 1.0.0

- Rebuilt the repository as a dual-language monorepo with native R and Python packages.
- Added framework-agnostic calibration APIs with `dr` and `r` losses.
- Added `isotonic`, `smooth_isotonic`, `linear`, and `histogram` calibration backends.
- Added first-class cross-calibration objects and prediction helpers.
- Added doubly robust calibration diagnostics with deterministic K-fold jackknife standard errors.
- Preserved the legacy paper-era implementation on `main_deprecated`.
