# Changelog

## 1.0.0

- Rebuilt the repository as a dual-language monorepo with native R and Python packages.
- Added framework-agnostic calibration APIs with `dr` and `r` losses.
- Added `isotonic`, `smooth_isotonic`, `linear`, and `histogram` calibration backends.
- Added first-class cross-calibration objects and prediction helpers.
- Added doubly robust calibration diagnostics with deterministic K-fold jackknife standard errors.
- Preserved the legacy paper-era implementation on `main_deprecated`.
