# `causal-calibration`

Native Python tools for causal calibration, cross-calibration, and calibration diagnostics for heterogeneous treatment effect models.

Install from the repo root with:

```bash
python3 -m pip install -e python
```

Quick start:

```python
from causal_calibration import fit_calibrator

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

tau_calibrated = calibrator.predict(tau_hat_new)
```

For the full workflow, see:

- [package website](https://larsvanderlaan.github.io/causalCalibration/)
- [Python workflow notebook](https://github.com/Larsvanderlaan/causalCalibration/blob/main/examples/python-workflow.ipynb)

Interpretation note:

- `loss="dr"` calibrates to the original study population but relies on inverse-propensity weighting.
- `loss="r"` calibrates an overlap-weighted target population and is often more robust under weak overlap.
