# Contributing

## Development workflow

- Put user-facing package changes in both `r/causalCalibration` and `python` when the API is intended to stay aligned.
- Keep parity fixtures in `shared/fixtures` up to date whenever algorithm behavior changes.
- Prefer additive documentation in `docs/` and runnable examples in `examples/`.

## Local checks

Python:

```bash
PYTHONPATH=python/src python3 -m unittest discover -s python/tests
```

R:

```r
testthat::test_dir("r/causalCalibration/tests/testthat")
```

## Release notes

- Update `CHANGELOG.md`.
- Confirm the root README still points readers to `main_deprecated` for the historical implementation.
- Regenerate parity expectations if calibration outputs changed intentionally.
