# Release Checklist

Use this checklist before creating a GitHub release.

## Acceptance gate

- `git status` is clean on the release branch.
- The release branch is the exact branch being pushed to `origin/main`.
- Python unit tests pass.
- R test suite passes.
- Cross-language parity checks pass with the documented `monotone_spline` tolerance.
- Editable Python install smoke check passes.
- R source install smoke check passes without `reticulate`.
- Optional R isotonic smoke check passes with `reticulate` plus Python `lightgbm`.
- Quarto docs render succeeds.
- Python notebook and R vignette execute successfully.
- Website link scan finds no repo-local dead links.
- Homepage, docs pages, notebook, and vignette all describe the same diagnostics and optional dependency behavior.

## Release steps

- Update `CHANGELOG.md` and package versions if needed.
- Push the release branch to `main`.
- Confirm GitHub Actions are green.
- Confirm the Pages deployment completed and the live site opens the expected pages.
- Spot-check the homepage, diagnostics page, workflow notebook link, and vignette link on the live site.
- Create the GitHub release with notes summarizing:
  - calibrated and cross-calibrated workflows,
  - diagnostics and BLP additions,
  - optional isotonic dependency behavior,
  - `monotone_spline` parity tolerance policy.
