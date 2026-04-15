"""Documentation and workflow asset checks."""

from __future__ import annotations

import json
import os
import unittest


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


class DocsTests(unittest.TestCase):
    def test_quarto_pages_exist(self) -> None:
        for relative_path in (
            "docs/_quarto.yml",
            "docs/index.qmd",
            "docs/getting-started.qmd",
            "docs/standard-calibration.qmd",
            "docs/cross-calibration.qmd",
            "docs/diagnostics.qmd",
            "docs/choosing-losses-and-methods.qmd",
            "docs/reference.qmd",
            "examples/python-workflow.ipynb",
            "r/causalCalibration/vignettes/getting-started.Rmd",
        ):
            self.assertTrue(os.path.exists(os.path.join(REPO_ROOT, relative_path)), relative_path)

    def test_method_guide_contains_target_population_language_and_citations(self) -> None:
        with open(
            os.path.join(REPO_ROOT, "docs", "choosing-losses-and-methods.qmd"),
            encoding="utf-8",
        ) as handle:
            guide = handle.read()
        self.assertIn("overlap-weighted population", guide)
        self.assertIn("inverse-propensity weighting", guide)
        self.assertIn("Nie and Wager", guide)
        self.assertIn("Kennedy", guide)
        self.assertIn("van der Laan et al. (2023)", guide)

    def test_cross_calibration_page_links_to_sources(self) -> None:
        with open(
            os.path.join(REPO_ROOT, "docs", "cross-calibration.qmd"),
            encoding="utf-8",
        ) as handle:
            page = handle.read()
        self.assertIn("../examples/python-workflow.ipynb", page)
        self.assertIn("../r/causalCalibration/vignettes/getting-started.Rmd", page)

    def test_python_notebook_executes(self) -> None:
        with open(os.path.join(REPO_ROOT, "examples", "python-workflow.ipynb"), encoding="utf-8") as handle:
            notebook = json.load(handle)
        namespace: dict[str, object] = {}
        cwd = os.getcwd()
        try:
            os.chdir(REPO_ROOT)
            for cell in notebook["cells"]:
                if cell["cell_type"] != "code":
                    continue
                source = "".join(cell["source"])
                exec(compile(source, "<python-workflow>", "exec"), namespace, namespace)
        finally:
            os.chdir(cwd)
        self.assertIn("tau_cross_calibrated", namespace)
        self.assertIn("diagnostics", namespace)


if __name__ == "__main__":
    unittest.main()
