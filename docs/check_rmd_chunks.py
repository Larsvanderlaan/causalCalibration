#!/usr/bin/env python3
"""Extract runnable R code chunks from an R Markdown file."""

from __future__ import annotations

import re
import sys
from pathlib import Path


def main() -> int:
    if len(sys.argv) != 3:
        print("usage: check_rmd_chunks.py <input.Rmd> <output.R>", file=sys.stderr)
        return 2

    input_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2])
    text = input_path.read_text(encoding="utf-8")
    chunks = re.findall(r"```\{r[^}]*\}\n(.*?)```", text, flags=re.DOTALL)
    script = "\n\n".join(chunk.strip() for chunk in chunks if chunk.strip())
    output_path.write_text(script + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
