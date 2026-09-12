"""Convenience wrapper: `python source/scripts/run_proctoring.py --video ... --out ...`."""

from __future__ import annotations

import sys
from pathlib import Path

THIS = Path(__file__).resolve()
SOURCE_DIR = THIS.parents[1]
if str(SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIR))

from proctoring.cli import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main(["run", *sys.argv[1:]]))
