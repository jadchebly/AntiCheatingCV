#!/usr/bin/env python3
"""Launch the local web UI.

    python3 source/scripts/run_ui.py

Then open http://127.0.0.1:5000 in a browser.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from webui.app import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
