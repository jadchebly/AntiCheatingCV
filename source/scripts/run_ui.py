#!/usr/bin/env python3
"""Launch the local web UI.

    python3 source/scripts/run_ui.py
    python3 source/scripts/run_ui.py --port 8080 --no-browser

Serves on http://127.0.0.1:8000 by default and opens a browser window.
Port 5000 is avoided on purpose: on macOS it belongs to AirPlay Receiver.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from webui.app import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
