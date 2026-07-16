from __future__ import annotations

import sys
from pathlib import Path


# Keep `pytest` usable directly from a source checkout as well as after install.
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
