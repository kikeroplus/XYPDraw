"""XYPDraw GUIアプリの起動スクリプト。

使い方:
    python scripts/gui_app.py

(パッケージがインストール済みなら `python -m xypdraw.gui` でも起動できる)
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from xypdraw.gui import main  # noqa: E402

if __name__ == "__main__":
    main()
