from __future__ import annotations

import numpy as np

from xypdraw.xdog import xdog_binary


def test_flat_image_has_no_edges():
    gray = np.full((50, 50), 128, dtype=np.uint8)
    mask = xdog_binary(gray)
    assert not np.any(mask)


def test_step_edge_is_detected():
    gray = np.zeros((60, 60), dtype=np.uint8)
    gray[:, 30:] = 255
    mask = xdog_binary(gray)
    assert np.any(mask)
    # エッジは境界(列30付近)に集中しているはず
    cols_with_ink = np.where(np.any(mask, axis=0))[0]
    assert cols_with_ink.min() >= 20
    assert cols_with_ink.max() <= 40
