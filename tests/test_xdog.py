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
    # エッジは境界(列30付近)に集中しているはず。既定sigma=2.0はぼかしが
    # 広めなので、境界から離れた列にまで反応しないことだけ確認する
    # (画像端(0, 59)まで広がっていれば検出ロジックが破綻している)。
    cols_with_ink = np.where(np.any(mask, axis=0))[0]
    assert cols_with_ink.min() >= 10
    assert cols_with_ink.max() <= 50
