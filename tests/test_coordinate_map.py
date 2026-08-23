from __future__ import annotations

import numpy as np

from xypdraw.coordinate_map import compute_px_to_mm_scale, image_size_mm, polylines_to_mm
from xypdraw.types import Polyline


def test_compute_px_to_mm_scale():
    scale = compute_px_to_mm_scale((100, 200), target_long_side_mm=100.0)
    assert scale == 0.5  # long_side_px=200 -> 100/200=0.5


def test_image_size_mm():
    scale = 0.5
    w_mm, h_mm = image_size_mm((100, 200), scale)
    assert w_mm == 100.0  # width(=cols)=200 * 0.5
    assert h_mm == 50.0  # height(=rows)=100 * 0.5


def test_polylines_to_mm_orientation_and_offset():
    poly = Polyline(points=np.array([[0.0, 0.0], [10.0, 20.0]]), closed=False)
    out = polylines_to_mm([poly], px_to_mm=1.0, origin_offset_mm=(5.0, 5.0))
    pts = out[0].points
    # x_mm = col*scale+offset_x, y_mm = -row*scale+offset_y
    assert np.allclose(pts[0], [5.0, 5.0])
    assert np.allclose(pts[1], [25.0, -5.0])
