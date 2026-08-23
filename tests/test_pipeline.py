from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from xypdraw.hatching import HatchingConfig
from xypdraw.pipeline import XYPDrawConfig, process_image


@pytest.fixture
def sample_image_path(tmp_path):
    h, w = 120, 120
    yy, xx = np.mgrid[0:h, 0:w]
    img = (xx / w * 255).astype(np.uint8)
    mask = (yy - 60) ** 2 + (xx - 60) ** 2 <= 40**2
    img[mask] = np.clip(img[mask].astype(int) - 120, 0, 255).astype(np.uint8)
    path = tmp_path / "sample.jpg"
    Image.fromarray(img).convert("RGB").save(path, quality=90)
    return str(path)


def test_process_image_produces_job(sample_image_path):
    config = XYPDrawConfig(max_long_side_px=120, target_long_side_mm=100.0)
    result = process_image(sample_image_path, config)
    assert result.job is not None
    assert len(result.job.polylines) > 0
    assert result.job.canvas_size_mm[0] == pytest.approx(100.0, rel=0.05)
    assert len(result.contour_polylines_px) > 0  # 円の輪郭が検出される


def test_process_image_without_hatching(sample_image_path):
    config = XYPDrawConfig(max_long_side_px=120, enable_hatching=False)
    result = process_image(sample_image_path, config)
    assert result.hatching_polylines_px == []


def test_process_image_with_simplify(sample_image_path):
    config = XYPDrawConfig(
        max_long_side_px=120,
        hatching_config=HatchingConfig(spacing_px=15.0),
        simplify_tolerance_mm=1.0,
    )
    result = process_image(sample_image_path, config)
    assert result.job is not None
