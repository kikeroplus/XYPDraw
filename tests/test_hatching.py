from __future__ import annotations

import numpy as np

from xypdraw.hatching import HatchLevel, HatchingConfig, auto_hatch_levels, generate_hatching


def test_bright_image_has_no_hatching():
    gray = np.full((80, 80), 255, dtype=np.uint8)
    polylines = generate_hatching(gray)
    assert polylines == []


def test_gradient_image_has_hatching_in_dark_region():
    # 左が暗く(0)右が明るい(255)グラデーション。auto_hatch_levelsは明度分布から
    # しきい値を算出するため、完全に一様な画像(分散ゼロ)だと機能しない
    # (percentileが全て同じ値になり、`gray < 閾値`が常にFalseになる)。
    gray = np.tile(np.linspace(0, 255, 80).astype(np.uint8), (80, 1))
    config = HatchingConfig(spacing_px=4.0)
    polylines = generate_hatching(gray, config)
    assert len(polylines) > 10
    for poly in polylines:
        assert poly.points.shape[1] == 2
        assert not poly.closed
        # ハッチングは暗い(=画像左側、col小)領域に生成されているはず
        assert poly.points[:, 1].max() < 60


def test_auto_hatch_levels_adapts_to_brightness_distribution():
    dark_gray = np.tile(np.linspace(0, 100, 80).astype(np.uint8), (80, 1))
    bright_gray = np.tile(np.linspace(155, 255, 80).astype(np.uint8), (80, 1))
    dark_levels = auto_hatch_levels(dark_gray)
    bright_levels = auto_hatch_levels(bright_gray)
    # 画像全体が暗ければしきい値も低く、明るければしきい値も高くなる
    assert max(l.max_gray for l in dark_levels) < max(l.max_gray for l in bright_levels)
    assert len(dark_levels) == len(bright_levels) == 4


def test_single_level_single_angle_produces_parallel_lines():
    gray = np.full((100, 100), 10, dtype=np.uint8)
    config = HatchingConfig(levels=[HatchLevel(200.0, [0.0])], spacing_px=10.0)
    polylines = generate_hatching(gray, config)
    assert len(polylines) > 0
    # angle=0 -> direction=(sin0,cos0)=(0,1) つまり列方向に伸びる横線。
    # 各線分の行(row)はほぼ一定のはず。
    for poly in polylines:
        rows = poly.points[:, 0]
        assert np.ptp(rows) < 1.0
