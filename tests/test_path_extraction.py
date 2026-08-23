from __future__ import annotations

import time

import numpy as np

from xypdraw.path_extraction import compute_stats, order_trails_nearest_neighbor
from xypdraw.types import Polyline


def _line(p0, p1, closed=False) -> Polyline:
    return Polyline(points=np.array([p0, p1], dtype=float), closed=closed)


def test_order_preserves_all_trails_without_duplication():
    trails = [_line([0, 0], [1, 1]), _line([5, 5], [6, 6]), _line([2, 2], [3, 3])]
    ordered = order_trails_nearest_neighbor(trails, start_pos=(0.0, 0.0))
    assert len(ordered) == len(trails)
    # 各元trailの始点集合が、順序変更後も(向き反転含め)全て現れる
    orig_endpoints = {tuple(t.points[0]) for t in trails} | {tuple(t.points[-1]) for t in trails}
    got_endpoints = {tuple(t.points[0]) for t in ordered} | {tuple(t.points[-1]) for t in ordered}
    assert orig_endpoints == got_endpoints


def test_order_picks_nearest_first():
    trails = [_line([100, 100], [101, 101]), _line([1, 0], [2, 0])]
    ordered = order_trails_nearest_neighbor(trails, start_pos=(0.0, 0.0))
    assert np.allclose(ordered[0].points[0], [1, 0])


def test_closed_loop_keeps_orientation():
    loop = Polyline(points=np.array([[0, 0], [1, 0], [1, 1], [0, 0]], dtype=float), closed=True)
    ordered = order_trails_nearest_neighbor([loop], start_pos=(0.0, 0.0))
    assert np.array_equal(ordered[0].points, loop.points)


def test_large_input_completes_quickly():
    rng = np.random.default_rng(0)
    trails = []
    for _ in range(3000):
        p0 = rng.uniform(0, 1000, size=2)
        p1 = p0 + rng.uniform(-2, 2, size=2)
        trails.append(_line(p0, p1))
    t0 = time.time()
    ordered = order_trails_nearest_neighbor(trails, start_pos=(0.0, 0.0))
    elapsed = time.time() - t0
    assert len(ordered) == len(trails)
    assert elapsed < 10.0  # O(n^2)実装だと数千本でこの時間を大幅に超える


def test_compute_stats_counts_pen_lifts():
    trails = [_line([0, 0], [1, 0]), _line([5, 0], [6, 0])]
    stats = compute_stats(trails, start_pos=(0.0, 0.0))
    assert stats.n_paths == 2
    assert stats.n_pen_lifts == 2
