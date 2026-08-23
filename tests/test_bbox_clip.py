from __future__ import annotations

import numpy as np

from xypdraw.bbox_clip import clip_job_to_bounds, clip_polylines_to_bbox
from xypdraw.path_extraction import compute_stats
from xypdraw.types import PlotJob, Polyline


def _job(points: list[tuple[float, float]]) -> PlotJob:
    poly = Polyline(points=np.array(points, dtype=float), closed=False)
    stats = compute_stats([poly], start_pos=(0.0, 0.0))
    return PlotJob(polylines=[poly], canvas_size_mm=(100.0, 100.0), stats=stats)


def test_clip_polylines_fully_inside_is_unchanged():
    poly = Polyline(points=np.array([[10.0, -10.0], [20.0, -20.0]]), closed=False)
    out = clip_polylines_to_bbox([poly], x_min=0.0, y_min=-100.0, x_max=100.0, y_max=0.0)
    assert len(out) == 1
    assert np.allclose(out[0].points, poly.points)


def test_clip_polylines_partially_outside_is_trimmed():
    poly = Polyline(points=np.array([[-10.0, 0.0], [10.0, 0.0]]), closed=False)
    out = clip_polylines_to_bbox([poly], x_min=0.0, y_min=-100.0, x_max=100.0, y_max=0.0)
    assert len(out) == 1
    assert out[0].points[0, 0] == 0.0  # x=-10からx=0で切り取られる


def test_clip_polylines_fully_outside_is_dropped():
    poly = Polyline(points=np.array([[200.0, 0.0], [300.0, 0.0]]), closed=False)
    out = clip_polylines_to_bbox([poly], x_min=0.0, y_min=-100.0, x_max=100.0, y_max=0.0)
    assert out == []


def test_clip_job_to_bounds_reports_trimmed_flag():
    job = _job([(0.0, 0.0), (200.0, -50.0)])
    clipped, trimmed = clip_job_to_bounds(job, max_x=100.0, max_y=100.0)
    assert trimmed is True
    assert clipped.stats.total_draw_distance < job.stats.total_draw_distance


def test_clip_job_to_bounds_no_trim_when_within_range():
    job = _job([(0.0, 0.0), (50.0, -50.0)])
    clipped, trimmed = clip_job_to_bounds(job, max_x=100.0, max_y=100.0)
    assert trimmed is False
