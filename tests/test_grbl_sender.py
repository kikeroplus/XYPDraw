from __future__ import annotations

import numpy as np

from xypdraw.grbl_sender import check_xy_bounds, parse_status
from xypdraw.types import PlotJob, Polyline
from xypdraw.path_extraction import compute_stats


def _job(points: list[tuple[float, float]]) -> PlotJob:
    poly = Polyline(points=np.array(points, dtype=float), closed=False)
    stats = compute_stats([poly], start_pos=(0.0, 0.0))
    return PlotJob(polylines=[poly], canvas_size_mm=(100.0, 100.0), stats=stats)


def test_parse_status_with_work_offset():
    line = "<Idle|MPos:0.000,0.000,0.000|FS:0,0|WCO:-180.000,-240.000,0.000>"
    parsed = parse_status(line)
    assert parsed["state"] == "Idle"
    assert parsed["machine_position"] == (0.0, 0.0, 0.0)
    assert parsed["work_offset"] == (-180.0, -240.0, 0.0)


def test_parse_status_without_work_offset():
    line = "<Run|MPos:10.500,-5.250,0.000|FS:1500,0>"
    parsed = parse_status(line)
    assert parsed["state"] == "Run"
    assert parsed["machine_position"] == (10.5, -5.25, 0.0)
    assert parsed["work_offset"] is None


def test_parse_status_unrecognized_line():
    parsed = parse_status("garbage")
    assert parsed == {"state": None, "machine_position": None, "work_offset": None}


def test_check_xy_bounds_within_range():
    job = _job([(0.0, 0.0), (50.0, -50.0), (100.0, -100.0)])
    violations = check_xy_bounds(job, max_x=100.0, max_y=100.0)
    assert violations == []


def test_check_xy_bounds_x_violation():
    job = _job([(0.0, 0.0), (150.0, -50.0)])
    violations = check_xy_bounds(job, max_x=100.0, max_y=100.0)
    assert len(violations) == 1
    assert "X座標" in violations[0]


def test_check_xy_bounds_y_violation():
    job = _job([(0.0, 0.0), (10.0, 20.0)])  # Y>0は範囲外(許容は0以下)
    violations = check_xy_bounds(job, max_x=100.0, max_y=100.0)
    assert len(violations) == 1
    assert "Y座標" in violations[0]
