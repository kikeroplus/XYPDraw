"""書き出し範囲(プロッター可動範囲)外のポリラインをトリミングする。

mm座標(coordinate_map.polylines_to_mmで変換済み)に対して矩形クリップを
行う。範囲をまたぐ線分は境界で分割し、範囲外に完全に出る区間は破棄する
(結果、1本のポリラインが複数本に分かれることがある)。

XYPWriter (xypwriter/bbox_clip.py) から移植した実装。
"""
from __future__ import annotations

import numpy as np

from .path_extraction import compute_stats
from .types import PlotJob, Polyline


def _clip_segment(
    p0: np.ndarray, p1: np.ndarray, x_min: float, y_min: float, x_max: float, y_max: float
) -> tuple[float, float] | None:
    """Liang-Barsky法で線分[p0,p1]を矩形にクリップし、交差区間のパラメータ
    (t0, t1) (0<=t0<=t1<=1、p0+t*(p1-p0)が交差区間)を返す。交差しなければNone。
    """
    dx = p1[0] - p0[0]
    dy = p1[1] - p0[1]
    t0, t1 = 0.0, 1.0
    for p, q in (
        (-dx, p0[0] - x_min),
        (dx, x_max - p0[0]),
        (-dy, p0[1] - y_min),
        (dy, y_max - p0[1]),
    ):
        if p == 0:
            if q < 0:
                return None
            continue
        t = q / p
        if p < 0:
            if t > t1:
                return None
            t0 = max(t0, t)
        else:
            if t < t0:
                return None
            t1 = min(t1, t)
    if t0 > t1:
        return None
    return t0, t1


def clip_polylines_to_bbox(
    polylines: list[Polyline], x_min: float, y_min: float, x_max: float, y_max: float
) -> list[Polyline]:
    """各ポリラインを矩形[x_min,x_max]x[y_min,y_max]でクリップする。"""
    out: list[Polyline] = []
    for poly in polylines:
        pts = poly.points
        n = len(pts)
        if n == 0:
            continue
        if n == 1:
            x, y = pts[0]
            if x_min <= x <= x_max and y_min <= y <= y_max:
                out.append(Polyline(points=pts.copy(), closed=False))
            continue

        edges = [(i, i + 1) for i in range(n - 1)]
        if poly.closed:
            edges.append((n - 1, 0))

        current: list[np.ndarray] = []
        for i, j in edges:
            clipped = _clip_segment(pts[i], pts[j], x_min, y_min, x_max, y_max)
            if clipped is None:
                if len(current) >= 2:
                    out.append(Polyline(points=np.array(current), closed=False))
                current = []
                continue
            t0, t1 = clipped
            c0 = pts[i] + t0 * (pts[j] - pts[i])
            c1 = pts[i] + t1 * (pts[j] - pts[i])
            if current and not np.allclose(current[-1], c0, atol=1e-9):
                if len(current) >= 2:
                    out.append(Polyline(points=np.array(current), closed=False))
                current = []
            if not current:
                current = [c0]
            current.append(c1)
        if len(current) >= 2:
            out.append(Polyline(points=np.array(current), closed=False))
    return out


def clip_job_to_bounds(job: PlotJob, max_x: float, max_y: float) -> tuple[PlotJob, bool]:
    """PlotJobを機体可動範囲 X:[0,max_x] Y:[-max_y,0] にトリミングする。

    Returns: (トリミング後のPlotJob, 実際に何か削られたか)
    """
    clipped_polylines = clip_polylines_to_bbox(
        job.polylines, x_min=0.0, y_min=-max_y, x_max=max_x, y_max=0.0
    )
    stats = compute_stats(clipped_polylines, start_pos=(0.0, 0.0))
    clipped_job = PlotJob(
        polylines=clipped_polylines,
        canvas_size_mm=job.canvas_size_mm,
        stats=stats,
    )
    trimmed = stats.total_draw_distance < job.stats.total_draw_distance - 1e-6
    return clipped_job, trimmed
