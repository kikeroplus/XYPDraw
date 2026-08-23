"""ポリライン単純化（高速描画モード用）。

スケルトン/輪郭の画素列をそのままG-code点列にすると1セグメントあたり
0.1mm未満の極短区間が数百連続し、GRBLの加減速プランナーが目標速度に
到達できないまま減速を繰り返す・G-code行数の通信オーバーヘッドが支配的に
なる、という速度低下の主因になる。Douglas-Peuckerで許容誤差内の中間点を
間引き、セグメントを長くすることで実効描画速度を上げる。

品質を優先する既定の動作（間引きなし）はそのまま維持し、この単純化は
simplify_tolerance_mm を明示指定した場合のみ有効になるオプトイン機能とする。

XYPWriter (xypwriter/path_simplify.py) から移植した汎用ロジック。
"""
from __future__ import annotations

import numpy as np

from .types import Polyline


def _point_segment_distance(pt: np.ndarray, start: np.ndarray, end: np.ndarray) -> float:
    line = end - start
    line_len_sq = float(np.dot(line, line))
    if line_len_sq == 0.0:
        return float(np.hypot(*(pt - start)))
    t = float(np.dot(pt - start, line) / line_len_sq)
    t = max(0.0, min(1.0, t))
    proj = start + t * line
    return float(np.hypot(*(pt - proj)))


def douglas_peucker(points: np.ndarray, tolerance: float) -> np.ndarray:
    """Douglas-Peuckerアルゴリズムで点列を単純化する。

    Args:
        points: shape (N, 2)
        tolerance: 許容誤差（pointsと同じ単位）。元の折れ線からこの距離以内なら間引く。
    """
    if len(points) < 3 or tolerance <= 0:
        return points

    start, end = points[0], points[-1]
    dmax = 0.0
    index = 0
    for i in range(1, len(points) - 1):
        d = _point_segment_distance(points[i], start, end)
        if d > dmax:
            dmax = d
            index = i

    if dmax <= tolerance:
        return np.array([start, end])

    left = douglas_peucker(points[: index + 1], tolerance)
    right = douglas_peucker(points[index:], tolerance)
    return np.vstack([left[:-1], right])


def simplify_polyline(poly: Polyline, tolerance: float) -> Polyline:
    simplified = douglas_peucker(poly.points, tolerance)
    return Polyline(points=simplified, closed=poly.closed, source_edge_ids=poly.source_edge_ids)


def simplify_polylines(polylines: list[Polyline], tolerance: float) -> list[Polyline]:
    return [simplify_polyline(p, tolerance) for p in polylines]
