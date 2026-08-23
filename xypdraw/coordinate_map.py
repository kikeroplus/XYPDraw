"""座標マッピング。

px -> mm 変換: `mm = px * px_to_mm_scale`。原点は画像左上、X+が右、
Y-(マイナス)方向が下——人が紙に描く向きに合わせる(numpy座標のrow増加=下方向を
そのままY減少に対応させる)。プロッター原点オフセットはmm変換後に加算する。

XYPWriterはPDFのdpiからpx->mmスケールを決めていたが、JPG画像には
物理的な解像度情報が無いため、代わりに「出力したい長辺の実寸(mm)」から
逆算する。
"""
from __future__ import annotations

import numpy as np

from .types import Polyline


def compute_px_to_mm_scale(image_shape: tuple[int, int], target_long_side_mm: float) -> float:
    """画像の長辺がtarget_long_side_mmになるようなpx->mmスケールを求める。"""
    h, w = image_shape
    long_side_px = max(h, w)
    if long_side_px <= 0:
        return 1.0
    return target_long_side_mm / long_side_px


def image_size_mm(image_shape: tuple[int, int], px_to_mm: float) -> tuple[float, float]:
    h, w = image_shape
    return (w * px_to_mm, h * px_to_mm)


def polylines_to_mm(
    polylines_px: list[Polyline],
    px_to_mm: float,
    origin_offset_mm: tuple[float, float] = (0.0, 0.0),
) -> list[Polyline]:
    """px空間(row, col)のポリライン列をmm空間(x, y)へ変換する。"""
    offset_x, offset_y = origin_offset_mm
    out: list[Polyline] = []
    for poly in polylines_px:
        if len(poly.points) == 0:
            out.append(poly)
            continue
        rows = poly.points[:, 0]
        cols = poly.points[:, 1]
        x_mm = cols * px_to_mm + offset_x
        y_mm = -rows * px_to_mm + offset_y
        pts_mm = np.stack([x_mm, y_mm], axis=1)
        out.append(Polyline(points=pts_mm, closed=poly.closed, source_edge_ids=poly.source_edge_ids))
    return out
