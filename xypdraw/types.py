"""パイプライン全ステージで共有するデータ構造。

データフロー:
    ImageRaster -> (edge_mask -> SkeletonGraph -> trail Polyline群)
                 -> (shade_mask -> hatching Polyline群)
    -> List[Polyline] (px空間, 順序最適化済み) -> PlotJob (mm空間)

XYPWriter (xypwriter/types.py) のグリフ非依存部分を踏襲。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

NodeRole = Literal["endpoint", "junction", "synthetic"]


@dataclass
class ImageRaster:
    """入力JPG画像の前処理結果。"""

    gray: np.ndarray  # uint8, shape (H, W)。前処理(CLAHE等)後のグレースケール画像
    source_path: str


@dataclass
class SkeletonResult:
    """細線化＋スパー除去の結果。"""

    raw_skeleton: np.ndarray  # bool
    pruned_skeleton: np.ndarray  # bool
    stroke_width_px: float
    spur_threshold_px: float
    n_components_before: int
    n_components_after: int
    warnings: list[str] = field(default_factory=list)


@dataclass
class SkeletonNode:
    id: int
    pixels: list[tuple[int, int]]  # (row, col) 元画素群
    position: tuple[float, float]  # (row, col) float、統合後は重心
    role: NodeRole


@dataclass
class SkeletonEdge:
    id: int
    node_a: int
    node_b: int
    pixels: list[tuple[int, int]]  # node_a -> node_b の順に並んだ (row, col) 画素列（両端ノード込み）
    length_px: float


@dataclass
class SkeletonGraph:
    nodes: dict[int, SkeletonNode]
    edges: dict[int, SkeletonEdge]
    image_shape: tuple[int, int]

    def degree(self, node_id: int) -> int:
        return sum(1 for e in self.edges.values() if e.node_a == node_id or e.node_b == node_id)

    def incident_edges(self, node_id: int) -> list[SkeletonEdge]:
        return [e for e in self.edges.values() if e.node_a == node_id or e.node_b == node_id]


@dataclass
class Polyline:
    points: np.ndarray  # shape (N, 2)、列は (row, col) または (x_mm, y_mm)。文脈で判断。
    closed: bool = False
    source_edge_ids: list[int] = field(default_factory=list)

    def reversed(self) -> "Polyline":
        return Polyline(
            points=self.points[::-1].copy(),
            closed=self.closed,
            source_edge_ids=list(reversed(self.source_edge_ids)),
        )

    def length(self) -> float:
        if len(self.points) < 2:
            return 0.0
        diffs = np.diff(self.points, axis=0)
        return float(np.hypot(diffs[:, 0], diffs[:, 1]).sum())


@dataclass
class PlotStats:
    n_paths: int
    total_draw_distance: float
    total_travel_distance: float
    n_pen_lifts: int

    def summary(self) -> str:
        return (
            f"総パス数: {self.n_paths}, "
            f"総描画距離: {self.total_draw_distance:.2f}, "
            f"総ペンアップ移動距離: {self.total_travel_distance:.2f}, "
            f"ペンアップダウン回数: {self.n_pen_lifts}"
        )


@dataclass
class PlotJob:
    """mm座標・描画順に確定済みのポリライン列。"""

    polylines: list[Polyline]  # mm座標、描画順に確定済み
    canvas_size_mm: tuple[float, float]  # (width, height)
    stats: PlotStats
