"""全ステージのオーケストレーション。

JPG読込 -> 前処理(バイラテラルフィルタ+CLAHE)
    -> XDoGで輪郭二値マスク生成 -> 細線化+スパー除去 -> グラフ化
    -> 交差点集約 -> trail抽出(輪郭ポリライン群)
    -> ハッチング線分生成(陰影ポリライン群)
    -> 輪郭+ハッチングを最近傍法で順序最適化
    -> px -> mm変換 -> PlotJob

px -> mm 変換は coordinate_map.polylines_to_mm の1箇所に閉じ込める。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .coordinate_map import compute_px_to_mm_scale, image_size_mm, polylines_to_mm
from .graph_build import build_skeleton_graph
from .hatching import HatchingConfig, generate_hatching
from .image_load import denoise, enhance_contrast, load_grayscale
from .intersection_merge import check_merge_safety, merge_close_junctions
from .metrics import estimate_merge_radius
from .path_extraction import compute_stats, extract_trails, order_trails_nearest_neighbor
from .path_simplify import simplify_polylines
from .skeleton_stage import remove_spurs, skeletonize_mask
from .types import PlotJob, Polyline
from .xdog import xdog_binary


@dataclass
class XYPDrawConfig:
    max_long_side_px: int = 1600  # 処理解像度の上限(長辺px)。大きいほど精細だが低速
    # ---- 前処理 ----
    bilateral_d: int = 5
    bilateral_sigma_color: float = 50.0
    bilateral_sigma_space: float = 50.0
    clahe_clip_limit: float = 2.0
    # ---- XDoG(輪郭抽出) ----
    xdog_sigma: float = 1.2
    xdog_k: float = 1.6
    xdog_tau: float = 0.98
    xdog_epsilon: float = -0.01
    xdog_phi: float = 200.0
    xdog_threshold: float = 0.5
    min_object_size_px: int = 4
    spur_factor: float = 1.4
    merge_factor: float = 0.85
    # ---- ハッチング(陰影) ----
    enable_hatching: bool = True
    hatching_config: HatchingConfig = field(default_factory=HatchingConfig)
    # ---- 出力 ----
    target_long_side_mm: float = 200.0  # プロッター上での出力サイズ(長辺mm)
    origin_offset_mm: tuple[float, float] = (0.0, 0.0)
    simplify_tolerance_mm: float | None = None  # 既定None=間引きなし


@dataclass
class PipelineResult:
    gray: np.ndarray  # 前処理後グレースケール(CLAHE適用後。プレビュー/ハッチング判定に使用)
    edge_mask: np.ndarray  # XDoG二値マスク(可視化用)
    contour_polylines_px: list[Polyline]
    hatching_polylines_px: list[Polyline]
    warnings: list[str] = field(default_factory=list)
    job: PlotJob | None = None


def process_image(image_path: str | Path, config: XYPDrawConfig) -> PipelineResult:
    warnings: list[str] = []

    raw_gray = load_grayscale(image_path, max_long_side_px=config.max_long_side_px)
    smoothed = denoise(
        raw_gray,
        d=config.bilateral_d,
        sigma_color=config.bilateral_sigma_color,
        sigma_space=config.bilateral_sigma_space,
    )
    gray = enhance_contrast(smoothed, clip_limit=config.clahe_clip_limit)

    # XDoGはCLAHE前(smoothed)にかける: CLAHE後の急なコントラストはXDoGの
    # 輪郭を過剰に増やし線画を汚す傾向があるため、輪郭抽出と陰影判定で
    # 異なる下地画像を使い分ける。
    edge_mask = xdog_binary(
        smoothed,
        sigma=config.xdog_sigma,
        k=config.xdog_k,
        tau=config.xdog_tau,
        epsilon=config.xdog_epsilon,
        phi=config.xdog_phi,
        threshold=config.xdog_threshold,
    )

    contour_polylines_px: list[Polyline] = []
    if np.any(edge_mask):
        skel_raw = skeletonize_mask(edge_mask, clean_noise=True, min_object_size=config.min_object_size_px)
        skeleton_result = remove_spurs(edge_mask, skel_raw, factor=config.spur_factor)
        graph_raw = build_skeleton_graph(skeleton_result.pruned_skeleton)
        merge_radius = estimate_merge_radius(skeleton_result.stroke_width_px, config.merge_factor)
        graph_merged = merge_close_junctions(graph_raw, merge_radius)
        warnings.extend(skeleton_result.warnings)
        warnings.extend(check_merge_safety(graph_raw, graph_merged))
        contour_polylines_px = extract_trails(graph_merged)

    hatching_polylines_px: list[Polyline] = []
    if config.enable_hatching:
        hatching_polylines_px = generate_hatching(gray, config.hatching_config)

    all_px = contour_polylines_px + hatching_polylines_px
    ordered_px = order_trails_nearest_neighbor(all_px, start_pos=(0.0, 0.0))

    px_to_mm = compute_px_to_mm_scale(raw_gray.shape, config.target_long_side_mm)
    ordered_mm = polylines_to_mm(ordered_px, px_to_mm=px_to_mm, origin_offset_mm=config.origin_offset_mm)
    if config.simplify_tolerance_mm is not None:
        ordered_mm = simplify_polylines(ordered_mm, config.simplify_tolerance_mm)

    stats = compute_stats(ordered_mm, start_pos=(0.0, 0.0))
    canvas_mm = image_size_mm(raw_gray.shape, px_to_mm)

    job = PlotJob(polylines=ordered_mm, canvas_size_mm=canvas_mm, stats=stats)

    return PipelineResult(
        gray=gray,
        edge_mask=edge_mask,
        contour_polylines_px=contour_polylines_px,
        hatching_polylines_px=hatching_polylines_px,
        warnings=warnings,
        job=job,
    )
