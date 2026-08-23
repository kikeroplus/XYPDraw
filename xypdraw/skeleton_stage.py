"""細線化（skeletonize）＋スパー（ヒゲ）除去。

XYPWriter (xypwriter/skeleton_stage.py) から移植した汎用ロジック。
画像固有の前提はないため、XDoGで抽出した線画の2値マスクに対して
そのまま使える。
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import convolve, distance_transform_edt
from skimage.morphology import remove_small_objects, skeletonize

from .metrics import count_components, estimate_spur_threshold, estimate_stroke_width
from .types import SkeletonResult

KERNEL = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])


def clean_binary_noise(binary: np.ndarray, min_size: int = 4) -> np.ndarray:
    """アンチエイリアス由来の微小な孤立画素を除去する。"""
    return remove_small_objects(binary, max_size=max(min_size - 1, 0), connectivity=2)


def skeletonize_mask(binary: np.ndarray, clean_noise: bool = True, min_object_size: int = 4) -> np.ndarray:
    """2値マスクからセンターライン（スケルトン）を抽出する。"""
    work = clean_binary_noise(binary, min_object_size) if clean_noise else binary
    return skeletonize(work)


def neighbor_count(skel: np.ndarray) -> np.ndarray:
    return convolve(skel.astype(int), KERNEL, mode="constant", cval=0)


def get_neighbors(skel: np.ndarray, pt: tuple[int, int]) -> list[tuple[int, int]]:
    y, x = pt
    out = []
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0:
                continue
            ny, nx = y + dy, x + dx
            if 0 <= ny < skel.shape[0] and 0 <= nx < skel.shape[1] and skel[ny, nx]:
                out.append((ny, nx))
    return out


def prune_spurs(
    skel: np.ndarray,
    min_length: float,
    dist: np.ndarray | None = None,
    stroke_width_px: float = 0.0,
    endpoint_radius_ratio: float = 1.0,
) -> np.ndarray:
    """スケルトンの端点から分岐点までの経路長が min_length 以下なら
    トメ・ハネ由来のノイズとみなして除去する。

    経路が分岐点に到達せずもう一方の端点で終わる場合（他画から独立した
    孤立ストローク全体）は、意図的な点画の可能性があるため除去しない。
    """
    skel = skel.copy()
    nb = neighbor_count(skel)
    endpoints = list(zip(*np.where(skel & (nb == 1))))
    to_remove = set()
    stroke_radius = stroke_width_px / 2.0
    for ep in endpoints:
        if ep in to_remove:
            continue
        path = [ep]
        prev, cur = None, ep
        reached_branch = False
        while True:
            nbrs = [n for n in get_neighbors(skel, cur) if n != prev]
            if len(nbrs) == 0:
                break
            if len(nbrs) > 1:
                break
            nxt = nbrs[0]
            if nb[nxt] >= 3:
                reached_branch = True
                break
            path.append(nxt)
            prev, cur = cur, nxt
        if not reached_branch or len(path) > min_length:
            continue
        if dist is not None and stroke_radius > 0 and dist[ep] >= endpoint_radius_ratio * stroke_radius:
            continue
        to_remove.update(path)
    for (y, x) in to_remove:
        skel[y, x] = False
    return skel


def remove_spurs(
    binary: np.ndarray,
    raw_skeleton: np.ndarray,
    threshold_px: float | None = None,
    factor: float = 1.4,
) -> SkeletonResult:
    """スパー除去を実行し、統計・警告を含む SkeletonResult を返す。"""
    dist = distance_transform_edt(binary)
    stroke_width_px = estimate_stroke_width(binary, raw_skeleton, dist=dist)
    threshold = threshold_px if threshold_px is not None else estimate_spur_threshold(
        stroke_width_px, factor
    )

    n_before = count_components(raw_skeleton)
    pruned = prune_spurs(raw_skeleton, threshold, dist=dist, stroke_width_px=stroke_width_px)
    n_after = count_components(pruned)

    warnings: list[str] = []
    if n_after != n_before:
        warnings.append(
            f"スパー除去前後で連結成分数が変化しました ({n_before} -> {n_after})。"
            "閾値が大きすぎて本画を切断した可能性があります。"
        )

    return SkeletonResult(
        raw_skeleton=raw_skeleton,
        pruned_skeleton=pruned,
        stroke_width_px=stroke_width_px,
        spur_threshold_px=threshold,
        n_components_before=n_before,
        n_components_after=n_after,
        warnings=warnings,
    )
