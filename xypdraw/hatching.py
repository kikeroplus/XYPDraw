"""明るさに応じたクロスハッチングによる陰影表現。

XDoGは輪郭線しか抽出しないため、そのままでは写真の階調(陰影)が失われる。
暗い領域ほど重ねる線の角度数を増やす(=密度を上げる)ことで、階調を
線の密度として表現する。等間隔の平行線群を各レベルの明度マスクで
クリップし、マスク内側だけを短いポリライン群として残す。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .types import Polyline


@dataclass
class HatchLevel:
    max_gray: float  # gray < max_gray の画素をこのレベルの対象とする(小さいほど暗い領域のみ)
    angles_deg: list[float]


_DEFAULT_ANGLE_SETS: list[list[float]] = [
    [45.0],
    [45.0, 135.0],
    [45.0, 135.0, 0.0],
    [45.0, 135.0, 0.0, 90.0],
]


def auto_hatch_levels(
    gray: np.ndarray,
    n_levels: int = 4,
    dark_percentile_max: float = 40.0,
    angle_sets: list[list[float]] | None = None,
) -> list[HatchLevel]:
    """画像の明度分布(パーセンタイル)からハッチングしきい値を自動算出する。

    固定のグレースケール値(例: 200/150/100/50)は、露出・コントラストが
    画像ごとに大きく異なる写真では「常に真っ黒」「常に真っ白」に破綻し
    やすい。暗い方からdark_percentile_max%までをn_levels段階に均等分割
    することで、画像ごとの明度分布に関わらず安定して陰影階調を再現する。
    """
    angle_sets = angle_sets or _DEFAULT_ANGLE_SETS
    percentiles = np.linspace(dark_percentile_max, dark_percentile_max / n_levels, n_levels)
    thresholds = np.percentile(gray, percentiles)
    return [
        HatchLevel(float(th), angle_sets[min(i, len(angle_sets) - 1)]) for i, th in enumerate(thresholds)
    ]


@dataclass
class HatchingConfig:
    levels: list[HatchLevel] | None = None  # Noneなら画像の明度分布から自動算出する
    n_levels: int = 4  # 自動算出時の階調段階数
    dark_percentile_max: float = 40.0  # 自動算出時、画像の暗い方から何%までをハッチング対象にするか
    spacing_px: float = 6.0  # 平行線の間隔
    step_px: float = 1.0  # 線上のサンプリング間隔(細かいほど輪郭のクリップが正確)
    min_segment_len_px: float = 3.0  # これより短い線分は描かない(ノイズ的な点を除外)

    def resolve_levels(self, gray: np.ndarray) -> list[HatchLevel]:
        if self.levels is not None:
            return self.levels
        return auto_hatch_levels(gray, n_levels=self.n_levels, dark_percentile_max=self.dark_percentile_max)


def _extract_segments(ink: np.ndarray, pts: np.ndarray, min_len_px: float) -> list[np.ndarray]:
    """bool配列inkの連続True区間を、対応する座標列ptsから切り出す。"""
    if not ink.any():
        return []
    idx = np.flatnonzero(ink)
    breaks = np.flatnonzero(np.diff(idx) > 1)
    starts = np.concatenate(([0], breaks + 1))
    ends = np.concatenate((breaks, [len(idx) - 1]))

    segments: list[np.ndarray] = []
    for s, e in zip(starts, ends):
        i0, i1 = idx[s], idx[e]
        if i1 - i0 < 1:
            continue
        seg = pts[i0 : i1 + 1]
        if float(np.hypot(*(seg[-1] - seg[0]))) >= min_len_px:
            segments.append(seg)
    return segments


def generate_hatching(gray: np.ndarray, config: HatchingConfig | None = None) -> list[Polyline]:
    """グレースケール画像から陰影表現用のハッチング線分群を生成する(px空間, row/col)。

    角度ごとに全平行線×全サンプル点の座標を(n_lines, n_steps)の2次元配列として
    一括生成し、マスク参照もベクトル化する。Pythonループは「インクを含む行
    (=実際に線分が存在する平行線)」のセグメント抽出にのみ残す。
    """
    config = config or HatchingConfig()
    h, w = gray.shape
    center = np.array([h / 2.0, w / 2.0])
    diag = float(np.hypot(h, w))
    half_len = diag / 2.0 + 1.0

    n_steps = max(2, int(2 * half_len / config.step_px) + 1)
    ts = np.linspace(-half_len, half_len, n_steps)

    polylines: list[Polyline] = []
    for level in config.resolve_levels(gray):
        mask = gray < level.max_gray
        if not np.any(mask):
            continue
        for angle_deg in level.angles_deg:
            theta = np.radians(angle_deg)
            direction = np.array([np.sin(theta), np.cos(theta)])  # (drow, dcol)
            normal = np.array([-np.cos(theta), np.sin(theta)])

            n_lines = max(1, int(2 * half_len / config.spacing_px) + 1)
            offsets = np.linspace(-half_len, half_len, n_lines)

            base = center[None, :] + offsets[:, None] * normal[None, :]  # (n_lines, 2)
            rows = base[:, 0:1] + ts[None, :] * direction[0]  # (n_lines, n_steps)
            cols = base[:, 1:2] + ts[None, :] * direction[1]  # (n_lines, n_steps)

            in_bounds = (rows >= 0) & (rows < h) & (cols >= 0) & (cols < w)
            ri = np.clip(rows.astype(int), 0, h - 1)
            ci = np.clip(cols.astype(int), 0, w - 1)
            ink = np.zeros_like(in_bounds)
            ink[in_bounds] = mask[ri[in_bounds], ci[in_bounds]]

            for li in np.flatnonzero(ink.any(axis=1)):
                pts = np.stack([rows[li], cols[li]], axis=1)
                for seg in _extract_segments(ink[li], pts, config.min_segment_len_px):
                    polylines.append(Polyline(points=seg, closed=False))

    return polylines
