"""ストローク幅推定・各種閾値の自動計算。

スパー除去・交差点集約・太さ判定しきい値のいずれも、
「文書ごとに手動チューニングしなくて済む」よう、この基準値を共有する。
XYPWriter (xypwriter/metrics.py) から移植した汎用ロジック。
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import distance_transform_edt
from skimage.measure import label


def count_components(mask: np.ndarray) -> int:
    """2値マスクの8連結成分数を数える。"""
    if not np.any(mask):
        return 0
    _, n = label(mask, connectivity=2, return_num=True)
    return int(n)


def estimate_stroke_width(binary: np.ndarray, skeleton: np.ndarray, dist: np.ndarray | None = None) -> float:
    """二値画像とそのスケルトンから、代表的なストローク幅(px)を推定する。

    塗りつぶし画像の距離変換をスケルトン画素位置でサンプリングし、
    中央値の2倍（中心から両端までの距離の合計）をストローク幅とする。
    スケルトンが空の場合は0.0を返す。

    dist: 呼び出し側が既にdistance_transform_edt(binary)を計算済みなら
        渡すことで再計算を避けられる(remove_spursが同じbinaryに対して
        直後にも距離変換を必要とするため)。
    """
    if not np.any(skeleton):
        return 0.0
    if dist is None:
        dist = distance_transform_edt(binary)
    radii = dist[skeleton]
    radii = radii[radii > 0]
    if radii.size == 0:
        return 0.0
    return float(2.0 * np.median(radii))


def estimate_spur_threshold(stroke_width_px: float, factor: float = 1.4) -> float:
    """スパー（ヒゲ）除去の長さ閾値をストローク幅から自動算出する。"""
    if stroke_width_px <= 0:
        return 0.0
    return factor * stroke_width_px


def estimate_merge_radius(stroke_width_px: float, factor: float = 0.85) -> float:
    """交差点集約（junction merge）の半径をストローク幅から自動算出する。"""
    if stroke_width_px <= 0:
        return 0.0
    return factor * stroke_width_px


def estimate_thickness_threshold(stroke_width_px: float, factor: float = 2.0) -> float:
    """細線/塗り図形の分岐しきい値(px)をストローク幅の定数倍から算出する(フォールバック用)。

    spec上「文字ストローク幅の想定値の2倍程度を初期値とする」という方針の
    素朴な実装。太字見出しやストローク交差部は局所的にこの値を超えやすく、
    文字が誤って「太い(塗り図形)」側に分類される事例が確認されたため、
    通常の自動算出は `estimate_thickness_gap_threshold` (太さ分布のギャップ検出)
    を優先し、この関数は成分数が少なくギャップ検出できない場合のフォールバックに
    ほぼ使われない(閾値を極端に大きくして実質全て「細い」扱いにする)。
    """
    if stroke_width_px <= 0:
        return 0.0
    return factor * stroke_width_px


def estimate_thickness_gap_threshold(
    max_half_widths: np.ndarray, min_gap_ratio: float = 2.5
) -> float | None:
    """連結成分ごとの太さ(半幅)分布の中から、文字と塗り図形を分ける自然な
    「空白(ギャップ)」を検出し、その中点をしきい値(px、半幅の2倍)として返す。

    文字(太字見出しやストローク交差部を含む)は同じ書体内での太さのばらつきが
    せいぜい数割程度に収まるのに対し、塗りつぶし図形(矩形・円・ロゴ等)は
    文字ストロークより1桁以上太いことが多い。この非連続な飛躍を
    「ソートした半幅の隣接比の最大値」として検出することで、ストローク幅の
    絶対値に依存しない、文書ごとに適応的なしきい値を得る。

    半幅の値が少ない、またはmin_gap_ratio以上の明確な飛躍が見つからない場合は
    Noneを返す(呼び出し側は「太さでは判定できない」とみなし、安全側=全て
    「細い」扱いにフォールバックすること)。
    """
    widths = np.asarray(max_half_widths, dtype=float)
    widths = widths[widths > 0]
    if widths.size < 2:
        return None
    sorted_widths = np.sort(widths)
    ratios = sorted_widths[1:] / sorted_widths[:-1]
    gap_idx = int(np.argmax(ratios))
    if ratios[gap_idx] < min_gap_ratio:
        return None
    threshold_half = float(np.sqrt(sorted_widths[gap_idx] * sorted_widths[gap_idx + 1]))
    return threshold_half * 2.0
