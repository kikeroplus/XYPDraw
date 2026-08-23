"""XDoG (eXtended Difference-of-Gaussians) による輪郭線抽出。

Winnemoeller et al. の XDoG (2012) に基づく実装。通常のCanny法より
質感ノイズに強く、滑らかで連続的な「鉛筆スケッチ」風の輪郭線を安定して
抽出できるため、写真から線画を作る用途の起点として採用する。
"""
from __future__ import annotations

import cv2
import numpy as np


def xdog(
    gray: np.ndarray,
    sigma: float = 0.8,
    k: float = 1.6,
    tau: float = 0.98,
    epsilon: float = -0.01,
    phi: float = 200.0,
) -> np.ndarray:
    """XDoGを適用し、[0, 1]の連続値画像(1.0=白/背景、0に近いほど線)を返す。

    Args:
        gray: uint8グレースケール画像
        sigma: 基準ガウシアンぼかしの標準偏差(px)。大きいほど太く大まかな輪郭になる
        k: 2つ目のガウシアンのsigma倍率(通常1.6程度)
        tau: DoGの減算係数。小さいほど線が濃く・太くなる(通常0.9〜0.98)
        epsilon: ソフト閾値化の閾値。DoGの値域は画素値0-1スケールでおおよそ
            ±0.05程度に収まることが多いため、その範囲内の小さな値(-0.01前後)にする
        phi: ソフト閾値化の急峻さ。epsilonがこのスケールのため、大きめの値
            (100〜300程度)でないと二値に近づかない
    """
    img = gray.astype(np.float64) / 255.0
    g1 = cv2.GaussianBlur(img, (0, 0), sigmaX=sigma)
    g2 = cv2.GaussianBlur(img, (0, 0), sigmaX=sigma * k)
    dog = g1 - tau * g2

    result = np.where(dog >= epsilon, 1.0, 1.0 + np.tanh(phi * (dog - epsilon)))
    return np.clip(result, 0.0, 1.0)


def xdog_binary(
    gray: np.ndarray,
    sigma: float = 0.8,
    k: float = 1.6,
    tau: float = 0.98,
    epsilon: float = -0.01,
    phi: float = 200.0,
    threshold: float = 0.5,
) -> np.ndarray:
    """XDoG出力をしきい値で二値化する。True=線(インク)。"""
    soft = xdog(gray, sigma=sigma, k=k, tau=tau, epsilon=epsilon, phi=phi)
    return soft < threshold
