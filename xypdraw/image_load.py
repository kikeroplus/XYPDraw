"""JPG(その他ラスタ画像)の読込・前処理。"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps


def load_grayscale(path: str | Path, max_long_side_px: int | None = 1600) -> np.ndarray:
    """画像を読み込み、EXIF Orientationを反映した上でグレースケール(uint8)として返す。

    max_long_side_pxを指定すると、長辺がこれを超える場合のみ縮小する
    (処理時間とXDoG/ハッチングの線本数を扱いやすい範囲に抑えるため)。
    """
    img = Image.open(path)
    img = ImageOps.exif_transpose(img)
    img = img.convert("L")
    if max_long_side_px is not None:
        w, h = img.size
        long_side = max(w, h)
        if long_side > max_long_side_px:
            ratio = max_long_side_px / long_side
            new_size = (max(1, round(w * ratio)), max(1, round(h * ratio)))
            img = img.resize(new_size, Image.LANCZOS)
    return np.array(img, dtype=np.uint8)


def enhance_contrast(gray: np.ndarray, clip_limit: float = 2.0, tile_grid_size: int = 8) -> np.ndarray:
    """CLAHEでローカルコントラストを強調する(ハッチングの階調分離に使う)。"""
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_grid_size, tile_grid_size))
    return clahe.apply(gray)


def denoise(gray: np.ndarray, d: int = 5, sigma_color: float = 50.0, sigma_space: float = 50.0) -> np.ndarray:
    """バイラテラルフィルタでエッジを保ちながら質感ノイズを平滑化する。"""
    return cv2.bilateralFilter(gray, d, sigma_color, sigma_space)
