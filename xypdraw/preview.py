"""matplotlibによる中間結果・最終結果の可視化。計算コアはここに依存しない。"""
from __future__ import annotations

import numpy as np

from .pipeline import PipelineResult
from .types import PlotJob

_JP_FONT_CANDIDATES = ["Yu Gothic", "Meiryo", "MS Gothic", "Noto Sans CJK JP"]


def _configure_japanese_font() -> None:
    import matplotlib

    matplotlib.rcParams["font.family"] = "sans-serif"
    matplotlib.rcParams["font.sans-serif"] = _JP_FONT_CANDIDATES + list(
        matplotlib.rcParams["font.sans-serif"]
    )
    matplotlib.rcParams["axes.unicode_minus"] = False


def render_job_preview(job: PlotJob, pen_width_mm: float = 0.3, show_travel: bool = False):
    """確定した描画順でプロッター出力結果をプレビューする。"""
    import matplotlib.pyplot as plt

    _configure_japanese_font()
    width_mm, height_mm = job.canvas_size_mm
    fig_w = max(width_mm / 25.4, 4.0)
    fig_h = max(height_mm / 25.4, 3.0) + 0.4
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    linewidth_pt = pen_width_mm / 25.4 * 72
    prev_end: np.ndarray | None = None
    for poly in job.polylines:
        pts = poly.points
        if len(pts) == 0:
            continue
        ax.plot(pts[:, 0], pts[:, 1], color="black", linewidth=linewidth_pt, solid_capstyle="round")
        if show_travel and prev_end is not None:
            ax.plot(
                [prev_end[0], pts[0, 0]],
                [prev_end[1], pts[0, 1]],
                color="red",
                linestyle=":",
                linewidth=0.5,
            )
        prev_end = pts[-1]

    ax.set_xlim(0, width_mm)
    ax.set_ylim(-height_mm, 0)
    ax.set_aspect("equal")
    ax.set_title(job.stats.summary(), fontsize=9, wrap=True)
    fig.tight_layout()
    return fig


def render_stage_debug(result: PipelineResult):
    """パイプライン中間結果(前処理〜輪郭抽出〜ハッチング)を並べて可視化する。"""
    import matplotlib.pyplot as plt

    _configure_japanese_font()
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    ax = axes.ravel()

    ax[0].imshow(result.gray, cmap="gray")
    ax[0].set_title("1. 前処理後グレースケール(CLAHE適用後)")

    ax[1].imshow(result.edge_mask, cmap="gray_r")
    ax[1].set_title("2. XDoG二値マスク")

    ax[2].imshow(np.zeros_like(result.gray), cmap="gray")
    for poly in result.contour_polylines_px:
        pts = poly.points
        ax[2].plot(pts[:, 1], pts[:, 0], color="orange", linewidth=1.0)
    ax[2].set_title(f"3. 輪郭trail抽出 ({len(result.contour_polylines_px)}本)")

    ax[3].imshow(np.zeros_like(result.gray), cmap="gray")
    for poly in result.hatching_polylines_px:
        pts = poly.points
        ax[3].plot(pts[:, 1], pts[:, 0], color="cyan", linewidth=0.5)
    ax[3].set_title(f"4. ハッチング ({len(result.hatching_polylines_px)}本)")

    for a in ax:
        a.axis("off")
        a.set_aspect("equal")
    fig.tight_layout()
    return fig
