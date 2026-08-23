"""G-code生成。

単位はG21(mm)、座標系はG90(絶対座標)。ペンアップ/ダウン制御は
pen_control.PenController に委譲するため、実プロッターの制御方式が
変わってもインスタンスを差し替えるだけでよい。

XYPWriter (xypwriter/gcode_export.py) から移植した汎用ロジック。
"""
from __future__ import annotations

from pathlib import Path

from .pen_control import PenController, ZAxisPenController
from .types import PlotJob


def build_gcode_lines(
    job: PlotJob,
    pen: PenController | None = None,
    feed_rate: float = 1500.0,
    travel_feed_rate: float = 3000.0,
) -> list[str]:
    pen = pen or ZAxisPenController()
    lines: list[str] = []
    lines.extend(pen.setup())
    lines.append(f"G0 F{travel_feed_rate:.1f}")
    lines.extend(pen.pen_up())

    for poly in job.polylines:
        if len(poly.points) == 0:
            continue
        x0, y0 = poly.points[0]
        lines.append(f"G0 X{x0:.3f} Y{y0:.3f}")
        lines.extend(pen.pen_down())
        lines.append(f"G1 F{feed_rate:.1f}")
        for x, y in poly.points[1:]:
            lines.append(f"G1 X{x:.3f} Y{y:.3f}")
        lines.extend(pen.pen_up())
        lines.append(f"G0 F{travel_feed_rate:.1f}")

    lines.extend(pen.teardown())
    return lines


def build_outline_check_gcode(width_mm: float, height_mm: float, feed_rate: float = 100.0) -> list[str]:
    """描画範囲の外周だけをペンアップのままなぞる、物理的な範囲安全確認用G-code。

    原点復帰やソフト/ハードリミットが無効な機体では、電源投入時の
    キャリッジ位置がそのまま機械原点として扱われるため、実際に描画する前に
    「この矩形の範囲が本当に安全に動けるか」を、インクを一切出さずに
    低速で確認できるようにする。原点(0,0)は「画像の左上」に対応させる想定
    のため、矩形はX+(右)・Y-(下)方向に展開する。
    """
    x0, y0 = 0.0, 0.0
    x1, y1 = width_mm, -height_mm
    lines = [
        "G21",
        "G90",
        f"G1 F{feed_rate:.1f}",
        f"G1 X{x0:.3f} Y{y0:.3f}",
        f"G1 X{x1:.3f} Y{y0:.3f}",
        f"G1 X{x1:.3f} Y{y1:.3f}",
        f"G1 X{x0:.3f} Y{y1:.3f}",
        f"G1 X{x0:.3f} Y{y0:.3f}",
    ]
    return lines


def export_gcode(
    job: PlotJob,
    path: str | Path,
    pen: PenController | None = None,
    feed_rate: float = 1500.0,
    travel_feed_rate: float = 3000.0,
) -> None:
    lines = build_gcode_lines(job, pen=pen, feed_rate=feed_rate, travel_feed_rate=travel_feed_rate)
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")
