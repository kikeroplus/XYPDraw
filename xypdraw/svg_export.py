"""SVG出力（プレビュー用、ポリラインとして）。"""
from __future__ import annotations

from pathlib import Path

from .types import PlotJob


def build_svg(job: PlotJob, stroke_width_mm: float = 0.5, stroke_color: str = "black") -> str:
    width_mm, height_mm = job.canvas_size_mm
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width_mm}mm" height="{height_mm}mm" '
        f'viewBox="0 0 {width_mm} {height_mm}">',
        f'<rect x="0" y="0" width="{width_mm}" height="{height_mm}" fill="white"/>',
    ]
    for poly in job.polylines:
        pts = poly.points
        if len(pts) == 0:
            continue
        if len(pts) == 1:
            x, y = pts[0]
            parts.append(
                f'<circle cx="{x:.3f}" cy="{-y:.3f}" r="{stroke_width_mm / 2:.3f}" '
                f'fill="{stroke_color}"/>'
            )
            continue
        # mm座標は原点=左上・Y-方向が下(既にSVGと同じ向き)なので符号反転のみで変換できる
        points_attr = " ".join(f"{x:.3f},{-y:.3f}" for x, y in pts)
        parts.append(
            f'<polyline points="{points_attr}" fill="none" stroke="{stroke_color}" '
            f'stroke-width="{stroke_width_mm:.3f}" stroke-linecap="round" stroke-linejoin="round"/>'
        )
    parts.append("</svg>")
    return "\n".join(parts)


def export_svg(
    job: PlotJob, path: str | Path, stroke_width_mm: float = 0.5, stroke_color: str = "black"
) -> None:
    Path(path).write_text(build_svg(job, stroke_width_mm, stroke_color), encoding="utf-8")
