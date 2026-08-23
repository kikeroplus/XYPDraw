"""CLI: `python -m xypdraw input.jpg --out-dir output --svg --gcode`

JPG(等のラスタ画像)を入力とし、XDoGによる輪郭抽出とクロスハッチングによる
陰影表現を組み合わせた線画に変換し、プレビュー画像/SVG/G-codeを出力する。
"""
from __future__ import annotations

import argparse
from pathlib import Path

from .gcode_export import export_gcode
from .hatching import HatchingConfig
from .pen_control import GpioPenController, PenController, ServoAnglePenController, ZAxisPenController
from .pipeline import XYPDrawConfig, process_image
from .preview import render_job_preview, render_stage_debug
from .svg_export import export_svg


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="xypdraw", description="JPG画像 -> 線画(XDoG+ハッチング) -> G-code 変換パイプライン"
    )
    parser.add_argument("image", type=Path, help="入力画像ファイルパス(JPG等)")
    parser.add_argument("--out-dir", type=Path, default=Path("output"), help="出力先ディレクトリ")
    parser.add_argument(
        "--max-long-side-px", type=int, default=1600, help="処理解像度の上限(長辺px)。大きいほど精細だが低速"
    )
    # 前処理
    parser.add_argument("--bilateral-d", type=int, default=5, help="バイラテラルフィルタの近傍直径")
    parser.add_argument("--bilateral-sigma-color", type=float, default=50.0, help="バイラテラルフィルタの色空間シグマ")
    parser.add_argument("--bilateral-sigma-space", type=float, default=50.0, help="バイラテラルフィルタの空間シグマ")
    parser.add_argument("--clahe-clip-limit", type=float, default=2.0, help="CLAHEのコントラスト強調上限")
    # XDoG
    parser.add_argument("--xdog-sigma", type=float, default=1.2, help="XDoG基準ガウシアンの標準偏差(px)")
    parser.add_argument("--xdog-k", type=float, default=1.6, help="XDoG第2ガウシアンのsigma倍率")
    parser.add_argument("--xdog-tau", type=float, default=0.98, help="XDoGのDoG減算係数(小さいほど線が太くなる)")
    parser.add_argument("--xdog-epsilon", type=float, default=-0.01, help="XDoGソフト閾値化の閾値")
    parser.add_argument("--xdog-phi", type=float, default=200.0, help="XDoGソフト閾値化の急峻さ")
    parser.add_argument("--xdog-threshold", type=float, default=0.5, help="XDoG出力の二値化しきい値(0-1)")
    parser.add_argument("--min-object-size-px", type=int, default=4, help="ノイズとみなす最小連結成分サイズ(px)")
    parser.add_argument("--spur-factor", type=float, default=1.4, help="スパー(ヒゲ)除去のストローク幅倍率")
    parser.add_argument("--merge-factor", type=float, default=0.85, help="交差点集約半径のストローク幅倍率")
    # ハッチング
    parser.add_argument("--no-hatching", action="store_true", help="陰影ハッチングを無効化し、輪郭線のみ出力する")
    parser.add_argument("--hatch-spacing-px", type=float, default=6.0, help="ハッチング平行線の間隔(px)")
    parser.add_argument(
        "--hatch-min-segment-px", type=float, default=3.0, help="これより短いハッチング線分は描かない"
    )
    parser.add_argument(
        "--hatch-n-levels", type=int, default=4, help="ハッチングの階調段階数(暗さに応じて重ねる角度数を増やす段数)"
    )
    parser.add_argument(
        "--hatch-dark-percentile-max",
        type=float,
        default=40.0,
        help="画像の暗い方から何%%までをハッチング対象にするか(明度分布から自動でしきい値を算出する)",
    )
    # 出力
    parser.add_argument("--target-long-side-mm", type=float, default=200.0, help="プロッター出力サイズ(長辺mm)")
    parser.add_argument(
        "--origin-offset-mm", type=float, nargs=2, metavar=("X", "Y"), default=(0.0, 0.0), help="プロッター原点オフセット(mm)"
    )
    parser.add_argument(
        "--simplify-tolerance-mm",
        type=float,
        default=None,
        help="指定するとポリラインをこの許容誤差(mm)でDouglas-Peucker単純化する(高速描画モード)",
    )
    parser.add_argument("--pen-width-mm", type=float, default=0.3, help="プレビュー用ペン幅(mm)")
    parser.add_argument("--show-travel", action="store_true", help="プレビューにペンアップ移動線(赤点線)を表示する")
    parser.add_argument("--svg", action="store_true", help="SVGファイルも出力する")
    parser.add_argument("--gcode", action="store_true", help="G-codeファイルも出力する")
    parser.add_argument("--debug", action="store_true", help="各ステージの中間結果画像を出力する")
    parser.add_argument(
        "--pen-mode", choices=["z", "servo", "gpio"], default="z", help="G-code出力時のペンアップ/ダウン制御方式"
    )
    parser.add_argument("--pen-up-z", type=float, default=5.0, help="pen-mode=z時のペンアップZ座標(mm)")
    parser.add_argument("--pen-down-z", type=float, default=0.0, help="pen-mode=z時のペンダウンZ座標(mm)")
    parser.add_argument("--feed-rate", type=float, default=1500.0, help="G-code描画時の送り速度")
    parser.add_argument("--travel-feed-rate", type=float, default=3000.0, help="G-codeペンアップ移動時の送り速度")
    return parser


def _build_pen_controller(args: argparse.Namespace) -> PenController:
    if args.pen_mode == "z":
        return ZAxisPenController(up_z=args.pen_up_z, down_z=args.pen_down_z)
    if args.pen_mode == "servo":
        return ServoAnglePenController()
    return GpioPenController()


def _build_hatching_config(args: argparse.Namespace) -> HatchingConfig:
    return HatchingConfig(
        n_levels=args.hatch_n_levels,
        dark_percentile_max=args.hatch_dark_percentile_max,
        spacing_px=args.hatch_spacing_px,
        min_segment_len_px=args.hatch_min_segment_px,
    )


def main(argv: list[str] | None = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    config = XYPDrawConfig(
        max_long_side_px=args.max_long_side_px,
        bilateral_d=args.bilateral_d,
        bilateral_sigma_color=args.bilateral_sigma_color,
        bilateral_sigma_space=args.bilateral_sigma_space,
        clahe_clip_limit=args.clahe_clip_limit,
        xdog_sigma=args.xdog_sigma,
        xdog_k=args.xdog_k,
        xdog_tau=args.xdog_tau,
        xdog_epsilon=args.xdog_epsilon,
        xdog_phi=args.xdog_phi,
        xdog_threshold=args.xdog_threshold,
        min_object_size_px=args.min_object_size_px,
        spur_factor=args.spur_factor,
        merge_factor=args.merge_factor,
        enable_hatching=not args.no_hatching,
        hatching_config=_build_hatching_config(args),
        target_long_side_mm=args.target_long_side_mm,
        origin_offset_mm=tuple(args.origin_offset_mm),
        simplify_tolerance_mm=args.simplify_tolerance_mm,
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)

    result = process_image(args.image, config)
    assert result.job is not None

    print(f"輪郭trail数: {len(result.contour_polylines_px)}, ハッチング線分数: {len(result.hatching_polylines_px)}")
    for w in result.warnings:
        print("[警告]", w)
    print(result.job.stats.summary())

    stem = args.image.stem

    if args.debug:
        fig = render_stage_debug(result)
        debug_path = args.out_dir / f"{stem}_debug.png"
        fig.savefig(debug_path, dpi=150)
        print(f"デバッグ画像を保存しました: {debug_path}")

    preview_fig = render_job_preview(result.job, pen_width_mm=args.pen_width_mm, show_travel=args.show_travel)
    preview_path = args.out_dir / f"{stem}_preview.png"
    preview_fig.savefig(preview_path, dpi=150)
    print(f"プレビュー画像を保存しました: {preview_path}")

    if args.svg:
        svg_path = args.out_dir / f"{stem}.svg"
        export_svg(result.job, svg_path, stroke_width_mm=args.pen_width_mm)
        print(f"SVGを保存しました: {svg_path}")

    if args.gcode:
        pen = _build_pen_controller(args)
        gcode_path = args.out_dir / f"{stem}.gcode"
        export_gcode(result.job, gcode_path, pen=pen, feed_rate=args.feed_rate, travel_feed_rate=args.travel_feed_rate)
        print(f"G-codeを保存しました: {gcode_path} (pen-mode={args.pen_mode})")


if __name__ == "__main__":
    main()
