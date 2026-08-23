"""GUI: JPGファイル選択・パラメータ調整・プレビュー表示・SVG/G-code保存。

処理(XDoG+ハッチング+順序最適化)は数秒〜十数秒かかり得るため、UIが固まらない
よう別スレッドで実行し、queueで結果をメインスレッドへ渡す。
"""
from __future__ import annotations

import json
import queue
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import matplotlib

matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from .gcode_export import export_gcode
from .hatching import HatchingConfig
from .pen_control import GpioPenController, PenController, ServoAnglePenController, ZAxisPenController
from .pipeline import PipelineResult, XYPDrawConfig, process_image
from .svg_export import export_svg
from .types import PlotJob

_JP_FONT_CANDIDATES = ["Yu Gothic", "Meiryo", "MS Gothic", "Noto Sans CJK JP"]
_SETTINGS_PATH = Path.home() / ".xypdraw_gui_settings.json"


def _configure_japanese_font() -> None:
    matplotlib.rcParams["font.family"] = "sans-serif"
    matplotlib.rcParams["font.sans-serif"] = _JP_FONT_CANDIDATES + list(
        matplotlib.rcParams["font.sans-serif"]
    )
    matplotlib.rcParams["axes.unicode_minus"] = False


class XYPDrawApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("XYPDraw - JPG線画プロッターツール")
        self.root.geometry("1200x800")

        self.image_path: str | None = None
        self.result: PipelineResult | None = None
        self._worker_queue: queue.Queue = queue.Queue()

        settings = self._load_settings()
        self.vars: dict[str, tk.Variable] = {}
        self._build_ui(settings)

        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    # ---- 設定の永続化 ----
    def _load_settings(self) -> dict:
        if _SETTINGS_PATH.exists():
            try:
                return json.loads(_SETTINGS_PATH.read_text(encoding="utf-8"))
            except Exception:
                return {}
        return {}

    def _save_settings(self) -> None:
        data = {k: v.get() for k, v in self.vars.items()}
        try:
            _SETTINGS_PATH.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:
            pass

    def _on_close(self) -> None:
        self._save_settings()
        self.root.destroy()

    # ---- UI構築 ----
    def _build_ui(self, s: dict) -> None:
        _configure_japanese_font()

        top = ttk.Frame(self.root, padding=8)
        top.pack(side=tk.TOP, fill=tk.X)
        ttk.Label(top, text="画像ファイル:").pack(side=tk.LEFT)
        self.path_var = tk.StringVar(value=s.get("path", ""))
        ttk.Entry(top, textvariable=self.path_var, width=70).pack(side=tk.LEFT, padx=4)
        ttk.Button(top, text="参照...", command=self._browse_image).pack(side=tk.LEFT, padx=4)

        body = ttk.Frame(self.root)
        body.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        params_container = ttk.Frame(body, width=340)
        params_container.pack(side=tk.LEFT, fill=tk.Y)
        params_container.pack_propagate(False)
        canvas = tk.Canvas(params_container, borderwidth=0, highlightthickness=0)
        scrollbar = ttk.Scrollbar(params_container, orient=tk.VERTICAL, command=canvas.yview)
        params = ttk.Frame(canvas)
        params.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=params, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        def add_section(title: str) -> None:
            ttk.Label(params, text=title, font=("", 10, "bold")).pack(anchor="w", pady=(10, 2), padx=6)

        def add_float(key: str, label: str, default: float) -> None:
            var = tk.DoubleVar(value=float(s.get(key, default)))
            self.vars[key] = var
            row = ttk.Frame(params)
            row.pack(fill=tk.X, padx=6, pady=1)
            ttk.Label(row, text=label, width=22).pack(side=tk.LEFT)
            ttk.Entry(row, textvariable=var, width=10).pack(side=tk.LEFT)

        def add_int(key: str, label: str, default: int) -> None:
            var = tk.IntVar(value=int(s.get(key, default)))
            self.vars[key] = var
            row = ttk.Frame(params)
            row.pack(fill=tk.X, padx=6, pady=1)
            ttk.Label(row, text=label, width=22).pack(side=tk.LEFT)
            ttk.Entry(row, textvariable=var, width=10).pack(side=tk.LEFT)

        def add_bool(key: str, label: str, default: bool) -> None:
            var = tk.BooleanVar(value=bool(s.get(key, default)))
            self.vars[key] = var
            ttk.Checkbutton(params, text=label, variable=var).pack(anchor="w", padx=6, pady=1)

        add_section("処理解像度")
        add_int("max_long_side_px", "処理解像度上限(px)", 1600)

        add_section("前処理")
        add_int("bilateral_d", "バイラテラル直径", 5)
        add_float("bilateral_sigma_color", "バイラテラル色シグマ", 50.0)
        add_float("bilateral_sigma_space", "バイラテラル空間シグマ", 50.0)
        add_float("clahe_clip_limit", "CLAHEコントラスト上限", 2.0)

        add_section("XDoG(輪郭)")
        add_float("xdog_sigma", "sigma", 1.2)
        add_float("xdog_k", "k", 1.6)
        add_float("xdog_tau", "tau", 0.98)
        add_float("xdog_epsilon", "epsilon", -0.01)
        add_float("xdog_phi", "phi", 200.0)
        add_float("xdog_threshold", "二値化しきい値(0-1)", 0.5)
        add_int("min_object_size_px", "最小成分サイズ(px)", 4)
        add_float("spur_factor", "スパー除去係数", 1.4)
        add_float("merge_factor", "交差点集約係数", 0.85)

        add_section("ハッチング(陰影)")
        add_bool("enable_hatching", "ハッチングを有効化", True)
        add_float("hatch_spacing_px", "線間隔(px)", 6.0)
        add_float("hatch_min_segment_px", "最小線分長(px)", 3.0)
        add_int("hatch_n_levels", "階調段階数", 4)
        add_float("hatch_dark_percentile_max", "暗い方から対象にする割合(%)", 40.0)

        add_section("出力")
        add_float("target_long_side_mm", "出力サイズ長辺(mm)", 200.0)
        add_float("origin_offset_x_mm", "原点オフセットX(mm)", 0.0)
        add_float("origin_offset_y_mm", "原点オフセットY(mm)", 0.0)
        add_float("simplify_tolerance_mm", "単純化許容誤差(mm, 0=無効)", 0.0)
        add_float("pen_width_mm", "プレビュー線幅(mm)", 0.3)
        add_bool("show_travel", "ペンアップ移動線を表示", False)

        add_section("ペン制御(G-code)")
        pen_mode_var = tk.StringVar(value=s.get("pen_mode", "z"))
        self.vars["pen_mode"] = pen_mode_var
        row = ttk.Frame(params)
        row.pack(fill=tk.X, padx=6, pady=1)
        ttk.Label(row, text="ペン制御方式", width=22).pack(side=tk.LEFT)
        ttk.Combobox(
            row, textvariable=pen_mode_var, values=["z", "servo", "gpio"], width=8, state="readonly"
        ).pack(side=tk.LEFT)
        add_float("pen_up_z", "ペンアップZ(mm)", 5.0)
        add_float("pen_down_z", "ペンダウンZ(mm)", 0.0)
        add_float("feed_rate", "描画送り速度", 1500.0)
        add_float("travel_feed_rate", "移動送り速度", 3000.0)

        # ---- 右側: プレビュー ----
        right = ttk.Frame(body)
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=8, pady=8)

        self.figure = Figure(figsize=(6, 6))
        self.ax = self.figure.add_subplot(111)
        self.canvas_widget = FigureCanvasTkAgg(self.figure, master=right)
        self.canvas_widget.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        bottom = ttk.Frame(self.root, padding=8)
        bottom.pack(side=tk.BOTTOM, fill=tk.X)
        self.generate_btn = ttk.Button(bottom, text="線画生成", command=self._on_generate)
        self.generate_btn.pack(side=tk.LEFT, padx=4)
        ttk.Button(bottom, text="SVG保存", command=self._on_save_svg).pack(side=tk.LEFT, padx=4)
        ttk.Button(bottom, text="G-code保存", command=self._on_save_gcode).pack(side=tk.LEFT, padx=4)
        self.status_var = tk.StringVar(value="画像を選択して「線画生成」を押してください。")
        ttk.Label(bottom, textvariable=self.status_var).pack(side=tk.LEFT, padx=12)

    # ---- ファイル選択 ----
    def _browse_image(self) -> None:
        path = filedialog.askopenfilename(
            title="画像ファイルを選択",
            filetypes=[("画像ファイル", "*.jpg *.jpeg *.png *.bmp *.webp"), ("すべてのファイル", "*.*")],
        )
        if path:
            self.path_var.set(path)

    # ---- 設定 -> Config ----
    def _current_config(self) -> XYPDrawConfig:
        v = {k: var.get() for k, var in self.vars.items()}
        simplify = v["simplify_tolerance_mm"]
        return XYPDrawConfig(
            max_long_side_px=int(v["max_long_side_px"]),
            bilateral_d=int(v["bilateral_d"]),
            bilateral_sigma_color=v["bilateral_sigma_color"],
            bilateral_sigma_space=v["bilateral_sigma_space"],
            clahe_clip_limit=v["clahe_clip_limit"],
            xdog_sigma=v["xdog_sigma"],
            xdog_k=v["xdog_k"],
            xdog_tau=v["xdog_tau"],
            xdog_epsilon=v["xdog_epsilon"],
            xdog_phi=v["xdog_phi"],
            xdog_threshold=v["xdog_threshold"],
            min_object_size_px=int(v["min_object_size_px"]),
            spur_factor=v["spur_factor"],
            merge_factor=v["merge_factor"],
            enable_hatching=bool(v["enable_hatching"]),
            hatching_config=HatchingConfig(
                n_levels=int(v["hatch_n_levels"]),
                dark_percentile_max=v["hatch_dark_percentile_max"],
                spacing_px=v["hatch_spacing_px"],
                min_segment_len_px=v["hatch_min_segment_px"],
            ),
            target_long_side_mm=v["target_long_side_mm"],
            origin_offset_mm=(v["origin_offset_x_mm"], v["origin_offset_y_mm"]),
            simplify_tolerance_mm=simplify if simplify > 0 else None,
        )

    def _build_pen_controller(self) -> PenController:
        mode = self.vars["pen_mode"].get()
        if mode == "z":
            return ZAxisPenController(up_z=self.vars["pen_up_z"].get(), down_z=self.vars["pen_down_z"].get())
        if mode == "servo":
            return ServoAnglePenController()
        return GpioPenController()

    # ---- 生成 ----
    def _on_generate(self) -> None:
        path = self.path_var.get().strip()
        if not path:
            messagebox.showwarning("XYPDraw", "画像ファイルを選択してください。")
            return
        if not Path(path).exists():
            messagebox.showerror("XYPDraw", f"ファイルが見つかりません: {path}")
            return

        try:
            config = self._current_config()
        except (tk.TclError, ValueError) as exc:
            messagebox.showerror("XYPDraw", f"パラメータの値が不正です: {exc}")
            return

        self.image_path = path
        self.generate_btn.config(state=tk.DISABLED)
        self.status_var.set("生成中...")

        def worker() -> None:
            try:
                result = process_image(path, config)
                self._worker_queue.put(("ok", result))
            except Exception as exc:  # noqa: BLE001
                self._worker_queue.put(("error", exc))

        threading.Thread(target=worker, daemon=True).start()
        self.root.after(100, self._poll_worker)

    def _poll_worker(self) -> None:
        try:
            status, payload = self._worker_queue.get_nowait()
        except queue.Empty:
            self.root.after(100, self._poll_worker)
            return

        self.generate_btn.config(state=tk.NORMAL)
        if status == "error":
            messagebox.showerror("XYPDraw", f"生成に失敗しました: {payload}")
            self.status_var.set("生成に失敗しました。")
            return

        self.result = payload
        assert self.result.job is not None
        self._render_preview(self.result.job)
        msg = self.result.job.stats.summary()
        if self.result.warnings:
            msg += f" / 警告{len(self.result.warnings)}件"
        self.status_var.set(msg)

    def _render_preview(self, job: PlotJob) -> None:
        self.ax.clear()
        width_mm, height_mm = job.canvas_size_mm
        pen_width_mm = self.vars["pen_width_mm"].get()
        show_travel = bool(self.vars["show_travel"].get())
        linewidth_pt = pen_width_mm / 25.4 * 72

        prev_end = None
        for poly in job.polylines:
            pts = poly.points
            if len(pts) == 0:
                continue
            self.ax.plot(pts[:, 0], pts[:, 1], color="black", linewidth=linewidth_pt, solid_capstyle="round")
            if show_travel and prev_end is not None:
                self.ax.plot(
                    [prev_end[0], pts[0, 0]], [prev_end[1], pts[0, 1]], color="red", linestyle=":", linewidth=0.5
                )
            prev_end = pts[-1]

        self.ax.set_xlim(0, width_mm)
        self.ax.set_ylim(-height_mm, 0)
        self.ax.set_aspect("equal")
        self.figure.tight_layout()
        self.canvas_widget.draw()

    # ---- 保存 ----
    def _require_result(self) -> PlotJob | None:
        if self.result is None or self.result.job is None:
            messagebox.showwarning("XYPDraw", "先に「線画生成」を実行してください。")
            return None
        return self.result.job

    def _on_save_svg(self) -> None:
        job = self._require_result()
        if job is None:
            return
        path = filedialog.asksaveasfilename(defaultextension=".svg", filetypes=[("SVG", "*.svg")])
        if not path:
            return
        export_svg(job, path, stroke_width_mm=self.vars["pen_width_mm"].get())
        self.status_var.set(f"SVGを保存しました: {path}")

    def _on_save_gcode(self) -> None:
        job = self._require_result()
        if job is None:
            return
        path = filedialog.asksaveasfilename(defaultextension=".gcode", filetypes=[("G-code", "*.gcode")])
        if not path:
            return
        pen = self._build_pen_controller()
        export_gcode(
            job,
            path,
            pen=pen,
            feed_rate=self.vars["feed_rate"].get(),
            travel_feed_rate=self.vars["travel_feed_rate"].get(),
        )
        self.status_var.set(f"G-codeを保存しました: {path}")


def main() -> None:
    root = tk.Tk()
    XYPDrawApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
