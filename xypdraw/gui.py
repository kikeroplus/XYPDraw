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

try:
    from tkinterdnd2 import DND_FILES, TkinterDnD

    _DND_AVAILABLE = True
except ImportError:  # ドラッグ&ドロップは任意機能。未インストールでも通常のTkウィンドウで動く。
    _DND_AVAILABLE = False

from .gcode_export import export_gcode
from .hatching import HatchingConfig
from .pen_control import GpioPenController, PenController, ServoAnglePenController, ZAxisPenController
from .plotter_panel import PlotterPanel
from .pipeline import PipelineResult, XYPDrawConfig, process_image
from .svg_export import export_svg
from .types import PlotJob

_JP_FONT_CANDIDATES = ["Yu Gothic", "Meiryo", "MS Gothic", "Noto Sans CJK JP"]
_SETTINGS_PATH = Path.home() / ".xypdraw_gui_settings.json"
_PRESETS_PATH = Path.home() / ".xypdraw_gui_presets.json"
_IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
_NUM_PRESETS = 5


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
        # タスクバー等を考慮した余白を差し引き、画面より大きくならないように
        # 初期ウィンドウサイズを実ディスプレイに合わせてクランプする。
        screen_w = self.root.winfo_screenwidth()
        screen_h = self.root.winfo_screenheight()
        win_w = min(1200, screen_w - 80)
        win_h = min(800, screen_h - 120)
        self.root.geometry(f"{win_w}x{win_h}")
        self.root.minsize(700, 400)

        self.image_path: str | None = None
        self.result: PipelineResult | None = None
        self._worker_queue: queue.Queue = queue.Queue()
        self._plotter_panel: PlotterPanel | None = None

        settings = self._load_settings()
        self.presets: list[dict | None] = self._load_presets()
        self.preset_buttons: list[ttk.Button] = []
        self.vars: dict[str, tk.Variable] = {}
        self.defaults: dict[str, object] = {}
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
        data["name"] = self.name_var.get()
        try:
            _SETTINGS_PATH.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:
            pass

    def _on_close(self) -> None:
        self._save_settings()
        if self._plotter_panel is not None and self._plotter_panel.winfo_exists():
            self._plotter_panel.shutdown()
        self.root.destroy()

    # ---- プリセットの永続化 ----
    def _load_presets(self) -> list[dict | None]:
        if _PRESETS_PATH.exists():
            try:
                data = json.loads(_PRESETS_PATH.read_text(encoding="utf-8"))
                slots = list(data.get("presets", []))[:_NUM_PRESETS]
                slots += [None] * (_NUM_PRESETS - len(slots))
                return slots
            except Exception:
                pass
        return [None] * _NUM_PRESETS

    def _save_presets(self) -> None:
        try:
            _PRESETS_PATH.write_text(
                json.dumps({"presets": self.presets}, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        except Exception:
            pass

    def _preset_label(self, index: int) -> str:
        slot = self.presets[index]
        if slot and slot.get("name"):
            return slot["name"]
        return f"プリセット{index + 1}"

    def _load_preset(self, index: int) -> None:
        slot = self.presets[index]
        if not slot:
            messagebox.showinfo("XYPDraw", f"プリセット{index + 1}には何も保存されていません。")
            return
        for key, value in slot.get("values", {}).items():
            if key in self.vars:
                try:
                    self.vars[key].set(value)
                except tk.TclError:
                    pass
        self.name_var.set(slot.get("name", ""))
        self.status_var.set(f"プリセット{index + 1}「{self._preset_label(index)}」を読み込みました。")

    def _on_save_preset(self) -> None:
        dialog = tk.Toplevel(self.root)
        dialog.title("プリセットに保存")
        dialog.transient(self.root)
        dialog.grab_set()
        dialog.resizable(False, False)
        ttk.Label(dialog, text="保存先のプリセットを選択してください:", padding=(10, 10, 10, 4)).pack(
            anchor="w"
        )
        for i in range(_NUM_PRESETS):
            slot = self.presets[i]
            label = f"{i + 1}: {slot['name'] if slot and slot.get('name') else '(空)'}"
            ttk.Button(
                dialog, text=label, width=32, command=lambda i=i, d=dialog: self._save_to_preset(i, d)
            ).pack(fill=tk.X, padx=10, pady=2)
        ttk.Button(dialog, text="キャンセル", command=dialog.destroy).pack(pady=(6, 10))

    def _save_to_preset(self, index: int, dialog: tk.Toplevel) -> None:
        existing = self.presets[index]
        if existing is not None:
            if not messagebox.askyesno(
                "XYPDraw",
                f"プリセット{index + 1}「{existing.get('name', '')}」を上書きしますか?",
                parent=dialog,
            ):
                return
        name = self.name_var.get().strip() or f"プリセット{index + 1}"
        values = {k: v.get() for k, v in self.vars.items()}
        self.presets[index] = {"name": name, "values": values}
        self._save_presets()
        self.preset_buttons[index].config(text=self._preset_label(index))
        dialog.destroy()
        self.status_var.set(f"プリセット{index + 1}「{name}」に保存しました。")

    # ---- デフォルト値へのリセット ----
    def _reset_param(self, key: str) -> None:
        if key in self.defaults:
            self.vars[key].set(self.defaults[key])

    def _reset_all_params(self) -> None:
        for key, default in self.defaults.items():
            self.vars[key].set(default)

    # ---- UI構築 ----
    def _build_ui(self, s: dict) -> None:
        _configure_japanese_font()

        top = ttk.Frame(self.root, padding=8)
        top.pack(side=tk.TOP, fill=tk.X)
        drop_hint = "(画像をドラッグ&ドロップも可)" if _DND_AVAILABLE else ""
        ttk.Label(top, text=f"画像ファイル: {drop_hint}").pack(side=tk.LEFT)
        self.path_var = tk.StringVar(value=s.get("path", ""))
        path_entry = ttk.Entry(top, textvariable=self.path_var, width=70)
        path_entry.pack(side=tk.LEFT, padx=4)
        ttk.Button(top, text="参照...", command=self._browse_image).pack(side=tk.LEFT, padx=4)
        self._dnd_targets: list[tk.Widget] = [top, path_entry]

        preset_bar = ttk.Frame(self.root, padding=(8, 0, 8, 8))
        preset_bar.pack(side=tk.TOP, fill=tk.X)
        ttk.Label(preset_bar, text="設定名:").pack(side=tk.LEFT)
        self.name_var = tk.StringVar(value=s.get("name", ""))
        ttk.Entry(preset_bar, textvariable=self.name_var, width=16).pack(side=tk.LEFT, padx=(2, 12))
        ttk.Label(preset_bar, text="プリセット:").pack(side=tk.LEFT)
        for i in range(_NUM_PRESETS):
            btn = ttk.Button(preset_bar, text=self._preset_label(i), command=lambda i=i: self._load_preset(i))
            btn.pack(side=tk.LEFT, padx=2)
            self.preset_buttons.append(btn)
        ttk.Button(preset_bar, text="現在の設定を保存...", command=self._on_save_preset).pack(
            side=tk.LEFT, padx=(10, 0)
        )

        body = ttk.Frame(self.root)
        body.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        params_container = ttk.Frame(body, width=400)
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

        # マウスホイールでスクロールできるようにする(スクロールバーのドラッグだけだと
        # 項目数が多いパネルでは操作が煩わしいため)。ホバー中だけグローバルに
        # バインドし、離れたら解除することで、他ウィジェット上のホイール操作と
        # 競合しないようにする。
        def _on_mousewheel(event: tk.Event) -> None:
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

        canvas.bind("<Enter>", lambda e: canvas.bind_all("<MouseWheel>", _on_mousewheel))
        canvas.bind("<Leave>", lambda e: canvas.unbind_all("<MouseWheel>"))

        # 各セクションは「ラベル・入力欄」の組を横に2組並べる4カラムグリッド
        # (ラベル列, 入力欄列, ラベル列, 入力欄列)にして、項目数が多くても
        # 縦の長さを抑える。ラベル列の幅はgridが内容(最長ラベル)に応じて
        # 自動で揃えるため、width指定で日本語ラベルが見切れる問題を避けられる。
        # add_section呼び出しごとに新しいグリッド・スロット位置カウンタへ
        # リセットする。
        section_state: dict[str, object] = {"grid": None, "n": 0}

        def add_section(title: str) -> None:
            ttk.Label(params, text=title, font=("", 10, "bold")).pack(anchor="w", pady=(10, 2), padx=6)
            grid = ttk.Frame(params)
            grid.pack(fill=tk.X, padx=2)
            section_state["grid"] = grid
            section_state["n"] = 0

        def _next_slot() -> tuple[tk.Widget, int, int, int]:
            grid = section_state["grid"]
            n = section_state["n"]
            section_state["n"] = n + 1
            row, col = divmod(n, 2)
            return grid, row, col * 2, col * 2 + 1

        def add_float(key: str, label: str, default: float) -> None:
            self.defaults[key] = float(default)
            var = tk.DoubleVar(value=float(s.get(key, default)))
            self.vars[key] = var
            grid, row, label_col, entry_col = _next_slot()
            lbl = ttk.Label(grid, text=label, cursor="hand2", font=("", 8))
            lbl.grid(row=row, column=label_col, sticky="w", padx=(4, 3), pady=2)
            lbl.bind("<Double-Button-1>", lambda e, k=key: self._reset_param(k))
            ttk.Entry(grid, textvariable=var, width=8).grid(
                row=row, column=entry_col, sticky="w", padx=(0, 8), pady=2
            )

        def add_int(key: str, label: str, default: int) -> None:
            self.defaults[key] = int(default)
            var = tk.IntVar(value=int(s.get(key, default)))
            self.vars[key] = var
            grid, row, label_col, entry_col = _next_slot()
            lbl = ttk.Label(grid, text=label, cursor="hand2", font=("", 8))
            lbl.grid(row=row, column=label_col, sticky="w", padx=(4, 3), pady=2)
            lbl.bind("<Double-Button-1>", lambda e, k=key: self._reset_param(k))
            ttk.Entry(grid, textvariable=var, width=8).grid(
                row=row, column=entry_col, sticky="w", padx=(0, 8), pady=2
            )

        def add_bool(key: str, label: str, default: bool) -> None:
            self.defaults[key] = bool(default)
            var = tk.BooleanVar(value=bool(s.get(key, default)))
            self.vars[key] = var
            cb = ttk.Checkbutton(params, text=label, variable=var, cursor="hand2")
            cb.pack(anchor="w", padx=6, pady=1)
            # チェックボックス自体のダブルクリックは2回トグルされてしまうため、
            # ハンドラ側でデフォルト値へ明示的に上書きして確実にリセットする。
            cb.bind("<Double-Button-1>", lambda e, k=key: self._reset_param(k))

        reset_row = ttk.Frame(params)
        reset_row.pack(fill=tk.X, padx=6, pady=(6, 2))
        ttk.Button(reset_row, text="すべてデフォルトに戻す", command=self._reset_all_params).pack(
            side=tk.LEFT
        )
        ttk.Label(
            params,
            text="(各項目ラベルをダブルクリックでもその項目だけリセットできます)",
            font=("", 8),
            foreground="#666666",
            wraplength=300,
            justify="left",
        ).pack(anchor="w", padx=6, pady=(0, 4))

        add_section("処理解像度")
        add_int("max_long_side_px", "処理解像度上限(px)", 1600)

        add_section("前処理")
        add_int("bilateral_d", "バイラテラル直径", 5)
        add_float("bilateral_sigma_color", "バイラテラル色シグマ", 50.0)
        add_float("bilateral_sigma_space", "バイラテラル空間シグマ", 50.0)
        add_float("clahe_clip_limit", "CLAHEコントラスト上限", 2.0)

        add_section("XDoG(輪郭)")
        add_float("xdog_sigma", "sigma", 2.0)
        add_float("xdog_k", "k", 1.6)
        add_float("xdog_tau", "tau", 0.98)
        add_float("xdog_epsilon", "epsilon", -0.0001)
        add_float("xdog_phi", "phi", 200.0)
        add_float("xdog_threshold", "二値化しきい値(0-1)", 1.0)
        add_int("min_object_size_px", "最小成分サイズ(px)", 1)
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
        add_float("simplify_tolerance_mm", "単純化許容誤差(mm, 0=無効)", 0.1)
        add_float("pen_width_mm", "プレビュー線幅(mm)", 0.3)
        add_bool("show_travel", "ペンアップ移動線を表示", False)

        add_section("ペン制御(G-code)")
        self.defaults["pen_mode"] = "z"
        pen_mode_var = tk.StringVar(value=s.get("pen_mode", "z"))
        self.vars["pen_mode"] = pen_mode_var
        pen_mode_grid, pen_mode_row, pen_mode_label_col, pen_mode_entry_col = _next_slot()
        pen_mode_lbl = ttk.Label(pen_mode_grid, text="ペン制御方式", cursor="hand2", font=("", 8))
        pen_mode_lbl.grid(row=pen_mode_row, column=pen_mode_label_col, sticky="w", padx=(4, 3), pady=2)
        pen_mode_lbl.bind("<Double-Button-1>", lambda e: self._reset_param("pen_mode"))
        ttk.Combobox(
            pen_mode_grid, textvariable=pen_mode_var, values=["z", "servo", "gpio"], width=6, state="readonly"
        ).grid(row=pen_mode_row, column=pen_mode_entry_col, sticky="w", padx=(0, 8), pady=2)
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
        self._dnd_targets.extend([right, self.canvas_widget.get_tk_widget()])

        # ドラッグ&ドロップ受け付け(tkinterdnd2が利用可能な場合のみ)。
        # ファイル選択欄だけでなくプレビュー領域も受け皿にして、ドロップ位置に
        # 神経質にならずに使えるようにする。
        if _DND_AVAILABLE:
            for target in self._dnd_targets:
                target.drop_target_register(DND_FILES)  # type: ignore[attr-defined]
                target.dnd_bind("<<Drop>>", self._on_image_drop)  # type: ignore[attr-defined]

        bottom = ttk.Frame(self.root, padding=8)
        bottom.pack(side=tk.BOTTOM, fill=tk.X)
        self.generate_btn = ttk.Button(bottom, text="線画生成", command=self._on_generate)
        self.generate_btn.pack(side=tk.LEFT, padx=4)
        ttk.Button(bottom, text="SVG保存", command=self._on_save_svg).pack(side=tk.LEFT, padx=4)
        ttk.Button(bottom, text="G-code保存", command=self._on_save_gcode).pack(side=tk.LEFT, padx=4)
        ttk.Button(bottom, text="プロッターへ送信...", command=self._on_open_plotter_panel).pack(
            side=tk.LEFT, padx=4
        )
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

    def _on_image_drop(self, event: tk.Event) -> None:
        """ドラッグ&ドロップされたファイル群から画像を1つ選んで設定する。

        event.dataはTclリスト形式(パスにスペースを含む場合は`{...}`で囲まれる)
        なので、Tkの`splitlist`で正しく分割する。複数ファイルがドロップされた
        場合は先頭の画像ファイルを使う。
        """
        paths = self.root.tk.splitlist(event.data)
        image_path = next((p for p in paths if p.lower().endswith(_IMAGE_EXTENSIONS)), None)
        if image_path is None:
            messagebox.showerror("XYPDraw", "画像ファイル(jpg/jpeg/png/bmp/webp)をドロップしてください")
            return
        self.path_var.set(image_path)

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

    # ---- プロッター送信 ----
    def _on_open_plotter_panel(self) -> None:
        # job自体ではなくcallback経由で渡すことで、パネルを開いたままメイン側で
        # パラメータを変えて再生成しても、送信時には常に最新のjobが使われる。
        # 既存パネルがあれば(ウィンドウを閉じていても)再利用して表示するだけに
        # とどめ、接続・ゼロ点設定(座標系)を維持したまま呼び出せるようにする。
        if self._plotter_panel is not None and self._plotter_panel.winfo_exists():
            self._plotter_panel.show()
            return
        self._plotter_panel = PlotterPanel(self.root, get_job=lambda: self.result.job if self.result else None)


def main() -> None:
    root = TkinterDnD.Tk() if _DND_AVAILABLE else tk.Tk()
    XYPDrawApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
