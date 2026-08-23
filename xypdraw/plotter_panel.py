"""実機(GRBLプロッター)との接続・ジョグ操作・G-code送信を行う別ウィンドウ。

XYPWriter (xypwriter/gui.py の GrblControlApp) の機体制御パートを、
JPG線画パイプライン向けに移植したもの。この機体は $20(ソフトリミット)/
$21(ハードリミット)/$22(原点復帰) がすべて無効なため、GRBL自身は移動範囲の
逸脱を検知できない。そのため以下の安全設計をそのまま踏襲する:
- ジョブ送信前に必ず現在位置を作業原点(0,0,0)にゼロ設定する(G92)。
- 送信前にソフトウェア側でXY座標範囲をチェックし、範囲外は自動トリミング
  した上で確認ダイアログを挟む。
- ペン上下はRelativeZPenController(絶対Z座標を仮定しない相対移動)を使う。
- キャンセルは即座に停止せず、フィードホールド('!')で安全に一時停止する。
"""
from __future__ import annotations

import json
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk

from .bbox_clip import clip_job_to_bounds
from .gcode_export import build_gcode_lines, build_outline_check_gcode
from .grbl_sender import GrblConnection, GrblError, parse_status
from .pen_control import RelativeZPenController
from .types import PlotJob

# PyInstaller(onefile)実行時は__file__が実行のたびに変わる一時展開ディレクトリを
# 指すため、そこに設定を保存すると終了時に失われる。exe本体のディレクトリを使う。
if getattr(sys, "frozen", False):
    _BASE_DIR = Path(sys.executable).resolve().parent
else:
    _BASE_DIR = Path(__file__).resolve().parent.parent
_SETTINGS_PATH = _BASE_DIR / "xypdraw_plotter_settings.json"

STEP_OPTIONS = [0.01, 0.1, 1, 10, 100]
FEED_OPTIONS = [10, 50, 100, 500, 1000, 2000, 5000]
Z_FEED_OPTIONS = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1500, 2000, 3000, 4000, 5000]
XY_FEED_OPTIONS = [50, 100, 200, 300, 500, 800, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000]

# $1(Step Idle Delay)。この機体はENABLEピンがX/Y/Z共通配線のため、GRBL標準では
# 「Z軸のみ」励磁保持を切り替える手段がない。トルク保持ON/OFFは全軸に対して
# $1を切り替える形で実装する。
GRBL_IDLE_DELAY_HOLD = 255  # ON: 常時励磁(モーター温度上昇と引き換えにペン位置を保持)
GRBL_IDLE_DELAY_DEFAULT = 25  # OFF: 25ms後にアイドル解放(通常運用)

# XYPWriterと同じ実機を前提とするため、機体系パラメータの既定値は
# 実際に使われている調整済みの値をそのまま踏襲する。
DEFAULT_SETTINGS = {
    "port": "",
    "baud": 115200,
    "max_x": 140.0,
    "max_y": 200.0,
    "step_mm": 1.0,
    "jog_step_z": 1.0,
    "feed": 1000.0,
    "jog_feed_z": 200.0,
    "z_down_mm": 5.0,
    "z_feed": 5000.0,
    "final_lift_mm": 3.0,
    "draw_feed": 5000.0,
    "travel_feed": 5000.0,
}

_AXIS_INDEX = {"X": 0, "Y": 1, "Z": 2}


class PlotterPanel(tk.Toplevel):
    """プロッター接続・ジョグ・送信パネル(別ウィンドウ)。

    Args:
        master: 親ウィジェット(メインウィンドウ)。
        get_job: 呼び出し時点の最新PlotJobを返すcallback。jobそのものを
            固定で受け取るのではなくcallback経由にすることで、メイン側で
            パラメータを変えて再生成した最新のjobを常に送信対象にできる。
    """

    def __init__(self, master: tk.Widget, get_job):
        super().__init__(master)
        self.title("XYPDraw - プロッター操作パネル")
        self._get_job = get_job

        self.conn: GrblConnection | None = None
        self.last_wco: tuple[float, float, float] = (0.0, 0.0, 0.0)
        # 前回の送信でteardown時にZ軸をどれだけ退避させたか。次回送信の最初の
        # pen_downで、その分を追加で下げる補正に使う(RelativeZPenController参照)。
        self._last_final_lift_mm: float = 0.0
        self._send_thread: threading.Thread | None = None
        self._cancel_event: threading.Event | None = None
        self.settings = self._load_settings()

        self._build_ui()
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    # ---- 設定の保存/読込 ----
    def _load_settings(self) -> dict:
        settings = dict(DEFAULT_SETTINGS)
        if _SETTINGS_PATH.exists():
            try:
                settings.update(json.loads(_SETTINGS_PATH.read_text(encoding="utf-8")))
            except Exception:
                pass
        return settings

    def _save_settings(self) -> None:
        try:
            self.settings.update(
                {
                    "port": self.port_var.get(),
                    "baud": int(self.baud_var.get()),
                    "max_x": float(self.max_x_var.get()),
                    "max_y": float(self.max_y_var.get()),
                    "step_mm": float(self.step_var.get()),
                    "jog_step_z": float(self.jog_step_z_var.get()),
                    "feed": float(self.feed_var.get()),
                    "jog_feed_z": float(self.jog_feed_z_var.get()),
                    "z_down_mm": float(self.zdown_var.get()),
                    "z_feed": float(self.z_feed_var.get()),
                    "final_lift_mm": float(self.final_lift_var.get()),
                    "draw_feed": float(self.draw_feed_var.get()),
                    "travel_feed": float(self.travel_feed_var.get()),
                }
            )
            _SETTINGS_PATH.write_text(json.dumps(self.settings, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:
            pass  # 設定保存の失敗はウィンドウ終了を妨げない

    def _on_close(self) -> None:
        self._save_settings()
        if self.conn is not None:
            try:
                self.conn.close()
            except Exception:
                pass
        self.destroy()

    # ---- UI構築 ----
    def _build_ui(self) -> None:
        s = self.settings

        conn_frame = ttk.LabelFrame(self, text="接続")
        conn_frame.grid(row=0, column=0, columnspan=2, sticky="ew", padx=6, pady=4)
        self.port_var = tk.StringVar(value=s["port"])
        self.port_combo = ttk.Combobox(conn_frame, textvariable=self.port_var, width=12, values=self._list_ports())
        self.port_combo.grid(row=0, column=0, padx=4, pady=4)
        ttk.Button(conn_frame, text="ポート更新", command=self._refresh_ports).grid(row=0, column=1, padx=2)
        ttk.Label(conn_frame, text="baud").grid(row=0, column=2)
        self.baud_var = tk.StringVar(value=str(s["baud"]))
        ttk.Entry(conn_frame, textvariable=self.baud_var, width=8).grid(row=0, column=3, padx=4)
        self.connect_border = tk.Frame(conn_frame, bg="red")
        self.connect_border.grid(row=0, column=4, padx=4)
        self.connect_btn = ttk.Button(self.connect_border, text="接続", command=self._toggle_connect)
        self.connect_btn.pack(padx=2, pady=2)
        self.conn_status_var = tk.StringVar(value="未接続")
        ttk.Label(conn_frame, textvariable=self.conn_status_var).grid(row=0, column=5, padx=8)

        info_frame = ttk.LabelFrame(self, text="情報表示")
        info_frame.grid(row=1, column=0, columnspan=2, sticky="ew", padx=6, pady=4)
        self.state_var = tk.StringVar(value="状態: -")
        self.mpos_var = tk.StringVar(value="MPos(機械座標): -")
        self.wpos_var = tk.StringVar(value="WPos(作業座標): -")
        ttk.Label(info_frame, textvariable=self.state_var).grid(row=0, column=0, padx=6, sticky="w")
        ttk.Label(info_frame, textvariable=self.mpos_var).grid(row=1, column=0, padx=6, sticky="w")
        ttk.Label(info_frame, textvariable=self.wpos_var).grid(row=2, column=0, padx=6, sticky="w")
        ttk.Button(info_frame, text="更新", command=self._refresh_status).grid(row=0, column=1, rowspan=3, padx=8)

        zero_frame = ttk.LabelFrame(self, text="ゼロ点設定")
        zero_frame.grid(row=2, column=0, sticky="nsew", padx=6, pady=4)

        def _zero_button(text: str, axes: str, column: int) -> None:
            border = tk.Frame(zero_frame, bg="#87CEFA")
            border.grid(row=0, column=column, padx=4, pady=4)
            ttk.Button(border, text=text, command=lambda: self._on_zero(axes)).pack(padx=2, pady=2)

        _zero_button("X=0", "X", 0)
        _zero_button("Y=0", "Y", 1)
        _zero_button("Z=0", "Z", 2)
        _zero_button("XYZ=0", "XYZ", 3)

        range_frame = ttk.LabelFrame(self, text="可動範囲 (mm)")
        range_frame.grid(row=2, column=1, sticky="nsew", padx=6, pady=4)
        self.max_x_var = tk.StringVar(value=str(s["max_x"]))
        self.max_y_var = tk.StringVar(value=str(s["max_y"]))
        ttk.Label(range_frame, text="X最大").grid(row=0, column=0)
        ttk.Entry(range_frame, textvariable=self.max_x_var, width=8).grid(row=0, column=1)
        ttk.Label(range_frame, text="Y最大").grid(row=1, column=0)
        ttk.Entry(range_frame, textvariable=self.max_y_var, width=8).grid(row=1, column=1)

        movement_row = ttk.Frame(self)
        movement_row.grid(row=3, column=0, columnspan=2, sticky="w")

        speed_frame = ttk.LabelFrame(movement_row, text="ジョグ速度 (mm / mm/min)")
        speed_frame.pack(side="left", padx=6, pady=4)
        ttk.Label(speed_frame, text="ステップXY").grid(row=0, column=0)
        self.step_var = tk.StringVar(value=str(s["step_mm"]))
        ttk.Combobox(
            speed_frame, textvariable=self.step_var, values=STEP_OPTIONS, width=5, state="readonly"
        ).grid(row=0, column=1)
        ttk.Label(speed_frame, text="フィードXY").grid(row=1, column=0)
        self.feed_var = tk.StringVar(value=str(s["feed"]))
        ttk.Combobox(
            speed_frame, textvariable=self.feed_var, values=FEED_OPTIONS, width=6, state="readonly"
        ).grid(row=1, column=1)
        ttk.Separator(speed_frame, orient="horizontal").grid(row=2, column=0, columnspan=2, sticky="ew", pady=4)
        ttk.Label(speed_frame, text="ステップZ").grid(row=3, column=0)
        self.jog_step_z_var = tk.StringVar(value=str(s["jog_step_z"]))
        ttk.Combobox(
            speed_frame, textvariable=self.jog_step_z_var, values=STEP_OPTIONS, width=5, state="readonly"
        ).grid(row=3, column=1)
        ttk.Label(speed_frame, text="フィードZ").grid(row=4, column=0)
        self.jog_feed_z_var = tk.StringVar(value=str(s["jog_feed_z"]))
        ttk.Combobox(
            speed_frame, textvariable=self.jog_feed_z_var, values=FEED_OPTIONS, width=6, state="readonly"
        ).grid(row=4, column=1)

        nav_frame = ttk.LabelFrame(movement_row, text="ナビゲーション")
        nav_frame.pack(side="left", padx=6, pady=4)
        ttk.Button(nav_frame, text="Y+", command=lambda: self._on_jog("Y", 1)).grid(row=0, column=1, padx=4, pady=4)
        ttk.Button(nav_frame, text="X-", command=lambda: self._on_jog("X", -1)).grid(row=1, column=0, padx=4, pady=4)
        ttk.Button(nav_frame, text="原点(0,0)", command=self._on_return_to_origin).grid(
            row=1, column=1, padx=4, pady=4
        )
        ttk.Button(nav_frame, text="X+", command=lambda: self._on_jog("X", 1)).grid(row=1, column=2, padx=4, pady=4)
        ttk.Button(nav_frame, text="Y-", command=lambda: self._on_jog("Y", -1)).grid(row=2, column=1, padx=4, pady=4)
        ttk.Button(nav_frame, text="Z+", command=lambda: self._on_jog("Z", 1)).grid(row=0, column=3, padx=4, pady=4)
        ttk.Button(nav_frame, text="Z-", command=lambda: self._on_jog("Z", -1)).grid(row=2, column=3, padx=4, pady=4)

        safety_frame = ttk.LabelFrame(self, text="安全操作")
        safety_frame.grid(row=4, column=0, columnspan=2, sticky="ew", padx=6, pady=4)
        ttk.Button(safety_frame, text="フィードホールド(!)", command=self._on_feed_hold).grid(
            row=0, column=0, padx=4, pady=4
        )
        ttk.Button(safety_frame, text="再開(~)", command=self._on_resume).grid(row=0, column=1, padx=4, pady=4)
        ttk.Button(safety_frame, text="ソフトリセット", command=self._on_soft_reset).grid(
            row=0, column=2, padx=4, pady=4
        )
        ttk.Button(safety_frame, text="アラーム解除($X)", command=self._on_unlock).grid(
            row=0, column=3, padx=4, pady=4
        )
        ttk.Label(safety_frame, text="コマンド送信").grid(row=1, column=0, padx=4, pady=(0, 4), sticky="e")
        self.raw_cmd_var = tk.StringVar(value="")
        raw_cmd_entry = ttk.Entry(safety_frame, textvariable=self.raw_cmd_var, width=20)
        raw_cmd_entry.grid(row=1, column=1, columnspan=2, padx=4, pady=(0, 4), sticky="ew")
        raw_cmd_entry.bind("<Return>", lambda _e: self._on_send_raw_command())
        ttk.Button(safety_frame, text="送信", command=self._on_send_raw_command).grid(
            row=1, column=3, padx=4, pady=(0, 4)
        )
        self.torque_hold_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            safety_frame,
            text="トルク保持(全軸, $1=255)",
            variable=self.torque_hold_var,
            command=self._on_toggle_torque_hold,
        ).grid(row=1, column=4, padx=(12, 4), pady=(0, 4), sticky="w")

        send_frame = ttk.LabelFrame(self, text="G-code送信")
        send_frame.grid(row=5, column=0, columnspan=2, sticky="ew", padx=6, pady=4)
        ttk.Label(send_frame, text="Z下降量(mm)").grid(row=0, column=0)
        self.zdown_var = tk.StringVar(value=str(s["z_down_mm"]))
        ttk.Entry(send_frame, textvariable=self.zdown_var, width=6).grid(row=0, column=1)
        ttk.Label(send_frame, text="Zフィード").grid(row=0, column=2)
        self.z_feed_var = tk.StringVar(value=str(s["z_feed"]))
        ttk.Combobox(send_frame, textvariable=self.z_feed_var, values=Z_FEED_OPTIONS, width=6).grid(row=0, column=3)
        ttk.Label(send_frame, text="終了時退避(mm)").grid(row=0, column=4)
        self.final_lift_var = tk.StringVar(value=str(s["final_lift_mm"]))
        ttk.Entry(send_frame, textvariable=self.final_lift_var, width=6).grid(row=0, column=5)
        ttk.Label(send_frame, text="描画フィード").grid(row=1, column=0)
        self.draw_feed_var = tk.StringVar(value=str(s["draw_feed"]))
        ttk.Combobox(send_frame, textvariable=self.draw_feed_var, values=XY_FEED_OPTIONS, width=6).grid(
            row=1, column=1
        )
        ttk.Label(send_frame, text="移動フィード").grid(row=1, column=2)
        self.travel_feed_var = tk.StringVar(value=str(s["travel_feed"]))
        ttk.Combobox(send_frame, textvariable=self.travel_feed_var, values=XY_FEED_OPTIONS, width=6).grid(
            row=1, column=3
        )

        btn_row = ttk.Frame(send_frame)
        btn_row.grid(row=2, column=0, columnspan=6, pady=6)
        ttk.Button(btn_row, text="外周確認", command=self._on_outline_check).pack(side="left", padx=4)
        self.send_btn = ttk.Button(btn_row, text="送信", command=self._on_send)
        self.send_btn.pack(side="left", padx=4)
        self.cancel_btn = ttk.Button(btn_row, text="キャンセル", command=self._on_cancel_send, state="disabled")
        self.cancel_btn.pack(side="left", padx=4)

        self.job_status_var = tk.StringVar(value="先にプロッターへ接続してください。")
        ttk.Label(self, textvariable=self.job_status_var, wraplength=580, justify="left").grid(
            row=6, column=0, columnspan=2, sticky="w", padx=8, pady=(0, 8)
        )

    # ---- ポート一覧 ----
    def _list_ports(self) -> list[str]:
        import serial.tools.list_ports as list_ports

        return [p.device for p in list_ports.comports()]

    def _refresh_ports(self) -> None:
        self.port_combo["values"] = self._list_ports()

    # ---- 接続 ----
    def _toggle_connect(self) -> None:
        if self._warn_if_sending():
            return
        if self.conn is not None:
            try:
                self.conn.close()
            except Exception:
                pass
            self.conn = None
            self.conn_status_var.set("未接続")
            self.connect_btn.config(text="接続")
            self.connect_border.config(bg="red")
            return

        port = self.port_var.get().strip()
        if not port:
            messagebox.showerror("エラー", "COMポートを選択してください")
            return
        try:
            baud = int(self.baud_var.get())
            conn = GrblConnection(port, baudrate=baud)
            conn.connect()
            self.conn = conn
            self._last_final_lift_mm = 0.0  # 新規接続では物理的な基準が不明なのでリセット
            self.conn_status_var.set(f"接続済み: {port}")
            self.connect_btn.config(text="切断")
            self.connect_border.config(bg="green")
            self._refresh_status()
        except Exception as e:  # noqa: BLE001
            messagebox.showerror("接続エラー", str(e))

    def _require_conn(self) -> GrblConnection | None:
        if self.conn is None:
            messagebox.showerror("エラー", "先にプロッターへ接続してください")
            return None
        return self.conn

    def _warn_if_sending(self) -> bool:
        if self._cancel_event is not None:
            messagebox.showerror("エラー", "送信中は操作できません。先に「キャンセル」してください。")
            return True
        return False

    # ---- 情報表示 ----
    def _apply_status(self, parsed: dict) -> tuple[float, float, float] | None:
        if parsed.get("work_offset") is not None:
            self.last_wco = parsed["work_offset"]
        mpos = parsed.get("machine_position")
        if mpos is None:
            return None
        wpos = tuple(m - w for m, w in zip(mpos, self.last_wco))
        self.state_var.set(f"状態: {parsed.get('state') or '?'}")
        self.mpos_var.set("MPos(機械座標): X{:.3f} Y{:.3f} Z{:.3f}".format(*mpos))
        self.wpos_var.set("WPos(作業座標): X{:.3f} Y{:.3f} Z{:.3f}".format(*wpos))
        return wpos

    def _refresh_status(self) -> tuple[float, float, float] | None:
        conn = self._require_conn()
        if conn is None:
            return None
        if self._warn_if_sending():
            return None
        try:
            parsed = parse_status(conn.status())
        except Exception as e:  # noqa: BLE001
            messagebox.showerror("通信エラー", str(e))
            return None
        return self._apply_status(parsed)

    # ---- ゼロ点設定 ----
    def _on_zero(self, axes: str) -> None:
        conn = self._require_conn()
        if conn is None:
            return
        if self._warn_if_sending():
            return
        try:
            parsed = parse_status(conn.status())
            if parsed.get("work_offset") is not None:
                self.last_wco = parsed["work_offset"]
            mpos = parsed.get("machine_position")
            if mpos is None:
                messagebox.showerror("エラー", "現在位置を取得できませんでした")
                return

            conn.send_line("G92 " + " ".join(f"{a}0" for a in axes))

            if "Z" in axes:
                self._last_final_lift_mm = 0.0

            new_wco = list(self.last_wco)
            for a in axes:
                new_wco[_AXIS_INDEX[a]] = mpos[_AXIS_INDEX[a]]
            self.last_wco = tuple(new_wco)
            self._apply_status({"state": parsed.get("state"), "machine_position": mpos, "work_offset": None})
            self.job_status_var.set(f"{axes} をゼロ設定しました")
        except GrblError as e:
            messagebox.showerror("GRBLエラー", str(e))

    # ---- ナビゲーション ----
    def _on_jog(self, axis: str, sign: int) -> None:
        conn = self._require_conn()
        if conn is None:
            return
        if self._warn_if_sending():
            return
        try:
            if axis == "Z":
                step = float(self.jog_step_z_var.get())
                feed = float(self.jog_feed_z_var.get())
            else:
                step = float(self.step_var.get())
                feed = float(self.feed_var.get())
        except ValueError:
            messagebox.showerror("エラー", "ステップ/フィードの値を確認してください")
            return

        delta = step * sign
        try:
            conn.send_line("G91")
            conn.send_line(f"G1 {axis}{delta:.4f} F{feed:.1f}")
            conn.send_line("G90")
            if axis == "Z":
                self._last_final_lift_mm = 0.0
            self.job_status_var.set(f"{axis}を{delta:+.3f}mm移動しました")
        except GrblError as e:
            messagebox.showerror("GRBLエラー", str(e))
        finally:
            self._refresh_status()

    def _on_return_to_origin(self) -> None:
        conn = self._require_conn()
        if conn is None:
            return
        if self._warn_if_sending():
            return
        try:
            travel_feed = float(self.travel_feed_var.get())
        except ValueError:
            messagebox.showerror("エラー", "送り速度(移動)を確認してください")
            return
        if not messagebox.askyesno(
            "原点に戻る",
            "現在の作業原点(0,0)へXYだけを移動します(Zは動かしません)。\n続行しますか?",
        ):
            return
        try:
            conn.send_line("G90")
            conn.send_line(f"G0 X0.000 Y0.000 F{travel_feed:.1f}")
            self.job_status_var.set("原点(0,0)へ移動しました")
        except GrblError as e:
            messagebox.showerror("GRBLエラー", str(e))
        finally:
            self._refresh_status()

    # ---- 安全操作 ----
    def _on_feed_hold(self) -> None:
        conn = self._require_conn()
        if conn is None:
            return
        conn.feed_hold()
        self.job_status_var.set("フィードホールド(!)を送信しました")

    def _on_resume(self) -> None:
        conn = self._require_conn()
        if conn is None:
            return
        conn.cycle_resume()
        self.job_status_var.set("再開(~)を送信しました")

    def _on_soft_reset(self) -> None:
        conn = self._require_conn()
        if conn is None:
            return
        conn.soft_reset()
        self.job_status_var.set("ソフトリセットを送信しました")

    def _on_unlock(self) -> None:
        conn = self._require_conn()
        if conn is None:
            return
        if self._warn_if_sending():
            return
        try:
            conn.unlock()
            self.job_status_var.set("アラーム解除($X)を送信しました")
        except GrblError as e:
            messagebox.showerror("GRBLエラー", str(e))

    def _on_send_raw_command(self) -> None:
        conn = self._require_conn()
        if conn is None:
            return
        if self._warn_if_sending():
            return
        command = self.raw_cmd_var.get().strip()
        if not command:
            return
        try:
            resp = conn.send_line(command)
            self.job_status_var.set(f"'{command}' -> {resp}")
        except GrblError as e:
            messagebox.showerror("GRBLエラー", str(e))

    def _on_toggle_torque_hold(self) -> None:
        want_hold = self.torque_hold_var.get()
        conn = self._require_conn()
        if conn is None:
            self.torque_hold_var.set(not want_hold)
            return
        if self._warn_if_sending():
            self.torque_hold_var.set(not want_hold)
            return
        value = GRBL_IDLE_DELAY_HOLD if want_hold else GRBL_IDLE_DELAY_DEFAULT
        command = f"$1={value}"
        try:
            resp = conn.send_line(command)
            self._kick_stepper_idle_timer(conn)
            state_label = "トルク保持ON(常時励磁)" if want_hold else "トルク保持OFF(通常)"
            self.job_status_var.set(f"{state_label}: '{command}' -> {resp}")
        except GrblError as e:
            self.torque_hold_var.set(not want_hold)
            messagebox.showerror("GRBLエラー", str(e))

    def _kick_stepper_idle_timer(self, conn: GrblConnection) -> None:
        conn.send_line("G91")
        conn.send_line("G1 Z0.1 F60")
        conn.send_line("G1 Z-0.1 F60")
        conn.send_line("G90")

    def _auto_disable_torque_hold(self) -> None:
        if not self.torque_hold_var.get():
            return
        self.torque_hold_var.set(False)
        self._on_toggle_torque_hold()

    # ---- 送信対象job取得 ----
    def _require_job(self) -> PlotJob | None:
        job = self._get_job()
        if job is None:
            messagebox.showwarning("XYPDraw", "先にメイン画面で「線画生成」を実行してください。")
            return None
        return job

    # ---- 外周確認 ----
    def _on_outline_check(self) -> None:
        conn = self._require_conn()
        if conn is None:
            return
        if self._warn_if_sending():
            return
        job = self._require_job()
        if job is None:
            return
        canvas_w_mm, canvas_h_mm = job.canvas_size_mm

        if not messagebox.askyesno(
            "外周確認",
            f"描画範囲: X 0〜{canvas_w_mm:.1f}mm, Y 0〜-{canvas_h_mm:.1f}mm\n"
            "ペンは動かさず(Zコマンドなし)、低速(100mm/min)でXYだけを外周に沿って動かします。\n"
            "電源投入直後のキャリッジ位置が、この矩形の(0,0)角＝画像左上に対応している想定です。\n"
            "X+方向が右、Y-方向が下に動くか、必ず目視で確認してください。\n"
            "続行すると、現在位置を作業原点(0,0,0)にゼロ設定してから送信します。\n"
            "異常があればすぐ電源を切ってください。",
        ):
            return

        lines = build_outline_check_gcode(canvas_w_mm, canvas_h_mm, feed_rate=100.0)
        try:
            conn.zero_work_origin()
            parsed = parse_status(conn.status())
            if parsed.get("machine_position") is not None:
                self.last_wco = parsed["machine_position"]

            total = len(lines)
            for i, line in enumerate(lines):
                conn.send_line(line)
                self.job_status_var.set(f"外周確認送信中... {i + 1}/{total}")
                self.update_idletasks()
            self.job_status_var.set("外周確認 完了。X+が右、Y-が下に動いたか確認してください。")
        except GrblError as e:
            messagebox.showerror("GRBLエラー", str(e))
            self.job_status_var.set(f"エラーで中断: {e}")
        finally:
            self._refresh_status()

    # ---- 送信 ----
    def _on_send(self) -> None:
        conn = self._require_conn()
        if conn is None:
            return
        if self._warn_if_sending():
            return
        try:
            z_down = float(self.zdown_var.get())
            z_feed = float(self.z_feed_var.get())
            final_lift_mm = float(self.final_lift_var.get())
            max_x = float(self.max_x_var.get())
            max_y = float(self.max_y_var.get())
            draw_feed = float(self.draw_feed_var.get())
            travel_feed = float(self.travel_feed_var.get())
        except ValueError:
            messagebox.showerror("エラー", "数値項目を確認してください")
            return

        job = self._require_job()
        if job is None:
            return

        job, trimmed = clip_job_to_bounds(job, max_x=max_x, max_y=max_y)
        if trimmed and not messagebox.askyesno(
            "範囲外データのトリミング",
            f"描画データの一部が可動範囲(X 0〜{max_x:.1f}mm, Y 0〜-{max_y:.1f}mm)の外にあったため、"
            "自動的に切り取りました。\n\n"
            f"{job.stats.summary()}\n\n"
            "この内容のまま続行しますか？",
        ):
            return

        canvas_w_mm, canvas_h_mm = job.canvas_size_mm
        if not messagebox.askyesno(
            "送信確認",
            f"この線画を描画します。\n{job.stats.summary()}\n\n"
            f"描画範囲: X 0〜{canvas_w_mm:.1f}mm, Y 0〜-{canvas_h_mm:.1f}mm\n"
            f"送り速度: 描画{draw_feed:.0f}mm/min, 移動{travel_feed:.0f}mm/min, Z{z_feed:.0f}mm/min\n"
            f"終了時はペンを追加で{final_lift_mm:.1f}mm退避させて終わります\n"
            "原点(0,0)＝画像左上を基準に、X+(右)・Y-(下)方向へ描画します。\n"
            "実際にX+が右・Y-が下に動くか未確認の場合は、先に「外周確認」ボタンで低速動作を目視確認してください。\n\n"
            "電源投入位置から見て、キャリッジが描き始めたい位置にあり、"
            "この範囲(+X, -Y方向)に十分な余裕がありますか？\n"
            "続行すると、現在位置を作業原点(0,0,0)にゼロ設定してから送信します。",
        ):
            return

        pen = RelativeZPenController(
            down_travel_mm=z_down,
            z_feed=z_feed,
            final_lift_mm=final_lift_mm,
            initial_extra_down_mm=self._last_final_lift_mm,
        )
        lines = build_gcode_lines(job, pen=pen, feed_rate=draw_feed, travel_feed_rate=travel_feed)

        cancel_event = threading.Event()
        self._cancel_event = cancel_event
        self.send_btn.config(state="disabled")
        self.cancel_btn.config(state="normal")

        def worker() -> None:
            cancelled_at: int | None = None
            completed = False
            try:
                conn.zero_work_origin()
                parsed = parse_status(conn.status())
                if parsed.get("machine_position") is not None:
                    self.last_wco = parsed["machine_position"]

                total = len(lines)
                for i, line in enumerate(lines):
                    if cancel_event.is_set():
                        cancelled_at = i
                        break
                    conn.send_line(line)
                    if i % 10 == 0:
                        self.after(0, lambda i=i: self.job_status_var.set(f"送信中... {i + 1}/{total}"))

                if cancelled_at is not None:
                    msg = (
                        f"キャンセルしました({cancelled_at}/{total}行送信済み、フィードホールド中)。"
                        "「再開」で続行するか「ソフトリセット」で完全停止してください。"
                    )
                    self.after(0, lambda: self.job_status_var.set(msg))
                else:
                    self._last_final_lift_mm = final_lift_mm
                    self.after(0, lambda: self.job_status_var.set("送信完了"))
                    idle_wait = 0.0
                    while idle_wait < 10.0:
                        if parse_status(conn.status()).get("state") == "Idle":
                            break
                        idle_wait += 0.2
                    completed = True
            except GrblError as e:
                self.after(0, lambda: messagebox.showerror("GRBLエラー", str(e)))
                self.after(0, lambda: self.job_status_var.set(f"エラーで中断: {e}"))
            finally:
                self._cancel_event = None
                self._send_thread = None
                self.after(0, self._refresh_status)
                self.after(0, lambda: self.send_btn.config(state="normal"))
                self.after(0, lambda: self.cancel_btn.config(state="disabled"))
                if completed:
                    self.after(0, self._auto_disable_torque_hold)

        self._send_thread = threading.Thread(target=worker, daemon=True)
        self._send_thread.start()

    def _on_cancel_send(self) -> None:
        if self._cancel_event is None:
            return
        self._cancel_event.set()
        self.cancel_btn.config(state="disabled")
        if self.conn is not None:
            try:
                self.conn.feed_hold()
            except Exception:
                pass
        self.job_status_var.set("キャンセル要求を送信しました(フィードホールド中)...")
