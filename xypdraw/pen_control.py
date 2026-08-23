"""ペンアップ/ダウン制御のStrategy実装。

実プロッターの制御方式（Z軸上下/サーボ角度指定/GPIO直接制御）に応じて
複数実装を用意し、CLIの `--pen-mode` 切り替えだけで差し替えられるようにする。

XYPWriter (xypwriter/pen_control.py) から移植した実装。同じ実機を前提とする
場合はデフォルト値もそのまま踏襲する。
"""
from __future__ import annotations

from typing import Protocol


class PenController(Protocol):
    def setup(self) -> list[str]: ...
    def pen_up(self) -> list[str]: ...
    def pen_down(self) -> list[str]: ...
    def teardown(self) -> list[str]: ...


class ZAxisPenController:
    """一般的なG-code送信ソフト向け、Z軸でペンを上下させる方式。"""

    def __init__(self, up_z: float = 5.0, down_z: float = 0.0, feed_rate: float | None = None):
        self.up_z = up_z
        self.down_z = down_z
        self.feed_rate = feed_rate

    def _move_z(self, z: float) -> list[str]:
        cmd = f"G1 Z{z:.3f}"
        if self.feed_rate is not None:
            cmd += f" F{self.feed_rate:.1f}"
        return [cmd]

    def setup(self) -> list[str]:
        return ["G21", "G90"]  # mm単位, 絶対座標指定

    def pen_up(self) -> list[str]:
        return self._move_z(self.up_z)

    def pen_down(self) -> list[str]:
        return self._move_z(self.down_z)

    def teardown(self) -> list[str]:
        return self._move_z(self.up_z)


class ServoAnglePenController:
    """サーボ角度指定コマンド(M280等)でペンを上下させる方式。"""

    def __init__(self, up_angle: float = 90, down_angle: float = 0, servo_index: int = 0):
        self.up_angle = up_angle
        self.down_angle = down_angle
        self.servo_index = servo_index

    def setup(self) -> list[str]:
        return []

    def pen_up(self) -> list[str]:
        return [f"M280 P{self.servo_index} S{self.up_angle}"]

    def pen_down(self) -> list[str]:
        return [f"M280 P{self.servo_index} S{self.down_angle}"]

    def teardown(self) -> list[str]:
        return self.pen_up()


class GpioPenController:
    """GPIO直接制御(M42等)でペンを上下させる方式。"""

    def __init__(self, pin: int = 4, up_value: int = 0, down_value: int = 255):
        self.pin = pin
        self.up_value = up_value
        self.down_value = down_value

    def setup(self) -> list[str]:
        return []

    def pen_up(self) -> list[str]:
        return [f"M42 P{self.pin} S{self.up_value}"]

    def pen_down(self) -> list[str]:
        return [f"M42 P{self.pin} S{self.down_value}"]

    def teardown(self) -> list[str]:
        return self.pen_up()


class RelativeZPenController:
    """原点復帰(homing)が無効な機体向けに、相対移動でペンアップ/ダウンを行う。

    電源投入直後の安全な待機位置（ペンが紙から浮いている状態）をペンアップの
    基準とみなし、絶対Z座標は一切使わない。内部で「今アップかダウンか」を
    追跡し、既に同じ状態なら何もしない（同じ方向へ二重に動いて基準からずれる
    事故を防ぐ）。
    """

    def __init__(
        self,
        down_travel_mm: float = 3.0,
        z_feed: float = 200.0,
        final_lift_mm: float = 0.0,
        initial_extra_down_mm: float = 0.0,
    ):
        self.down_travel_mm = down_travel_mm
        self.z_feed = z_feed
        self.final_lift_mm = final_lift_mm  # 作業終了時、pen_upの位置からさらに退避させる追加上昇量(mm)
        self.initial_extra_down_mm = initial_extra_down_mm  # 最初のpen_downだけ追加で下げる補正量(mm)
        self._is_down = False
        self._first_down_done = False

    def setup(self) -> list[str]:
        self._is_down = False
        self._first_down_done = False
        return ["G21", "G90"]

    def pen_down(self) -> list[str]:
        if self._is_down:
            return []
        self._is_down = True
        travel = self.down_travel_mm
        if not self._first_down_done:
            travel += self.initial_extra_down_mm
            self._first_down_done = True
        return ["G91", f"G1 Z-{travel:.3f} F{self.z_feed:.1f}", "G90"]

    def pen_up(self) -> list[str]:
        if not self._is_down:
            return []
        self._is_down = False
        return ["G91", f"G1 Z{self.down_travel_mm:.3f} F{self.z_feed:.1f}", "G90"]

    def teardown(self) -> list[str]:
        """通常のペンアップに加えて、作業終了時はfinal_lift_mm分さらに退避する。"""
        lines = self.pen_up()
        if self.final_lift_mm > 0:
            lines = lines + ["G91", f"G1 Z{self.final_lift_mm:.3f} F{self.z_feed:.1f}", "G90"]
        return lines


class CustomCommandPenController:
    """任意のコマンド文字列を注入できる汎用プレースホルダー。"""

    def __init__(
        self,
        up_cmds: list[str],
        down_cmds: list[str],
        setup_cmds: list[str] | None = None,
        teardown_cmds: list[str] | None = None,
    ):
        self.up_cmds = up_cmds
        self.down_cmds = down_cmds
        self.setup_cmds = setup_cmds or []
        self.teardown_cmds = teardown_cmds or []

    def setup(self) -> list[str]:
        return list(self.setup_cmds)

    def pen_up(self) -> list[str]:
        return list(self.up_cmds)

    def pen_down(self) -> list[str]:
        return list(self.down_cmds)

    def teardown(self) -> list[str]:
        return list(self.teardown_cmds)
