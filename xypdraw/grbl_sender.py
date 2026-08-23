"""GRBL搭載の自作XYプロッターへG-codeをストリーミング送信する。

この自作機は $20(ソフトリミット)/$21(ハードリミット)/$22(原点復帰) が
すべて無効なため、GRBL自身は移動範囲の逸脱を検知できない。そのため
送信前にソフトウェア側で座標範囲をチェックする(check_xy_bounds)。

送信プロトコルは「1行送信 -> 'ok'/'error'応答待ち」という単純な同期方式。
GRBLの文字カウント方式ストリーミングより低速だが、実装が単純で確実。

XYPWriter (xypwriter/grbl_sender.py) から移植した実装。
"""
from __future__ import annotations

import re
import time

from .types import PlotJob


class GrblError(RuntimeError):
    """GRBLが 'error:' 応答を返した場合、または通信異常時に送出する。"""


_STATUS_RE = re.compile(
    r"<([^|>]+)\|MPos:(-?[\d.]+),(-?[\d.]+),(-?[\d.]+)"
    r"(?:.*?\|WCO:(-?[\d.]+),(-?[\d.]+),(-?[\d.]+))?"
)


def parse_status(status_line: str) -> dict:
    """GRBLの'?'応答を解析する。

    例: "<Idle|MPos:0.000,0.000,0.000|FS:0,0|WCO:-180.000,-240.000,0.000>"

    GRBLはWCO(作業座標オフセット)を毎回の応答には含めない(数回に1回のみ)。
    含まれない場合 work_offset は None になるので、呼び出し側で直前に
    取得した値を保持して補完すること。
    """
    m = _STATUS_RE.search(status_line)
    if not m:
        return {"state": None, "machine_position": None, "work_offset": None}
    state = m.group(1)
    mpos = tuple(float(x) for x in m.group(2, 3, 4))
    work_offset = None
    if m.group(5) is not None:
        work_offset = tuple(float(x) for x in m.group(5, 6, 7))
    return {"state": state, "machine_position": mpos, "work_offset": work_offset}


def check_xy_bounds(job: PlotJob, max_x: float, max_y: float) -> list[str]:
    """PlotJobのXY座標が機体の可動範囲[0, max_x] x [-max_y, 0]に収まっているか確認する。

    原点(0,0)は、実機でユーザーが「描き始めたいページ左上」にペンをジョグして
    作業原点としてゼロ点設定した位置に対応させることを前提とする。文書は
    そこからX+(右)・Y-(下)方向へ展開されるため、Yの許容範囲は0以下になる。
    """
    violations: list[str] = []
    for poly in job.polylines:
        if len(poly.points) == 0:
            continue
        xs = poly.points[:, 0]
        ys = poly.points[:, 1]
        if xs.min() < -0.01 or xs.max() > max_x + 0.01:
            violations.append(f"X座標が範囲外です: {xs.min():.2f}〜{xs.max():.2f}mm (上限{max_x}mm)")
        if ys.min() < -max_y - 0.01 or ys.max() > 0.01:
            violations.append(f"Y座標が範囲外です: {ys.min():.2f}〜{ys.max():.2f}mm (下限-{max_y}mm)")
    return violations


class GrblConnection:
    """pyserial経由でGRBLと通信する薄いラッパー。"""

    def __init__(
        self, port: str, baudrate: int = 115200, timeout: float = 2.0, response_timeout: float = 60.0
    ):
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.response_timeout = response_timeout  # send_line()が'ok'を待つ合計上限(秒)
        self._ser = None

    def connect(self) -> str:
        import serial  # 遅延importにして、シリアル送信を使わない用途ではpyserial必須にしない

        self._ser = serial.Serial(self.port, self.baudrate, timeout=self.timeout)
        # ArduinoベースのGRBLボードはシリアルポートを開くとリセットがかかり、
        # 起動メッセージを送るまで少し時間がかかる。
        time.sleep(2.0)
        self._ser.reset_input_buffer()
        self._ser.write(b"\r\n\r\n")
        time.sleep(2.0)
        startup = self._ser.read_all().decode(errors="replace")
        self._ser.reset_input_buffer()
        return startup.strip()

    def status(self) -> str:
        assert self._ser is not None
        self._ser.write(b"?")
        time.sleep(0.2)
        return self._ser.readline().decode(errors="replace").strip()

    def unlock(self) -> str:
        """$X でアラーム状態を解除する。"""
        return self.send_line("$X")

    def zero_work_origin(self) -> str:
        """現在の物理位置を作業座標系の原点(0,0,0)として宣言する。

        この機体は原点復帰が無効なため、過去の使用でG54等に残った
        作業座標オフセットが残っている可能性がある(実際に発生した事故の原因)。
        ジョブ送信前に必ずこれを呼び、G-codeが仮定する(0,0,0)と物理的な
        現在位置を一致させる。
        """
        return self.send_line("G92 X0 Y0 Z0")

    def send_line(self, line: str) -> str:
        """1行送信し、'ok'または'error:'応答が返るまでブロックする。

        GRBLの'ok'は「行を受理してプランナーバッファに積んだ」ことを意味し、
        バッファ(15ブロック程度)が一杯の間は、実行が進んで空きができるまで
        応答そのものを保留する。これは正常なフロー制御なので、1回の
        readline()タイムアウト(self.timeout)だけで即エラーにはせず、
        合計response_timeout秒に達するまでは無応答をリトライ扱いにする。
        """
        assert self._ser is not None, "connect()を先に呼んでください"
        payload = line.strip()
        if not payload:
            return ""
        self._ser.write((payload + "\n").encode())
        elapsed = 0.0
        while True:
            raw = self._ser.readline()
            if not raw:
                elapsed += self.timeout
                if elapsed >= self.response_timeout:
                    raise GrblError(
                        f"応答タイムアウト({self.response_timeout:.0f}秒): "
                        f"送信行 '{payload}' に対する応答がありません"
                    )
                continue
            resp = raw.decode(errors="replace").strip()
            if not resp:
                continue
            if resp.startswith("error:"):
                raise GrblError(f"'{payload}' -> {resp}")
            if resp.startswith("ALARM:"):
                raise GrblError(f"'{payload}' -> {resp} (アラーム状態。$Xで解除してから再開してください)")
            if resp == "ok":
                return resp
            # 起動メッセージや[MSG:...]等の情報行は無視して待ち続ける

    def feed_hold(self) -> None:
        """リアルタイムコマンド '!' で送り速度を保持(一時停止)する。"""
        assert self._ser is not None
        self._ser.write(b"!")

    def cycle_resume(self) -> None:
        """リアルタイムコマンド '~' でフィードホールドから再開する。"""
        assert self._ser is not None
        self._ser.write(b"~")

    def soft_reset(self) -> None:
        """リアルタイムコマンド Ctrl-X (0x18) でソフトリセットする。"""
        assert self._ser is not None
        self._ser.write(b"\x18")
        time.sleep(1.0)
        self._ser.reset_input_buffer()

    def stream(self, lines: list[str], on_progress=None) -> None:
        """複数行を順番に送信する。エラーが出た時点で例外を送出して停止する。"""
        for i, line in enumerate(lines):
            resp = self.send_line(line)
            if on_progress is not None:
                on_progress(i, len(lines), line, resp)

    def close(self) -> None:
        if self._ser is not None:
            self._ser.close()
            self._ser = None
