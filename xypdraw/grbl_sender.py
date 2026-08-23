"""GRBL搭載の自作XYプロッターへG-codeをストリーミング送信する。

この自作機は $20(ソフトリミット)/$21(ハードリミット)/$22(原点復帰) が
すべて無効なため、GRBL自身は移動範囲の逸脱を検知できない。そのため
送信前にソフトウェア側で座標範囲をチェックする(check_xy_bounds)。

複数行の送信(stream)はGRBL Wikiで説明されている文字カウント方式を使う。
GRBLの受信バッファを可能な限り先行して埋めておくことで、シリアル通信の
往復時間がボトルネックになって動きが止まる(実機上は「カクカクした
動き」として現れる)のを防ぐ。単発コマンド(send_line)は従来通り
1行送信->'ok'/'error'応答待ちの単純な同期方式。

XYPWriter (xypwriter/grbl_sender.py) から移植した実装(streamは文字
カウント方式に書き直している)。
"""
from __future__ import annotations

import re
import time
from typing import Callable

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

    def _read_response(self) -> str:
        """1件の応答が返るまでブロックする。'ok'ならそれを返し、'error:'/'ALARM:'ならGrblErrorを送出する。

        GRBLの'ok'は「行を受理してプランナーバッファに積んだ」ことを意味し、
        バッファ(15ブロック程度)が一杯の間は、実行が進んで空きができるまで
        応答そのものを保留する。これは正常なフロー制御なので、1回の
        readline()タイムアウト(self.timeout)だけで即エラーにはせず、
        合計response_timeout秒に達するまでは無応答をリトライ扱いにする。
        """
        assert self._ser is not None
        elapsed = 0.0
        while True:
            raw = self._ser.readline()
            if not raw:
                elapsed += self.timeout
                if elapsed >= self.response_timeout:
                    raise GrblError(f"応答タイムアウト({self.response_timeout:.0f}秒): 応答がありません")
                continue
            resp = raw.decode(errors="replace").strip()
            if not resp:
                continue
            if resp.startswith("error:"):
                raise GrblError(resp)
            if resp.startswith("ALARM:"):
                raise GrblError(f"{resp} (アラーム状態。$Xで解除してから再開してください)")
            if resp == "ok":
                return resp
            # 起動メッセージや[MSG:...]等の情報行は無視して待ち続ける

    def send_line(self, line: str) -> str:
        """1行送信し、'ok'または'error:'応答が返るまでブロックする。"""
        assert self._ser is not None, "connect()を先に呼んでください"
        payload = line.strip()
        if not payload:
            return ""
        self._ser.write((payload + "\n").encode())
        try:
            return self._read_response()
        except GrblError as e:
            raise GrblError(f"'{payload}' -> {e}") from e

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

    def stream(
        self,
        lines: list[str],
        on_progress: Callable[[int, int, str], None] | None = None,
        is_cancelled: Callable[[], bool] | None = None,
    ) -> int:
        """GRBL Wikiの"Streaming a G-Code Program"で説明されている文字カウント
        方式でストリーミング送信する。

        1行送信してから'ok'応答を待つ、を繰り返す同期方式(send_lineの単純な
        ループ)は、GRBL側のプランナーバッファ(15ブロック程度)を先行して
        埋められないため、シリアル通信の往復時間(GRBL側の処理時間・
        USB-シリアル変換の遅延等)がボトルネックになりやすい。往復のたびに
        バッファが枯渇して動きが止まり、次の応答到達で再開する、を繰り返すと
        実機上は「カクカクした動き」として現れる。

        この実装はGRBLの受信バッファ(既定128バイト)を可能な限り埋めた状態を
        保ちながら複数行を先行送信することでこれを回避する。送信済みだが
        まだ'ok'/'error'が返っていない行の文字数合計をpending_lensで追跡し、
        次の行を送るとバッファ容量を超える場合だけ応答を待つ。

        Args:
            lines: 送信するG-code行のリスト。
            on_progress: (完了行数, 総行数, 直前の応答)を受け取るcallback。
            is_cancelled: 呼ぶとTrue/Falseを返すcallback。Trueになったら
                新規行の送信を止め、送信済み(未確認)の行への応答だけ
                待ってから戻る(既に送った行の実行自体は止められないため、
                呼び出し側が別途feed_hold()等で安全に一時停止させること)。

        Returns:
            実際に送信した行数(キャンセルされた場合はlen(lines)未満)。
        """
        assert self._ser is not None, "connect()を先に呼んでください"
        rx_buffer_capacity = 100  # GRBL既定のRX_BUFFER_SIZE(128)に安全マージンを取った値
        pending_lens: list[int] = []  # 送信済み・未確認の各行の文字数(改行込み)
        next_idx = 0
        completed = 0
        total = len(lines)
        cancelled = False

        while next_idx < total or pending_lens:
            if not cancelled and is_cancelled is not None and is_cancelled():
                cancelled = True

            while (not cancelled) and next_idx < total:
                payload = lines[next_idx].strip()
                if not payload:
                    next_idx += 1
                    continue
                encoded_len = len(payload) + 1  # 改行分
                if pending_lens and sum(pending_lens) + encoded_len > rx_buffer_capacity:
                    break
                self._ser.write((payload + "\n").encode())
                pending_lens.append(encoded_len)
                next_idx += 1

            if not pending_lens:
                break  # キャンセル済みで、送信すべき行も残っていない

            resp = self._read_response()
            pending_lens.pop(0)
            completed += 1
            if on_progress is not None:
                on_progress(completed, total, resp)

        return completed

    def close(self) -> None:
        if self._ser is not None:
            self._ser.close()
            self._ser = None
