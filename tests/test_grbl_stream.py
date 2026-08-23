from __future__ import annotations

import pytest

from xypdraw.grbl_sender import GrblConnection, GrblError


class _FakeSerial:
    """GrblConnection.stream()を実機なしで検証するための疑似シリアルポート。

    write()された行をそのまま記録し、readline()が呼ばれるたびに'ok'
    (またはerror_atで指定した回数目だけ'error:1')を返す。
    """

    def __init__(self, error_at: int | None = None):
        self.written_lines: list[str] = []
        self.error_at = error_at
        self._resp_count = 0

    def write(self, data: bytes) -> int:
        text = data.decode().strip()
        if text:
            self.written_lines.append(text)
        return len(data)

    def readline(self) -> bytes:
        if self.error_at is not None and self._resp_count == self.error_at:
            self._resp_count += 1
            return b"error:1\n"
        self._resp_count += 1
        return b"ok\n"


def _connection(fake: _FakeSerial) -> GrblConnection:
    conn = GrblConnection(port="FAKE")
    conn._ser = fake  # connect()は実機(pyserial)依存のため、テストでは直接差し替える
    return conn


def test_stream_sends_all_lines_and_reports_progress():
    fake = _FakeSerial()
    conn = _connection(fake)
    lines = [f"G1 X{i}.000 Y0.000 F1500.0" for i in range(50)]

    progress_calls = []
    completed = conn.stream(lines, on_progress=lambda done, total, resp: progress_calls.append((done, total, resp)))

    assert completed == len(lines)
    assert fake.written_lines == lines
    assert progress_calls[-1] == (50, 50, "ok")
    assert [c[0] for c in progress_calls] == list(range(1, 51))


def test_stream_skips_empty_lines():
    fake = _FakeSerial()
    conn = _connection(fake)
    lines = ["G1 X1.000", "", "  ", "G1 X2.000"]

    completed = conn.stream(lines)

    assert fake.written_lines == ["G1 X1.000", "G1 X2.000"]
    assert completed == 2


def test_stream_raises_on_grbl_error():
    fake = _FakeSerial(error_at=3)
    conn = _connection(fake)
    lines = [f"G1 X{i}.000" for i in range(10)]

    with pytest.raises(GrblError):
        conn.stream(lines)


def test_stream_respects_rx_buffer_capacity():
    # 1行あたり約24バイト * 各行の未確認合計が100バイトを超えないよう
    # stream()が待ってから送るはずなので、一度に大量の行がwriteされきる
    # ことはない(readlineとwriteが交互に呼ばれる)ことを確認する。
    fake = _FakeSerial()
    conn = _connection(fake)
    lines = [f"G1 X{i:04d}.000 Y0.000 F1500" for i in range(200)]  # 各行約25文字

    completed = conn.stream(lines)

    assert completed == 200
    assert fake.written_lines == lines


def test_stream_cancellation_stops_new_sends():
    fake = _FakeSerial()
    conn = _connection(fake)
    lines = [f"G1 X{i}.000" for i in range(100)]

    call_count = {"n": 0}

    def is_cancelled() -> bool:
        call_count["n"] += 1
        return call_count["n"] > 2  # 数回のポーリング後にキャンセル

    completed = conn.stream(lines, is_cancelled=is_cancelled)

    assert completed < len(lines)
    assert fake.written_lines == lines[:completed]
