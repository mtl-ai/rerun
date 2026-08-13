"""Integration tests for the headless renderer spawned via the Python SDK."""

from __future__ import annotations

import platform
import shutil
import struct
import sys
from typing import TYPE_CHECKING

import pytest
import rerun as rr
from rerun.experimental import RenderClient

if TYPE_CHECKING:
    from pathlib import Path

# Same graphics-stack constraint as the headless viewer (see `test_headless_viewer.py`),
# plus `ffmpeg`, which the renderer shells out to for encoding.
pytestmark = [
    pytest.mark.skipif(
        sys.platform == "linux" and platform.machine() == "aarch64",
        reason="no software rasterizer on linux-arm64 wheel-test runner",
    ),
    pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="the renderer encodes via the ffmpeg CLI"),
]


def _log_ticks(url: str, count: int) -> None:
    """Log `count` sequence ticks, one flush per tick, which is one output frame each."""
    rec = rr.RecordingStream("rerun_example_render_test")
    rec.connect_grpc(url=url)
    for frame in range(count):
        rec.set_time("frame", sequence=frame)
        rec.log("points", rr.Points3D([[frame, 0, 0], [0, frame, 1]], colors=[255, 0, 0]))
        rec.flush()
    rec.disconnect()


def _assert_is_mp4(path: Path) -> None:
    """Check the ISO-BMFF `ftyp` box, and that frames followed it."""
    data = path.read_bytes()
    assert len(data) > 0, "renderer produced an empty file"
    box_size, box_type = struct.unpack(">I4s", data[:8])
    assert box_type == b"ftyp", f"not an mp4: first box is {box_type!r}"
    assert len(data) > box_size, "mp4 contains only its header — no encoded frames"


def test_render_produces_video(tmp_path: Path) -> None:
    """Log ticks into a spawned renderer and get a playable mp4 back."""
    out = tmp_path / "out.mp4"

    with RenderClient.spawn(out, fps=10, size=(320, 240)) as render:
        _log_ticks(render.url, count=10)

    assert render.returncode == 0
    _assert_is_mp4(out)


def test_video_is_only_finalized_after_finish(tmp_path: Path) -> None:
    """The mp4's trailing metadata is written on exit, so `finish()` is what completes it."""
    out = tmp_path / "out.mp4"

    render = RenderClient.spawn(out, fps=10, size=(320, 240))
    _log_ticks(render.url, count=5)

    # The producer already disconnected, so don't ask finish() to do it again.
    assert render.finish(disconnect=False) == 0
    _assert_is_mp4(out)


def test_spawn_picks_a_free_port_by_default(tmp_path: Path) -> None:
    """Two concurrent renders must not fight over the SDK's default port."""
    first = RenderClient.spawn(tmp_path / "a.mp4", fps=10, size=(320, 240))
    second = RenderClient.spawn(tmp_path / "b.mp4", fps=10, size=(320, 240))
    try:
        assert first.url != second.url
    finally:
        for render in (first, second):
            _log_ticks(render.url, count=2)
            render.finish(disconnect=False)


def test_viewer_client_render_shortcut(tmp_path: Path) -> None:
    """`ViewerClient.spawn(render=True)` is the same renderer behind a familiar door."""
    from rerun.experimental import ViewerClient

    out = tmp_path / "out.mp4"
    with ViewerClient.spawn(render=True, output=out) as render:
        assert isinstance(render, RenderClient)
        _log_ticks(render.url, count=5)

    _assert_is_mp4(out)


def test_bad_arguments_fail_fast(tmp_path: Path) -> None:
    """A renderer that dies at startup is reported, not waited on until timeout."""
    with pytest.raises(RuntimeError, match="exited with code"):
        RenderClient.spawn(tmp_path / "out.mp4", extra_args=["--definitely-not-a-flag"])
