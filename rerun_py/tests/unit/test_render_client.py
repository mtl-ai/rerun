"""Unit tests for the render-to-video command line, which needs no viewer binary."""

from __future__ import annotations

from pathlib import Path

import pytest
from rerun._spawn import _render_cli_args


def test_only_required_args_by_default() -> None:
    """Unset options are omitted so the CLI's own defaults apply."""
    args = _render_cli_args(output="out.mp4", listen_addr="0.0.0.0:9876")
    assert args == ["--listen", "0.0.0.0:9876", "--output", "out.mp4"]


def test_size_accepts_tuple_or_string() -> None:
    """`(width, height)` is spelled `WxH` for the CLI; a string passes through."""
    from_tuple = _render_cli_args(output="o.mp4", listen_addr="a:1", size=(1280, 720))
    from_str = _render_cli_args(output="o.mp4", listen_addr="a:1", size="1280x720")
    assert from_tuple == from_str
    assert "1280x720" in from_tuple


def test_paths_are_stringified() -> None:
    """`Path` arguments reach the CLI as plain strings."""
    args = _render_cli_args(
        output=Path("/tmp/out.mp4"),
        listen_addr="a:1",
        blueprint=Path("/tmp/bp.rbl"),
        ffmpeg_path=Path("/usr/bin/ffmpeg"),
    )
    assert all(isinstance(a, str) for a in args)
    assert "/tmp/out.mp4" in args
    assert "/tmp/bp.rbl" in args


def test_all_options_are_forwarded() -> None:
    """Every optional parameter maps to its CLI flag."""
    args = _render_cli_args(
        output="o.mp4",
        listen_addr="0.0.0.0:1",
        fps=60,
        size=(800, 600),
        codec="h264",
        crf=18,
        timeline="frame",
        blueprint="bp.rbl",
        connect_timeout=5.0,
        ffmpeg_path="/usr/bin/ffmpeg",
    )
    for flag, value in [
        ("--fps", "60"),
        ("--size", "800x600"),
        ("--codec", "h264"),
        ("--crf", "18"),
        ("--timeline", "frame"),
        ("--blueprint", "bp.rbl"),
        ("--connect-timeout", "5.0"),
        ("--ffmpeg-path", "/usr/bin/ffmpeg"),
    ]:
        assert args[args.index(flag) + 1] == value


def test_extra_args_go_last() -> None:
    """Escape hatch for flags this wrapper doesn't model."""
    args = _render_cli_args(output="o.mp4", listen_addr="a:1", extra_args=["--renderer", "metal"])
    assert args[-2:] == ["--renderer", "metal"]


def test_render_requires_an_output() -> None:
    """`render=True` without a file to write is a mistake worth catching early."""
    from rerun.experimental import ViewerClient

    with pytest.raises(ValueError, match="output"):
        ViewerClient.spawn(render=True)


def test_output_without_render_is_rejected() -> None:
    """A viewer has nothing to write to `output`, so silently ignoring it would mislead."""
    from rerun.experimental import ViewerClient

    with pytest.raises(ValueError, match="render"):
        ViewerClient.spawn(output="out.mp4")
