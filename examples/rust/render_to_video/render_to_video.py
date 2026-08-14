"""
Spawn the `render_to_video` renderer from a Python producer.

This is the Python half of the fork-free shape ("Option B" in
`PYTHON_TO_MP4_INTEGRATION.md`): your script logs Rerun archetypes as usual, and
each logged tick becomes one frame of an mp4 — no window, no intermediate `.rrd`.

It deliberately depends on **stock `rerun`** only, so it works against a
pip-installed SDK. That is the whole point of this branch: nothing inside the
Rerun tree is patched. (The in-tree branch instead ships
`rerun.experimental.RenderClient`, which needs a modified SDK.)

Usage::

    import rerun as rr
    from render_to_video import RenderToVideo

    rec = rr.RecordingStream("my_app")
    with RenderToVideo.spawn("out.mp4", fps=30, size=(1920, 1080)) as render:
        rec.connect_grpc(url=render.url)
        for frame in range(100):
            rec.set_time("frame", sequence=frame)
            rec.log("points", rr.Points3D(positions))
            rec.flush()  # `flush` is a stream method; there is no `rr.flush()`
        render.finish(rec)
    # out.mp4 is complete here

Two rules the renderer imposes on the producer (see `NOTES.md`):

- **Log on an integer (sequence) timeline.** Listen mode requires one, and says
  so if it does not find one.
- **Flush once per tick**, or the SDK's micro-batcher smears one tick's entities
  across a flush boundary.

Run this file directly for a self-contained demo::

    python render_to_video.py out.mp4
"""

from __future__ import annotations

import os
import shutil
import socket
import subprocess
import time
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Sequence
    from types import TracebackType
    from typing import Self

_BINARY_NAME = "render_to_video"


def _find_binary() -> str:
    """
    Locate the `render_to_video` binary.

    Checks `$RENDER_TO_VIDEO_BIN`, then this checkout's cargo target directory
    (release first — a debug renderer works but is much slower), then `PATH`.
    """
    from_env = os.environ.get("RENDER_TO_VIDEO_BIN")
    if from_env:
        return from_env

    # …/examples/rust/render_to_video/render_to_video.py → repository root
    repo_root = Path(__file__).resolve().parents[3]
    target_dir = Path(os.environ.get("CARGO_TARGET_DIR", repo_root / "target"))
    for profile in ("release", "debug"):
        candidate = target_dir / profile / _BINARY_NAME
        if candidate.exists():
            return str(candidate)

    on_path = shutil.which(_BINARY_NAME)
    if on_path:
        return on_path

    raise FileNotFoundError(
        f"Could not find the `{_BINARY_NAME}` binary. Build it with "
        f"`cargo build --release -p {_BINARY_NAME}`, or set `binary=` / `$RENDER_TO_VIDEO_BIN`.\n"
        f"Looked in: {target_dir}",
    )


def _free_port() -> int:
    """Ask the OS for an unused TCP port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


class RenderToVideo:
    """
    A spawned `render_to_video` process, and the URL to log into it.

    Unlike a viewer, a renderer must be **awaited** rather than killed: the mp4's
    trailing metadata is written when the process exits, so killing one truncates
    the file. Use a `with` block, or call [`finish`][].
    """

    def __init__(self, url: str, process: subprocess.Popen[bytes], output: str) -> None:
        self._url = url
        self._process = process
        self._output = output
        self._sentinel_entity: str | None = None

    @classmethod
    def spawn(
        cls,
        output: str | os.PathLike[str],
        *,
        fps: float | None = None,
        size: tuple[int, int] | str | None = None,
        codec: str | None = None,
        crf: int | None = None,
        timeline: str | None = None,
        blueprint: str | os.PathLike[str] | None = None,
        min_settle_steps: int | None = None,
        max_settle_steps: int | None = None,
        connect_timeout: float | None = None,
        quiet_timeout: float | None = None,
        sentinel_entity: str | None = None,
        ffmpeg_path: str | os.PathLike[str] | None = None,
        renderer: str | None = None,
        port: int | None = None,
        binary: str | os.PathLike[str] | None = None,
        startup_timeout: float = 60.0,
        extra_args: Sequence[str] = (),
    ) -> RenderToVideo:
        """
        Spawn the renderer and block until it is listening.

        Requires `ffmpeg` on `PATH` (or `ffmpeg_path=`) and a working graphics stack.
        Anything left as `None` keeps the renderer's own default.

        Parameters
        ----------
        output:
            Path of the `.mp4` to write. Only complete once [`finish`][] returns.
        fps:
            Frames per second stamped on the output. One logged tick is one frame
            regardless of wall-clock rate.
        size:
            Rendered resolution, as `(width, height)` or `"1920x1080"`.
        codec:
            `"h264"` (plays everywhere) or `"h265"` (smaller, pickier players).
        crf:
            Constant rate factor, 0 to 51; lower is better quality and a bigger file.
        timeline:
            Timeline to render. Must be a sequence timeline in listen mode.
        blueprint:
            A `.rbl` blueprint to apply. Worth setting for any output you care
            about: under the auto-generated blueprint the layout follows the entity
            set *as it currently is*, so a producer running ahead of the renderer can
            change the layout partway through the video (`NOTES.md`).
        min_settle_steps:
            Viewer frames to settle after each seek. The default of 2 suits raw
            archetypes; raise it for content that decodes asynchronously.
        max_settle_steps:
            Upper bound on settle steps per output frame.
        connect_timeout:
            Seconds to wait for this producer to connect and log its first data.
        quiet_timeout:
            Finish once the store has been unchanged this long (default 2s). This is
            the default end-of-stream rule, and costs that much wall-clock at the end
            of every run. It cannot tell a finished producer from a merely slow one.
        sentinel_entity:
            Entity path used as a "done" marker instead: rendering finishes as soon
            as it appears, which is prompt and deterministic. [`finish`][] logs it
            for you.

            Two caveats from `NOTES.md`: under the auto-generated blueprint the
            sentinel becomes visible content and can eat half the frame, so pair it
            with an explicit `blueprint`, or log it to a path the blueprint does not
            visualize. Its own tick also renders, adding one trailing frame.
        ffmpeg_path:
            Explicit path to the `ffmpeg` binary.
        renderer:
            Overrides the wgpu backend, e.g. `"metal"` or `"vulkan"`.
        port:
            Port to listen on. Defaults to an unused port chosen by the OS.
        binary:
            Path to the `render_to_video` binary. Defaults to `$RENDER_TO_VIDEO_BIN`,
            then this checkout's cargo target directory, then `PATH`.
        startup_timeout:
            Seconds to wait for the renderer to start listening.
        extra_args:
            Extra command-line arguments passed through verbatim.

        """
        if port is None:
            port = _free_port()
        if isinstance(size, tuple):
            size = f"{size[0]}x{size[1]}"

        argv = [
            str(binary) if binary is not None else _find_binary(),
            "--listen",
            f"0.0.0.0:{port}",
            "--output",
            os.fspath(output),
        ]
        for flag, value in (
            ("--fps", fps),
            ("--size", size),
            ("--codec", codec),
            ("--crf", crf),
            ("--timeline", timeline),
            ("--blueprint", blueprint),
            ("--min-settle-steps", min_settle_steps),
            ("--max-settle-steps", max_settle_steps),
            ("--connect-timeout", connect_timeout),
            ("--quiet-timeout", quiet_timeout),
            ("--sentinel-entity", sentinel_entity),
            ("--ffmpeg-path", ffmpeg_path),
            ("--renderer", renderer),
        ):
            if value is not None:
                argv += [flag, os.fspath(value) if isinstance(value, os.PathLike) else str(value)]
        argv += list(extra_args)

        process = subprocess.Popen(argv, stdin=subprocess.DEVNULL)

        deadline = time.monotonic() + startup_timeout
        while True:
            if process.poll() is not None:
                raise RuntimeError(
                    f"The renderer exited with code {process.returncode} before it started listening. "
                    "Its output above should say why (a rejected argument, or a missing `ffmpeg`).",
                )
            try:
                with socket.create_connection(("127.0.0.1", port), timeout=0.1):
                    break
            except OSError:
                if time.monotonic() > deadline:
                    process.kill()
                    raise RuntimeError(
                        f"Timed out after {startup_timeout}s waiting for the renderer to listen on port {port}.",
                    ) from None
                time.sleep(0.05)

        render = cls(f"rerun+http://127.0.0.1:{port}/proxy", process, os.fspath(output))
        render._sentinel_entity = sentinel_entity
        return render

    @property
    def url(self) -> str:
        """The `rerun+http://…/proxy` URL to point a recording stream at."""
        return self._url

    @property
    def output(self) -> str:
        """Path of the video file this renderer writes."""
        return self._output

    @property
    def returncode(self) -> int | None:
        """Exit code of the renderer, or `None` while it is still running."""
        return self._process.poll()

    def finish(
        self,
        recording: Any = None,
        *,
        disconnect: bool = True,
        timeout: float | None = 300.0,
    ) -> int:
        """
        End the stream, wait for the encoder, and return the renderer's exit code.

        **The mp4 is not playable until this returns.** If a `sentinel_entity` was
        given, it is logged here so the renderer stops immediately; otherwise the
        renderer stops on its own once the store has been quiet for its
        `quiet_timeout`.

        Safe to call more than once; later calls just re-report the exit code.

        Parameters
        ----------
        recording:
            `rerun.RecordingStream` to use. Defaults to the current one.
        disconnect:
            Set to `False` if you have already disconnected the producer yourself.
        timeout:
            Seconds to wait for the renderer to exit before giving up and killing it,
            which leaves the video truncated. `None` waits forever.

        """
        if self._process.poll() is None:
            import rerun as rr

            # `flush` is a RecordingStream method, not a module-level function, so the
            # sentinel path needs a concrete stream to flush even in the global case.
            stream = recording if recording is not None else rr.get_global_data_recording()

            if self._sentinel_entity is not None and stream is not None:
                stream.log(self._sentinel_entity, rr.TextLog("done"))
                stream.flush()

            if disconnect:
                if stream is not None:
                    stream.disconnect()
                else:
                    rr.disconnect()

        try:
            return self._process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            self._process.kill()
            self._process.wait()
            warnings.warn(
                f"The renderer did not exit within {timeout}s and was killed; "
                f"the video is likely truncated.\nFile path: {self._output}",
                UserWarning,
                stacklevel=2,
            )
            return self._process.returncode

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        if exc_type is not None:
            # The producer blew up mid-render, so the tail is missing anyway — don't
            # block the traceback waiting on ticks that will never arrive.
            self._process.kill()
            self._process.wait()
            return
        self.finish()


def _demo(output: str) -> None:
    """Log a rotating point cloud straight to `output`."""
    import numpy as np

    import rerun as rr

    rec = rr.RecordingStream("rerun_example_render_to_video")
    with RenderToVideo.spawn(output, fps=30, size=(1280, 720)) as render:
        rec.connect_grpc(url=render.url)
        for frame in range(90):
            rec.set_time("frame", sequence=frame)
            angle = frame * 0.07
            offsets = np.linspace(0.0, 6.28, 500)
            positions = np.stack(
                [np.cos(offsets + angle), np.sin(offsets + angle) * 0.5, offsets * 0.1],
                axis=1,
            )
            rec.log("cloud", rr.Points3D(positions, colors=[255, 140, 0], radii=0.02))
            rec.flush()
        render.finish(rec)
    print(f"wrote {output}")


if __name__ == "__main__":
    import sys

    _demo(sys.argv[1] if len(sys.argv) > 1 else "out.mp4")
