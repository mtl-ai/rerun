from __future__ import annotations

import os
import warnings
from typing import TYPE_CHECKING

from ._viewer_client import ViewerClient

if TYPE_CHECKING:
    import subprocess
    from collections.abc import Sequence
    from types import TracebackType

    from rerun.recording_stream import RecordingStream


class RenderClient(ViewerClient):
    """
    A connection to a headless render-to-video process.

    Where [`ViewerClient`][rerun.experimental.ViewerClient] talks to a viewer that shows
    your data, this talks to a renderer that *encodes* it: every tick you log becomes one
    frame of an mp4, with no window and no intermediate `.rrd`.

    Log to it exactly as you would to a viewer — point a recording stream at
    [`url`][rerun.experimental.ViewerClient.url] and log — then call
    [`finish`][rerun.experimental.RenderClient.finish] (or leave the `with` block) to
    close the stream and wait for the encoder:

    ```python
    import rerun as rr
    from rerun.experimental import RenderClient

    rr.init("rerun_example_render")
    with RenderClient.spawn("out.mp4", fps=30) as render:
        rr.connect_grpc(render.url)
        for frame in range(100):
            rr.set_time("frame", sequence=frame)
            rr.log("points", rr.Points3D(positions))
            rr.flush()
    # out.mp4 is complete here
    ```

    Two rules the renderer imposes on the producer:

    - **Log on an integer (sequence) timeline.** One logged tick becomes one output frame,
      and tick N is encoded once data for a later tick proves N is complete.
    - **Flush once per tick.** The SDK's micro-batcher otherwise smears one tick's entities
      across a flush boundary, which shows up as partially-filled frames.

    !!! warning
        This API is experimental and may change or be removed in future versions.

    """

    def __init__(
        self,
        url: str,
        *,
        _process: subprocess.Popen[bytes],
        _output: str,
    ) -> None:
        """
        Low-level constructor.

        Prefer [`RenderClient.spawn`][rerun.experimental.RenderClient.spawn].

        Parameters
        ----------
        url:
            The `rerun+http://…/proxy` URL of the render process.
        _process:
            Internal — the render process, awaited by
            [`finish`][rerun.experimental.RenderClient.finish].
        _output:
            Internal — the video path the renderer writes.

        """
        # `_pid`/`_kill_on_exit` stay unset: killing a renderer truncates its mp4, so
        # teardown goes through `finish()` instead of the inherited `close()`.
        super().__init__(url)
        self._process = _process
        self._output = _output

    @classmethod
    def spawn(  # type: ignore[override]
        cls,
        output: str | os.PathLike[str],
        *,
        fps: float | None = None,
        size: tuple[int, int] | str | None = None,
        codec: str | None = None,
        crf: int | None = None,
        timeline: str | None = None,
        blueprint: str | os.PathLike[str] | None = None,
        connect_timeout: float | None = None,
        ffmpeg_path: str | os.PathLike[str] | None = None,
        port: int | None = None,
        executable_path: str | os.PathLike[str] | None = None,
        subcommand: str | None = "render",
        startup_timeout: float = 30.0,
        extra_args: Sequence[str] = (),
    ) -> RenderClient:
        """
        Spawn a headless renderer and connect to it.

        Blocks until the renderer is listening, so it is safe to point a recording stream
        at [`url`][rerun.experimental.ViewerClient.url] as soon as this returns.

        Requires `ffmpeg` on `PATH` (or `ffmpeg_path=`), and a working graphics stack —
        a real GPU/driver or a software rasterizer such as Mesa's `lavapipe`. In a bare CI
        container with no adapter the renderer exits at startup with "No graphics adapter
        was found".

        Parameters
        ----------
        output:
            Path of the `.mp4` to write. Only complete once
            [`finish`][rerun.experimental.RenderClient.finish] returns.
        fps:
            Frames per second stamped on the output. One logged tick is always one frame,
            regardless of wall-clock rate. Defaults to the CLI's 30.
        size:
            Rendered resolution, as `(width, height)` or `"1920x1080"`. Odd dimensions are
            cropped down a pixel, since yuv420p needs even sizes. Defaults to 1920x1080.
        codec:
            Video codec to encode with, e.g. `"h264"`. Defaults to the CLI's `h264`.
        crf:
            Constant rate factor, 0 to 51; lower is better quality and a bigger file.
        timeline:
            Timeline to render. Defaults to the one the viewer would pick.
        blueprint:
            A `.rbl` blueprint to apply before rendering — this is how you control the
            layout and which views appear in the video.
        connect_timeout:
            How long the renderer waits for this client to connect and log its first data
            before giving up, in seconds.
        ffmpeg_path:
            Explicit path to the `ffmpeg` binary. Defaults to looking it up on `PATH`.
        port:
            Port for the renderer to listen on. Defaults to an unused port chosen by the
            OS, so a render never collides with a viewer already on 9876.
        executable_path:
            Binary to run. Defaults to the `rerun` binary bundled with this SDK.
        subcommand:
            Subcommand that invokes the renderer. Pass `None` when `executable_path` points
            at a standalone render binary that takes the flags directly.
        startup_timeout:
            How long to wait, in seconds, for the renderer to start listening.
        extra_args:
            Extra command-line arguments passed through verbatim.

        """
        from rerun._spawn import _spawn_render

        process, resolved_port = _spawn_render(
            output=output,
            port=port,
            executable_path=executable_path,
            subcommand=subcommand,
            startup_timeout=startup_timeout,
            fps=fps,
            size=size,
            codec=codec,
            crf=crf,
            timeline=timeline,
            blueprint=blueprint,
            connect_timeout=connect_timeout,
            ffmpeg_path=ffmpeg_path,
            extra_args=extra_args,
        )

        return cls(
            f"rerun+http://127.0.0.1:{resolved_port}/proxy",
            _process=process,
            _output=os.fspath(output),
        )

    @property
    def output(self) -> str:
        """Path of the video file this renderer writes."""
        return self._output

    @property
    def returncode(self) -> int | None:
        """Exit code of the render process, or `None` while it is still running."""
        return self._process.poll()

    def finish(
        self,
        *,
        recording: RecordingStream | None = None,
        disconnect: bool = True,
        timeout: float | None = 300.0,
    ) -> int:
        """
        Close the stream, wait for the encoder, and return the renderer's exit code.

        The renderer ends its video when the last producer disconnects, so this
        disconnects first and then waits. **The mp4 is not playable until this returns** —
        the trailing metadata is written on exit.

        Safe to call more than once; later calls just re-report the exit code.

        Parameters
        ----------
        recording:
            Recording stream to flush and disconnect. Defaults to the current one.
        disconnect:
            Set to `False` if you have already disconnected the producer yourself, or if
            another process is the producer. Waiting will hang until that producer goes
            away.
        timeout:
            Seconds to wait for the renderer to exit before giving up and killing it,
            which leaves the video truncated. `None` waits forever.

        """
        if self._process.poll() is None and disconnect:
            from rerun.sinks import disconnect as _disconnect

            # Positional: `disconnect` takes the wrapper and converts it itself.
            _disconnect(recording)

        try:
            return self._process.wait(timeout=timeout)
        except Exception:
            self._process.kill()
            self._process.wait()
            warnings.warn(
                f"The render process did not exit within {timeout}s and was killed; "
                f"the video is likely truncated.\nFile path: {self._output}",
                UserWarning,
                stacklevel=2,
            )
            return self._process.returncode

    def close(self) -> None:
        """
        Finish the render, waiting for the encoder to finalize the video.

        Equivalent to [`finish`][rerun.experimental.RenderClient.finish] with its defaults.
        Named for symmetry with [`ViewerClient.close`][rerun.experimental.ViewerClient.close],
        but note the difference in kind: closing a viewer kills it, whereas closing a
        renderer waits for it, since killing one would truncate the video.
        """
        self.finish()

    def __enter__(self) -> RenderClient:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        if exc_type is not None:
            # The producer blew up mid-render, so the tail of the video is missing anyway.
            # Don't block the traceback waiting on a renderer that will never see its
            # remaining ticks.
            self._process.kill()
            self._process.wait()
            return
        self.finish()

    def __del__(self) -> None:
        # Deliberately not finishing here: waiting on an encoder during garbage collection
        # (or interpreter shutdown) would stall for as long as the encode takes. A dropped
        # renderer is a bug in the caller's lifecycle, so say so instead.
        try:
            if self._process.poll() is None:
                self._process.kill()
                warnings.warn(
                    "A RenderClient was garbage collected while its renderer was still running, "
                    "so the video is incomplete. Use a `with` block or call `finish()`.\n"
                    f"File path: {self._output}",
                    UserWarning,
                    stacklevel=2,
                )
        except Exception:
            pass
