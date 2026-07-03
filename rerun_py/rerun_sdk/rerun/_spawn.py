from __future__ import annotations

import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import subprocess
    from collections.abc import Sequence


def _render_cli_args(
    *,
    output: str | os.PathLike[str],
    listen_addr: str,
    fps: float | None = None,
    size: tuple[int, int] | str | None = None,
    codec: str | None = None,
    crf: int | None = None,
    timeline: str | None = None,
    blueprint: str | os.PathLike[str] | None = None,
    connect_timeout: float | None = None,
    ffmpeg_path: str | os.PathLike[str] | None = None,
    extra_args: Sequence[str] = (),
) -> list[str]:
    """
    Build the argument list for a headless render-to-video process, minus the executable.

    Kept separate from process spawning so it can be unit-tested without a viewer binary.
    Anything left as `None` is omitted, so the CLI's own defaults apply.
    """
    if isinstance(size, tuple):
        width, height = size
        size = f"{width}x{height}"

    args = ["--listen", listen_addr, "--output", os.fspath(output)]

    for flag, value in (
        ("--fps", fps),
        ("--size", size),
        ("--codec", codec),
        ("--crf", crf),
        ("--timeline", timeline),
        ("--blueprint", blueprint),
        ("--connect-timeout", connect_timeout),
        ("--ffmpeg-path", ffmpeg_path),
    ):
        if value is not None:
            args += [flag, os.fspath(value) if isinstance(value, os.PathLike) else str(value)]

    return args + list(extra_args)


def _wait_for_port(port: int, process: subprocess.Popen[bytes], timeout: float) -> None:
    """
    Block until `port` accepts connections, the process dies, or `timeout` elapses.

    Raises `RuntimeError` on death or timeout — the caller can't usefully proceed in
    either case, and a dead renderer usually means a bad flag or a missing `ffmpeg`.
    """
    import socket
    import time

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(
                f"The render process exited with code {process.returncode} before it started listening. "
                "Its output above should say why (a rejected argument, or a missing `ffmpeg`).",
            )
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.1):
                return
        except OSError:
            time.sleep(0.05)

    process.kill()
    raise RuntimeError(f"Timed out after {timeout}s waiting for the render process to listen on port {port}.")


def _free_port() -> int:
    """Ask the OS for an unused TCP port."""
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _spawn_render(
    *,
    output: str | os.PathLike[str],
    port: int | None = None,
    executable_path: str | os.PathLike[str] | None = None,
    subcommand: str | None = "render",
    startup_timeout: float = 30.0,
    **render_args: object,
) -> tuple[subprocess.Popen[bytes], int]:
    """
    Spawn a headless render-to-video process and block until it is listening.

    Returns the process and the port it listens on. See
    [`rerun.experimental.RenderClient.spawn`][] for the parameters.
    """
    import subprocess

    if port is None:
        port = _free_port()

    if executable_path is None:
        from rerun_cli.__main__ import rerun_binary_path

        executable_path = rerun_binary_path()
        if not os.path.exists(executable_path):
            raise FileNotFoundError(
                f"Could not find the rerun binary. Set `executable_path=` or `RERUN_CLI_PATH`.\nLooked at: {executable_path}",
            )

    argv = [os.fspath(executable_path)]
    if subcommand is not None:
        argv.append(subcommand)
    argv += _render_cli_args(
        output=output,
        listen_addr=f"0.0.0.0:{port}",
        **render_args,  # type: ignore[arg-type]
    )

    # We exec the viewer binary directly rather than going through the `rerun` python shim,
    # so this is the renderer itself — `wait()`/`terminate()` reach it without the process
    # group dance that `ViewerClient.close` needs. stdout/stderr stay inherited so ffmpeg
    # and viewer diagnostics are visible.
    process = subprocess.Popen(argv, stdin=subprocess.DEVNULL)

    _wait_for_port(port, process, startup_timeout)
    return process, port


def _spawn_viewer(
    *,
    port: int = 9876,
    memory_limit: str = "75%",
    server_memory_limit: str = "1GiB",
    hide_welcome_screen: bool = False,
    detach_process: bool = True,
    executable_name: str = "rerun",
    executable_path: str | None = None,
    headless: bool = False,
) -> int | None:
    """
    Internal helper to spawn a Rerun Viewer, listening on the given port.

    Blocks until the viewer is ready to accept connections. Returns the spawned
    viewer's pid, or `None` if spawning was skipped (e.g. when
    `_RERUN_TEST_FORCE_SAVE` is set).

    Used by [rerun.spawn][] and [rerun.experimental.ViewerClient][].

    Parameters
    ----------
    port:
        The port to listen on.
    memory_limit:
        An upper limit on how much memory the Rerun Viewer should use.
        When this limit is reached, Rerun will drop the oldest data.
        Example: `16GB` or `50%` (of system total).
    server_memory_limit:
        An upper limit on how much memory the gRPC server running
        in the same process as the Rerun Viewer should use.
        When this limit is reached, Rerun will drop the oldest data.
        Example: `16GB` or `50%` (of system total).

        Defaults to `1GiB`.
    hide_welcome_screen:
        Hide the normal Rerun welcome screen.
    detach_process:
        Detach Rerun Viewer process from the application process.
    executable_name:
        Specifies the name of the Rerun executable.
        You can omit the `.exe` suffix on Windows.

        Defaults to `rerun`.
    executable_path:
        Enforce a specific executable to use instead of searching
        through PATH for `executable_name`.

        Unspecified by default.
    headless:
        Run the spawned viewer in headless mode (no OS window).
        The viewer still listens for gRPC connections, so the SDK can keep
        logging data and request screenshots via
        [`rerun.experimental.ViewerClient.save_screenshot`][].

    """

    import rerun_bindings

    # NOTE: If `_RERUN_TEST_FORCE_SAVE` is set, all recording streams will write to disk no matter
    # what, thus spawning a viewer is pointless (and probably not intended).
    if os.environ.get("_RERUN_TEST_FORCE_SAVE") is not None:
        return None
    return rerun_bindings.spawn(
        port=port,
        memory_limit=memory_limit,
        server_memory_limit=server_memory_limit,
        hide_welcome_screen=hide_welcome_screen,
        detach_process=detach_process,
        executable_name=executable_name,
        executable_path=executable_path,
        # Let the spawned rerun process know it's just an app (skips analytics opt-in etc.).
        extra_env=[("RERUN_APP_ONLY", "true")],
        headless=headless,
    )
