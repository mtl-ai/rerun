"""See `python3 -m rerun_cli --help`."""

from __future__ import annotations

import os
import subprocess
import sys


def exe_suffix() -> str:
    if sys.platform.startswith("win"):
        return ".exe"
    return ""


def add_exe_suffix(path: str) -> str:
    if not path.endswith(exe_suffix()):
        return path + exe_suffix()
    return path


def rerun_binary_path(*, verbose: bool = False) -> str:
    """
    Path of the bundled `rerun` binary this shim delegates to.

    The path is returned whether or not it exists, so callers can produce their own
    error message. `RERUN_CLI_PATH` overrides the lookup.
    """
    if "RERUN_CLI_PATH" in os.environ:
        if verbose:
            print(f"Using overridden RERUN_CLI_PATH={os.environ['RERUN_CLI_PATH']}", file=sys.stderr)
        target_path = os.environ["RERUN_CLI_PATH"]
    elif sys.platform == "darwin":
        bundled = os.path.join(os.path.dirname(__file__), "Rerun.app", "Contents", "MacOS", "Rerun")
        bare = os.path.join(os.path.dirname(__file__), "rerun")
        target_path = bundled if os.path.exists(bundled) else bare
    else:
        target_path = os.path.join(os.path.dirname(__file__), "rerun")

    return add_exe_suffix(target_path)


def main() -> int:
    target_path = rerun_binary_path(verbose=True)

    if not os.path.exists(target_path):
        print(f"Error: Could not find rerun binary at {target_path}", file=sys.stderr)
        return 1

    try:
        return subprocess.call([target_path, *sys.argv[1:]])
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    main()
