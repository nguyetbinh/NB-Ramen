"""Stream subprocess logs and stop the whole experiment group on interruption."""

from contextlib import nullcontext
import os
from pathlib import Path
import signal
import subprocess


def stop_group(process):
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=15)
    except subprocess.TimeoutExpired:
        pass
    # The parent can exit before a descendant; clear surviving group members too.
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait()


def run_logged(argv, *, env=None, log=None, cwd=None):
    argv = [str(value) for value in argv]
    with (Path(log).open("w") if log is not None else nullcontext(None)) as handle:
        process = subprocess.Popen(argv, cwd=cwd, env=env, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, bufsize=1,
                                   start_new_session=True)
        try:
            for line in process.stdout:
                if handle is not None:
                    handle.write(line)
                    handle.flush()
                print(line, end="", flush=True)
            code = process.wait()
        except BaseException:
            stop_group(process)
            raise
        finally:
            process.stdout.close()
    if code:
        # Also stop descendants of a failed driver before its caller saves a ZIP.
        stop_group(process)
        raise subprocess.CalledProcessError(code, argv)
    return subprocess.CompletedProcess(argv, code)
