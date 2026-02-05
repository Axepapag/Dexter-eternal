import os
import subprocess
import shlex
from typing import Any, Dict
from error_util import ToolError, catch_errors

@catch_errors("SH")
def run(command: str, timeout: int = 30, use_shell: bool = True) -> Dict[str, Any]:
    """Execute a command using the system shell (cmd on Windows)."""
    if not command:
        raise ToolError("Missing command parameter", code="SH_MISSING_ARG")

    try:
        # On Windows, launching GUI apps via subprocess.run() will hang until the app exits.
        # For common GUI apps, spawn and return immediately (shell=False to avoid injection).
        if os.name == "nt":
            cmd = (command or "").strip()
            try:
                tokens = shlex.split(cmd, posix=False) if cmd else []
            except ValueError:
                tokens = []
            base = (tokens[0].strip('"').lower() if tokens else "")
            gui_apps = {
                "notepad", "calc", "mspaint", "taskmgr", "explorer",
                "msedge", "chrome", "firefox", "code",
            }
            if base in gui_apps and tokens:
                proc = subprocess.Popen(tokens, shell=False)
                return {
                    "stdout": "",
                    "stderr": "",
                    "returncode": 0,
                    "success": True,
                    "spawned": True,
                    "pid": proc.pid,
                }

        result = subprocess.run(
            command,
            shell=use_shell,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return {
            "stdout": result.stdout,
            "stderr": result.stderr,
            "returncode": result.returncode,
            "success": result.returncode == 0,
        }
    except subprocess.TimeoutExpired:
        raise ToolError(f"Command timed out after {timeout}s", code="SH_TIMEOUT")
