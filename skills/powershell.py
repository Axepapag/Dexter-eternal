import subprocess
import time
from typing import Any, Dict, Optional
from error_util import ToolError, catch_errors


def _normalize_process_name(process_name: str) -> str:
    name = process_name.strip().strip('"').strip("'")
    if name.lower().endswith(".exe"):
        name = name[:-4]
    return name


def _check_process(ps_exec: str, process_name: str) -> Dict[str, Any]:
    if not process_name:
        return {"found": False, "pids": []}
    base_name = _normalize_process_name(process_name)
    check_cmd = (
        f"Get-Process -Name \"{base_name}\" -ErrorAction SilentlyContinue | "
        "Select-Object -ExpandProperty Id"
    )
    full_check = f"{ps_exec} -NoProfile -Command \"{check_cmd}\""
    check = subprocess.run(full_check, shell=True, capture_output=True, text=True)
    pids = []
    for line in check.stdout.splitlines():
        line = line.strip()
        if line.isdigit():
            pids.append(int(line))
    return {"found": bool(pids), "pids": pids, "returncode": check.returncode}


@catch_errors("PS")
def execute(
    command: str,
    use_core: bool = False,
    timeout_seconds: Optional[float] = None,
    working_directory: Optional[str] = None,
    verify_process: str = "",
    verify_timeout_seconds: float = 0.0,
) -> Dict[str, Any]:
    """Execute a PowerShell command."""
    if not command:
        raise ToolError("Missing command parameter", code="PS_MISSING_ARG")

    ps_exec = "pwsh" if use_core else "powershell"
    
    # Sanitize command to replace bash-style chaining with PowerShell chaining
    sanitized_command = command.replace(" && ", " ; ").replace(" || ", " ; ")
    if " && " in command or " || " in command:
        # Note: PowerShell 7+ supports && and ||, but 5.1 (standard Windows PowerShell) does not.
        # Replacing with ; is a safe fallback for simple chaining.
        pass

    full_command = f"{ps_exec} -NoProfile -Command \"{sanitized_command}\""

    run_kwargs: Dict[str, Any] = {"shell": True, "capture_output": True, "text": True}
    if timeout_seconds is not None:
        run_kwargs["timeout"] = timeout_seconds
    if working_directory:
        run_kwargs["cwd"] = working_directory

    try:
        result = subprocess.run(full_command, **run_kwargs)
    except subprocess.TimeoutExpired:
        raise ToolError(f"Command timed out after {timeout_seconds}s", code="PS_TIMEOUT")

    response = {
        "stdout": result.stdout,
        "stderr": result.stderr,
        "returncode": result.returncode,
        "success": result.returncode == 0,
    }

    if verify_process:
        deadline = time.time() + max(0.0, verify_timeout_seconds or 0.0)
        process_check = None
        while True:
            process_check = _check_process(ps_exec, verify_process)
            if process_check.get("found") or time.time() >= deadline:
                break
            time.sleep(0.2)
        response["process_check"] = process_check

    return response
