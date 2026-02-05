import asyncio
import contextlib
import importlib.util
import inspect
import io
import json
import os
import sys
import time
import traceback
from typing import Any, Dict, Optional, Tuple

from core.dependency_installer import extract_missing_module, install_for_missing_modules


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(BASE_DIR)
TOOLS_DIR = os.getenv("TOOLS_DIR", os.path.join(REPO_ROOT, "skills"))
_CONFIG_CACHE: Optional[Dict[str, Any]] = None
_INSTALL_ATTEMPTED = set()
_REGISTRY_CACHE: Optional[Dict[str, Tuple[str, str, Any]]] = None
_REGISTRY_TS: float = 0.0
_REGISTRY_TTL_SEC: float = 2.0


def _load_config() -> Dict[str, Any]:
    global _CONFIG_CACHE
    if _CONFIG_CACHE is not None:
        return _CONFIG_CACHE
    config_path = os.path.join(REPO_ROOT, "configs", "core_config.json")
    if os.path.exists(config_path):
        try:
            with open(config_path, "r", encoding="utf-8") as fh:
                _CONFIG_CACHE = json.load(fh)
                return _CONFIG_CACHE or {}
        except Exception:
            _CONFIG_CACHE = {}
            return {}
    _CONFIG_CACHE = {}
    return {}


def _auto_install_cfg() -> Dict[str, Any]:
    cfg = _load_config().get("dependency_auto_install", {}) or {}
    return {
        "enabled": bool(cfg.get("enabled", False)),
        "allowlist": set(cfg.get("allowlist") or []),
        "denylist": set(cfg.get("denylist") or []),
    }


def _attempt_install(module_name: str) -> bool:
    cfg = _auto_install_cfg()
    if not cfg.get("enabled"):
        return False
    if module_name in _INSTALL_ATTEMPTED:
        return False
    _INSTALL_ATTEMPTED.add(module_name)
    result = install_for_missing_modules([module_name], cfg["allowlist"], cfg["denylist"])
    return bool(result.get("success"))


def _load_module_from_path(module_name: str, file_path: str):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    if spec and spec.loader:
        try:
            spec.loader.exec_module(module)
        except Exception as exc:
            missing = extract_missing_module(exc)
            if missing and _attempt_install(missing):
                spec = importlib.util.spec_from_file_location(module_name, file_path)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
            else:
                raise
    return module


def _build_registry() -> Dict[str, Tuple[str, str, Any]]:
    registry: Dict[str, Tuple[str, str, Any]] = {}
    if TOOLS_DIR not in sys.path:
        sys.path.append(TOOLS_DIR)
    for filename in os.listdir(TOOLS_DIR):
        if not filename.endswith(".py") or filename.startswith("_"):
            continue
        module_name = filename[:-3]
        file_path = os.path.join(TOOLS_DIR, filename)
        try:
            module = _load_module_from_path(module_name, file_path)
        except Exception:
            continue
        tool_prefix = getattr(module, "__tool_prefix__", module_name) or module_name
        for name, func in inspect.getmembers(module, inspect.isfunction):
            if name.startswith("_") or func.__module__ != module_name:
                continue
            tool_name = f"{tool_prefix}.{name}"
            registry[tool_name] = (module_name, file_path, func)
    return registry


async def _get_registry() -> Dict[str, Tuple[str, str, Any]]:
    global _REGISTRY_CACHE, _REGISTRY_TS
    now = time.monotonic()
    if _REGISTRY_CACHE is not None and (now - _REGISTRY_TS) < _REGISTRY_TTL_SEC:
        return _REGISTRY_CACHE
    registry = await asyncio.to_thread(_build_registry)
    _REGISTRY_CACHE = registry
    _REGISTRY_TS = now
    return registry


async def _load_module_from_path_async(module_name: str, file_path: str):
    return await asyncio.to_thread(_load_module_from_path, module_name, file_path)


import functools

async def _run_tool(func, arguments: Dict[str, Any]) -> Any:
    if inspect.iscoroutinefunction(func):
        return await func(**arguments)
    
    loop = asyncio.get_running_loop()
    # Run sync tools in a thread to keep the event loop non-blocking
    result = await loop.run_in_executor(None, functools.partial(func, **arguments))
    
    if inspect.isawaitable(result):
        return await result
    return result


def _normalize_arguments(func, arguments: Dict[str, Any]) -> Dict[str, Any]:
    """
    Best-effort arg normalization to reduce "unexpected keyword" failures from LLM tool calls.

    We only remap when the target parameter exists in the function signature and the
    provided key does not.
    """
    if not isinstance(arguments, dict):
        return {}

    try:
        sig = inspect.signature(func)
    except Exception:
        return dict(arguments)

    params = sig.parameters
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()):
        # Tool accepts **kwargs; don't touch.
        return dict(arguments)

    out = {}
    # Common aliases the teacher LLM tends to emit.
    alias_sets = [
        # memory_ops.add_fact etc
        ("text", ["fact", "content", "message", "value"]),
        # shell.run
        ("command", ["cmd", "shell", "run", "powershell", "ps"]),
        # file/path tools
        ("path", ["file", "filepath", "file_path", "dir", "directory"]),
        ("query", ["q", "search", "term"]),
    ]

    reverse_alias = {}
    for target, aliases in alias_sets:
        for a in aliases:
            reverse_alias[a] = target

    for k, v in arguments.items():
        if k in params:
            out[k] = v
            continue
        mapped = reverse_alias.get(k)
        if mapped and mapped in params and mapped not in arguments:
            out[mapped] = v
            continue
        # Drop unknown args (prevents hard failure) unless the tool has a matching param.
        # If the tool requires it, it will error with a clearer "missing required argument".

    return out


def _parse_payload(raw: str) -> Dict[str, Any]:
    try:
        return json.loads(raw)
    except Exception:
        return {}


async def execute_tool(tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
    """Execute a tool locally and return the result."""
    if not tool_name:
        return {"ok": False, "error": "Tool name missing"}

    if not isinstance(arguments, dict):
        arguments = {}

    if TOOLS_DIR not in sys.path:
        sys.path.append(TOOLS_DIR)

    func = None
    registry = await _get_registry()
    entry = registry.get(tool_name)
    if entry:
        func = entry[2]

    if func is None:
        # Try finding it by module name if the tool name includes a dot
        if "." in tool_name:
            module_name, func_name = tool_name.rsplit(".", 1)
            file_path = os.path.join(TOOLS_DIR, f"{module_name}.py")
            if os.path.exists(file_path):
                try:
                    module = await _load_module_from_path_async(module_name, file_path)
                    func = getattr(module, func_name, None)
                except Exception:
                    pass

    if func is None:
        return {"ok": False, "error": f"Tool '{tool_name}' not found"}

    started = time.monotonic()
    try:
        arguments = _normalize_arguments(func, arguments)
        stdout_buf = io.StringIO()
        stderr_buf = io.StringIO()
        with contextlib.redirect_stdout(stdout_buf), contextlib.redirect_stderr(stderr_buf):
            result = await _run_tool(func, arguments)

        captured_stdout = stdout_buf.getvalue()
        captured_stderr = stderr_buf.getvalue()

        # If the tool returns a structured {success/ok/error} payload, propagate it to the gateway.
        if isinstance(result, dict):
            if result.get("ok") is False or result.get("success") is False:
                err = result.get("error") or result.get("message") or "Tool reported failure"
                return {
                    "ok": False,
                    "success": False,
                    "error": str(err),
                    "result": result,
                    "stdout": captured_stdout,
                    "stderr": captured_stderr,
                    "duration_ms": int((time.monotonic() - started) * 1000),
                }

        return {
            "ok": True,
            "result": result,
            "stdout": captured_stdout,
            "stderr": captured_stderr,
            "success": True,
            "duration_ms": int((time.monotonic() - started) * 1000),
        }
    except Exception as e:
        missing = extract_missing_module(e)
        if missing and _attempt_install(missing):
            try:
                stdout_buf = io.StringIO()
                stderr_buf = io.StringIO()
                with contextlib.redirect_stdout(stdout_buf), contextlib.redirect_stderr(stderr_buf):
                    result = await _run_tool(func, arguments)
                return {
                    "ok": True,
                    "result": result,
                    "stdout": stdout_buf.getvalue(),
                    "stderr": stderr_buf.getvalue(),
                    "success": True,
                    "duration_ms": int((time.monotonic() - started) * 1000),
                    "auto_installed": missing,
                }
            except Exception as e2:
                e = e2

        error_info = {
            "ok": False,
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc(),
            "duration_ms": int((time.monotonic() - started) * 1000),
        }
        # Extract code and context if it's a ToolError or similar
        if hasattr(e, "code"):
            error_info["code"] = e.code
        if hasattr(e, "context"):
            error_info["context"] = e.context
        return error_info


async def _execute(payload: Dict[str, Any]) -> Dict[str, Any]:
    tool_name = payload.get("name") or payload.get("tool_name")
    arguments = payload.get("arguments") or {}
    return await execute_tool(tool_name, arguments)


def main() -> None:
    raw = ""
    if len(sys.argv) > 1:
        raw = sys.argv[1]
    else:
        raw = sys.stdin.read()
    payload = _parse_payload(raw)
    try:
        result = asyncio.run(_execute(payload))
    except Exception as exc:
        result = {"ok": False, "error": str(exc)}
    print(json.dumps(result, ensure_ascii=True, default=str))


if __name__ == "__main__":
    main()
