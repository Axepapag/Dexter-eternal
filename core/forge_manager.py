#!/usr/bin/env python3
"""
Forge Manager: tool discovery, schema reconciliation, and targeted validation.
"""
from __future__ import annotations

import argparse
import asyncio
import importlib
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(BASE_DIR)
DEFAULT_SCHEMA_PATH = os.path.join(BASE_DIR, "tool_schemas.json")
DEFAULT_REPORT_PATH = os.path.join(REPO_ROOT, "artifacts", "forge_schema_report.json")
DEFAULT_VALIDATE_REPORT_PATH = os.path.join(REPO_ROOT, "artifacts", "forge_validation_report.json")
DEFAULT_TEST_INPUTS_PATH = os.path.join(BASE_DIR, "tool_test_inputs.json")
DEFAULT_LOOP_LOG_PATH = os.path.join(REPO_ROOT, "artifacts", "forge_manager_loop.log")


ALIAS_RULES = [
    {
        "id": "file_system_ops_to_file_ops",
        "from_prefix": "file_system_ops",
        "to_prefix": "file_ops",
        "rename": {},
    },
    {
        "id": "file_system_to_file_ops",
        "from_prefix": "file_system",
        "to_prefix": "file_ops",
        "rename": {},
    },
    {
        "id": "powershell_ops_to_powershell",
        "from_prefix": "powershell_ops",
        "to_prefix": "powershell",
        "rename": {"powershell_execute": "execute"},
    },
    {
        "id": "gui_automation_to_ui_control",
        "from_prefix": "gui_automation",
        "to_prefix": "ui_control",
        "rename": {
            "keyboard_hotkey": "hotkey",
            "keyboard_press": "press",
            "keyboard_type": "type_text",
            "mouse_click": "click",
            "mouse_move": "move",
        },
    },
    {
        "id": "ui_automation_to_ui_control",
        "from_prefix": "ui_automation",
        "to_prefix": "ui_control",
        "rename": {
            "click_at": "click",
            "press_key": "press",
        },
    },
    {
        "id": "web_automation_to_browser_workflows",
        "from_prefix": "web_automation",
        "to_prefix": "browser_workflows",
        "rename": {
            "google_search": "browser_search",
            "open_browser_url": "browser_open_url",
        },
    },
]


UNSAFE_NAME_TOKENS = [
    "delete",
    "remove",
    "kill",
    "shutdown",
    "reboot",
    "format",
    "wipe",
    "send",
    "email",
    "login",
    "open_app",
    "click",
    "press",
    "type",
    "hotkey",
    "drag",
    "scroll",
    "move",
    "write",
    "append",
    "save",
    "upload",
    "download",
    "transaction",
    "transfer",
]


def _load_json(path: str, default: Any) -> Any:
    if not path or not os.path.exists(path):
        return default
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _write_json(path: str, payload: Any) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=True)


def _append_log(path: Optional[str], message: str) -> None:
    if not path:
        return
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(f"[{timestamp}] {message}\n")


def _schema_name(entry: Dict[str, Any]) -> str:
    if not isinstance(entry, dict):
        return ""
    func = entry.get("function")
    if isinstance(func, dict):
        return func.get("name", "") or ""
    return entry.get("name", "") or ""


def _set_schema_name(entry: Dict[str, Any], name: str) -> None:
    func = entry.get("function")
    if isinstance(func, dict):
        func["name"] = name
    else:
        entry["name"] = name


def _build_registry(tools_dir: Optional[str]) -> Dict[str, Tuple[str, str, Any]]:
    if tools_dir:
        os.environ["TOOLS_DIR"] = tools_dir
    if BASE_DIR not in sys.path:
        sys.path.append(BASE_DIR)
    tool_executor = importlib.import_module("tool_executor")
    if tools_dir:
        tool_executor.TOOLS_DIR = tools_dir
        if tools_dir not in sys.path:
            sys.path.append(tools_dir)
    return tool_executor._build_registry()


def _build_schema_entry(tool_name: str, func: Any) -> Dict[str, Any]:
    if BASE_DIR not in sys.path:
        sys.path.append(BASE_DIR)
    toolbook_utils = importlib.import_module("toolbook_utils")
    return toolbook_utils.build_tool_schema(tool_name, func)


def _suggest_alias(name: str, registry_names: set) -> Optional[Dict[str, str]]:
    if "." not in name:
        return None
    prefix, func_name = name.split(".", 1)
    for rule in ALIAS_RULES:
        if prefix != rule["from_prefix"]:
            continue
        new_func = rule.get("rename", {}).get(func_name, func_name)
        target = f"{rule['to_prefix']}.{new_func}"
        if target in registry_names:
            return {"from": name, "to": target, "rule": rule["id"]}
    return None


def _is_unsafe_tool(tool_name: str) -> bool:
    lowered = tool_name.lower()
    return any(token in lowered for token in UNSAFE_NAME_TOKENS)


def _default_for_param(param_name: str, schema: Dict[str, Any], repo_root: str) -> Optional[Any]:
    param_type = (schema.get("type") or "").lower()
    name = param_name.lower()
    if name in ("path", "file_path"):
        return os.path.join(repo_root, "tools", "tool_schemas.json")
    if name in ("directory", "dir", "folder"):
        return repo_root
    if name in ("url", "link"):
        return "https://example.com"
    if name in ("command", "cmd"):
        return "echo hello"
    if name in ("query", "text", "pattern", "service", "username"):
        return "test"
    if name in ("email", "recipient", "sender"):
        return "test@example.com"
    if param_type == "integer":
        return 1
    if param_type == "number":
        return 1.0
    if param_type == "boolean":
        return False
    if param_type == "array":
        return []
    if param_type == "object":
        return {}
    return "test"


def _build_test_cases(
    tool_name: str,
    schema_entry: Optional[Dict[str, Any]],
    test_inputs: Dict[str, Any],
    repo_root: str,
) -> Tuple[List[Dict[str, Any]], str]:
    def expand_placeholders(value: Any) -> Any:
        if isinstance(value, str):
            return value.replace("{repo_root}", repo_root).replace("${REPO_ROOT}", repo_root)
        if isinstance(value, list):
            return [expand_placeholders(item) for item in value]
        if isinstance(value, dict):
            return {key: expand_placeholders(item) for key, item in value.items()}
        return value

    if tool_name in test_inputs:
        custom = test_inputs[tool_name]
        if isinstance(custom, list):
            cases = [case for case in custom if isinstance(case, dict)]
            return [expand_placeholders(case) for case in cases], "custom"
        if isinstance(custom, dict):
            return [expand_placeholders(custom)], "custom"
        return [], "invalid_custom"

    if not schema_entry:
        return [], "missing_schema"

    func = schema_entry.get("function", {})
    params = func.get("parameters", {}) if isinstance(func, dict) else {}
    required = params.get("required", []) or []
    properties = params.get("properties", {}) or {}

    if not required:
        return [{}], "no_required_params"

    args: Dict[str, Any] = {}
    for param in required:
        spec = properties.get(param, {}) if isinstance(properties, dict) else {}
        default = _default_for_param(param, spec, repo_root)
        if default is None:
            return [], "missing_required_defaults"
        args[param] = default
    return [args], "heuristic_defaults"


def _discover_tools(tools_dir: Optional[str], output_path: Optional[str]) -> Dict[str, Any]:
    registry = _build_registry(tools_dir)
    tools = []
    for name, (module_name, file_path, _) in sorted(registry.items()):
        tools.append({"name": name, "module": module_name, "file": file_path})
    report = {
        "tools_dir": tools_dir or BASE_DIR,
        "count": len(tools),
        "tools": tools,
    }
    if output_path:
        _write_json(output_path, report)
    return report


def _reconcile_schema(
    schema_path: str,
    tools_dir: Optional[str],
    apply_aliases: bool,
    add_missing: bool,
    prune_missing: bool,
    output_path: Optional[str],
    write_schema: bool,
) -> Dict[str, Any]:
    registry = _build_registry(tools_dir)
    registry_names = set(registry.keys())

    raw_schema = _load_json(schema_path, [])
    schema_entries = [entry for entry in raw_schema if isinstance(entry, dict)]

    schema_by_name: Dict[str, Dict[str, Any]] = {}
    duplicate_names: List[str] = []
    for entry in schema_entries:
        name = _schema_name(entry)
        if not name:
            continue
        if name in schema_by_name:
            duplicate_names.append(name)
        schema_by_name[name] = entry

    schema_names = set(schema_by_name.keys())
    missing_in_registry = sorted(schema_names - registry_names)
    missing_in_schema = sorted(registry_names - schema_names)

    alias_suggestions: List[Dict[str, str]] = []
    for name in missing_in_registry:
        suggestion = _suggest_alias(name, registry_names)
        if suggestion:
            alias_suggestions.append(suggestion)

    renamed: List[Dict[str, str]] = []
    pruned_duplicates: List[str] = []
    if apply_aliases and alias_suggestions:
        for suggestion in alias_suggestions:
            old_name = suggestion["from"]
            new_name = suggestion["to"]
            entry = schema_by_name.get(old_name)
            if not entry:
                continue
            if new_name in schema_by_name and schema_by_name[new_name] is not entry:
                pruned_duplicates.append(old_name)
                continue
            _set_schema_name(entry, new_name)
            rebuilt = False
            rebuild_error = None
            if new_name in registry:
                try:
                    _, _, func = registry[new_name]
                    schema_entry = _build_schema_entry(new_name, func)
                    entry.clear()
                    entry.update(schema_entry)
                    rebuilt = True
                except Exception as exc:
                    rebuild_error = str(exc)
            rename_entry = {"from": old_name, "to": new_name, "rule": suggestion["rule"], "rebuilt": rebuilt}
            if rebuild_error:
                rename_entry["rebuild_error"] = rebuild_error
            renamed.append(rename_entry)
            schema_by_name.pop(old_name, None)
            schema_by_name[new_name] = entry

        if pruned_duplicates:
            schema_entries = [entry for entry in schema_entries if _schema_name(entry) not in pruned_duplicates]

    if prune_missing:
        schema_entries = [entry for entry in schema_entries if _schema_name(entry) in registry_names]

    added: List[str] = []
    if add_missing:
        for name in sorted(registry_names):
            if name in {_schema_name(entry) for entry in schema_entries}:
                continue
            module_name, file_path, func = registry[name]
            try:
                schema_entry = _build_schema_entry(name, func)
                schema_entries.append(schema_entry)
                added.append(name)
            except Exception:
                continue

    if write_schema:
        _write_json(schema_path, schema_entries)

    affected = set()
    affected.update([item["to"] for item in renamed])
    affected.update(added)

    report = {
        "schema_path": schema_path,
        "tools_dir": tools_dir or BASE_DIR,
        "registry_count": len(registry_names),
        "schema_count": len(schema_entries),
        "duplicate_schema_names": sorted(set(duplicate_names)),
        "missing_in_registry": missing_in_registry,
        "missing_in_schema": missing_in_schema,
        "alias_suggestions": alias_suggestions,
        "renamed": renamed,
        "pruned_duplicates": pruned_duplicates,
        "added": added,
        "pruned_missing": prune_missing,
        "applied_aliases": apply_aliases,
        "affected_tools": sorted(affected),
    }

    if output_path:
        _write_json(output_path, report)
    return report


async def _run_tool_case(tool_name: str, args: Dict[str, Any], timeout: float) -> Dict[str, Any]:
    if BASE_DIR not in sys.path:
        sys.path.append(BASE_DIR)
    tool_executor = importlib.import_module("tool_executor")
    try:
        result = await asyncio.wait_for(tool_executor.execute_tool(tool_name, args), timeout=timeout)
        return {"ok": result.get("ok", False), "result": result}
    except asyncio.TimeoutError:
        return {"ok": False, "error": f"Timeout after {timeout}s"}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


async def _validate_tools(
    tool_names: List[str],
    schema_path: str,
    tools_dir: Optional[str],
    test_inputs_path: Optional[str],
    timeout: float,
    max_concurrency: int,
    unsafe_ok: bool,
    output_path: Optional[str],
) -> Dict[str, Any]:
    registry = _build_registry(tools_dir)
    registry_names = set(registry.keys())

    raw_schema = _load_json(schema_path, [])
    schema_by_name: Dict[str, Dict[str, Any]] = {}
    for entry in raw_schema:
        name = _schema_name(entry)
        if name:
            schema_by_name[name] = entry

    test_inputs = _load_json(test_inputs_path, {}) if test_inputs_path else {}
    if not isinstance(test_inputs, dict):
        test_inputs = {}

    results: Dict[str, Any] = {}
    started_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    semaphore = asyncio.Semaphore(max(1, max_concurrency))

    async def run_tool(tool_name: str) -> None:
        if tool_name not in registry_names:
            results[tool_name] = {"status": "skipped", "reason": "not_in_registry"}
            return
        if not unsafe_ok and _is_unsafe_tool(tool_name):
            results[tool_name] = {"status": "skipped", "reason": "unsafe_tool_name"}
            return
        schema_entry = schema_by_name.get(tool_name)
        cases, case_reason = _build_test_cases(tool_name, schema_entry, test_inputs, REPO_ROOT)
        if not cases:
            results[tool_name] = {"status": "skipped", "reason": case_reason}
            return

        tool_cases = []
        async with semaphore:
            for args in cases:
                case_result = await _run_tool_case(tool_name, args, timeout)
                tool_cases.append({"args": args, **case_result})
        status = "ok" if all(case.get("ok") for case in tool_cases) else "failed"
        results[tool_name] = {"status": status, "cases": tool_cases, "case_source": case_reason}

    await asyncio.gather(*[run_tool(name) for name in tool_names])

    summary = {"ok": 0, "failed": 0, "skipped": 0}
    for entry in results.values():
        status = entry.get("status")
        if status in summary:
            summary[status] += 1

    report = {
        "started_at": started_at,
        "schema_path": schema_path,
        "tools_dir": tools_dir or BASE_DIR,
        "count": len(tool_names),
        "summary": summary,
        "results": results,
    }

    if output_path:
        _write_json(output_path, report)
    return report


def _load_report_tool_list(report_path: str) -> List[str]:
    report = _load_json(report_path, {})
    if not isinstance(report, dict):
        return []
    tools = report.get("affected_tools") or report.get("tools") or report.get("tool_names")
    if isinstance(tools, list):
        return [tool for tool in tools if isinstance(tool, str)]
    return []


def _main() -> None:
    parser = argparse.ArgumentParser(description="Forge Manager tools.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    discover_parser = subparsers.add_parser("discover", help="Discover available tools.")
    discover_parser.add_argument("--tools-dir", default=None, help="Override tools directory.")
    discover_parser.add_argument("--output", default=None, help="Write discovery report JSON.")

    reconcile_parser = subparsers.add_parser("reconcile", help="Reconcile schema with discovered tools.")
    reconcile_parser.add_argument("--schema", default=DEFAULT_SCHEMA_PATH, help="Schema file path.")
    reconcile_parser.add_argument("--tools-dir", default=None, help="Override tools directory.")
    reconcile_parser.add_argument("--report", default=DEFAULT_REPORT_PATH, help="Write reconciliation report JSON.")
    reconcile_parser.add_argument("--apply-aliases", action="store_true", help="Rename schema tools using alias rules.")
    reconcile_parser.add_argument("--add-missing", action="store_true", help="Add missing schema entries.")
    reconcile_parser.add_argument("--prune-missing", action="store_true", help="Remove schema entries not in registry.")
    reconcile_parser.add_argument("--write-schema", action="store_true", help="Write schema changes to disk.")

    validate_parser = subparsers.add_parser("validate", help="Validate affected tools.")
    validate_parser.add_argument("--schema", default=DEFAULT_SCHEMA_PATH, help="Schema file path.")
    validate_parser.add_argument("--tools-dir", default=None, help="Override tools directory.")
    validate_parser.add_argument("--report", default=None, help="Reconciliation report with affected tools.")
    validate_parser.add_argument("--tools", nargs="*", default=None, help="Explicit tool names to validate.")
    validate_parser.add_argument("--test-inputs", default=DEFAULT_TEST_INPUTS_PATH, help="Tool test inputs JSON.")
    validate_parser.add_argument("--timeout", type=float, default=20.0, help="Per-tool timeout seconds.")
    validate_parser.add_argument("--max-concurrency", type=int, default=4, help="Max concurrent tool executions.")
    validate_parser.add_argument("--unsafe-ok", action="store_true", help="Allow potentially unsafe tool names.")
    validate_parser.add_argument("--output", default=DEFAULT_VALIDATE_REPORT_PATH, help="Write validation report JSON.")

    loop_parser = subparsers.add_parser("loop", help="Continuously reconcile schema and optionally validate tools.")
    loop_parser.add_argument("--schema", default=DEFAULT_SCHEMA_PATH, help="Schema file path.")
    loop_parser.add_argument("--tools-dir", default=None, help="Override tools directory.")
    loop_parser.add_argument("--report", default=DEFAULT_REPORT_PATH, help="Write reconciliation report JSON.")
    loop_parser.add_argument("--apply-aliases", action="store_true", help="Rename schema tools using alias rules.")
    loop_parser.add_argument("--add-missing", action="store_true", help="Add missing schema entries.")
    loop_parser.add_argument("--prune-missing", action="store_true", help="Remove schema entries not in registry.")
    loop_parser.add_argument("--write-schema", action="store_true", help="Write schema changes to disk.")
    loop_parser.add_argument("--validate", action="store_true", help="Validate affected tools each iteration.")
    loop_parser.add_argument("--validate-all", action="store_true", help="Validate all tools each iteration.")
    loop_parser.add_argument("--test-inputs", default=DEFAULT_TEST_INPUTS_PATH, help="Tool test inputs JSON.")
    loop_parser.add_argument("--timeout", type=float, default=20.0, help="Per-tool timeout seconds.")
    loop_parser.add_argument("--max-concurrency", type=int, default=4, help="Max concurrent tool executions.")
    loop_parser.add_argument("--unsafe-ok", action="store_true", help="Allow potentially unsafe tool names.")
    loop_parser.add_argument("--validate-report", default=DEFAULT_VALIDATE_REPORT_PATH, help="Write validation report JSON.")
    loop_parser.add_argument("--log", default=DEFAULT_LOOP_LOG_PATH, help="Append loop summary log.")
    loop_parser.add_argument("--interval", type=float, default=300.0, help="Seconds between iterations.")
    loop_parser.add_argument("--iterations", type=int, default=0, help="Stop after N iterations (0 = infinite).")

    args = parser.parse_args()

    if args.command == "discover":
        report = _discover_tools(args.tools_dir, args.output)
        print(json.dumps(report, indent=2, ensure_ascii=True))
        return

    if args.command == "reconcile":
        report = _reconcile_schema(
            schema_path=args.schema,
            tools_dir=args.tools_dir,
            apply_aliases=args.apply_aliases,
            add_missing=args.add_missing,
            prune_missing=args.prune_missing,
            output_path=args.report,
            write_schema=args.write_schema,
        )
        print(json.dumps(report, indent=2, ensure_ascii=True))
        return

    if args.command == "validate":
        tool_names = []
        if args.report:
            tool_names = _load_report_tool_list(args.report)
        if args.tools:
            tool_names = args.tools
        if not tool_names:
            print("No tools provided for validation.", file=sys.stderr)
            sys.exit(1)

        report = asyncio.run(
            _validate_tools(
                tool_names=tool_names,
                schema_path=args.schema,
                tools_dir=args.tools_dir,
                test_inputs_path=args.test_inputs,
                timeout=args.timeout,
                max_concurrency=args.max_concurrency,
                unsafe_ok=args.unsafe_ok,
                output_path=args.output,
            )
        )
        print(json.dumps(report, indent=2, ensure_ascii=True))
        return

    if args.command == "loop":
        iteration = 0
        while True:
            iteration += 1
            start = time.monotonic()
            report = _reconcile_schema(
                schema_path=args.schema,
                tools_dir=args.tools_dir,
                apply_aliases=args.apply_aliases,
                add_missing=args.add_missing,
                prune_missing=args.prune_missing,
                output_path=args.report,
                write_schema=args.write_schema,
            )
            summary = (
                f"iter={iteration} registry={report.get('registry_count')} "
                f"schema={report.get('schema_count')} "
                f"added={len(report.get('added', []))} "
                f"renamed={len(report.get('renamed', []))} "
                f"missing_in_registry={len(report.get('missing_in_registry', []))}"
            )
            print(summary)
            _append_log(args.log, summary)

            do_validate = args.validate or args.validate_all
            if do_validate:
                if args.validate_all:
                    registry = _build_registry(args.tools_dir)
                    tool_names = sorted(registry.keys())
                else:
                    tool_names = report.get("affected_tools") or []

                if tool_names:
                    validate_report = asyncio.run(
                        _validate_tools(
                            tool_names=tool_names,
                            schema_path=args.schema,
                            tools_dir=args.tools_dir,
                            test_inputs_path=args.test_inputs,
                            timeout=args.timeout,
                            max_concurrency=args.max_concurrency,
                            unsafe_ok=args.unsafe_ok,
                            output_path=args.validate_report,
                        )
                    )
                    val_summary = validate_report.get("summary", {})
                    val_line = (
                        f"validate ok={val_summary.get('ok', 0)} "
                        f"failed={val_summary.get('failed', 0)} "
                        f"skipped={val_summary.get('skipped', 0)}"
                    )
                    print(val_line)
                    _append_log(args.log, val_line)
                else:
                    _append_log(args.log, "validate skipped: no affected tools")

            if args.iterations and iteration >= args.iterations:
                break

            elapsed = time.monotonic() - start
            sleep_for = max(0.0, float(args.interval) - elapsed)
            time.sleep(sleep_for)
        return


if __name__ == "__main__":
    _main()
