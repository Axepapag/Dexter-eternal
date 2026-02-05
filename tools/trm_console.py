#!/usr/bin/env python3
"""
trm_console.py - Standalone TRM console (no Dexter runtime required).

Lets you directly interface with all three trained TRMs:
- Tool TRM: select_tool(task, context, last_error)
- Memory TRM: query(query, memory_type)
- Reasoning TRM: reason(prompt, context)

This script intentionally avoids depending on `dexter.py` so you can debug TRMs
in isolation.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, Optional


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


def _safe_json(obj: Any) -> str:
    try:
        return json.dumps(obj, ensure_ascii=False, indent=2, default=str)
    except Exception:
        return json.dumps({"repr": repr(obj)}, ensure_ascii=False, indent=2, default=str)


def _print_help() -> None:
    print(
        "\n".join(
            [
                "Commands:",
                "  help                         Show this help",
                "  status                       Show TRM readiness + carry steps (if available)",
                "  use tool|memory|reasoning     Select active TRM",
                "  reset                         Reset carry for active TRM (if supported)",
                "  tool <task>                   Run Tool TRM tool selection",
                "  memory <query>                Run Memory TRM query (memory_type=all)",
                "  memory_type <type> <query>    Memory query with explicit memory_type",
                "  reason <prompt>               Run Reasoning TRM reasoning",
                "  observe <sender> <type> <msg> Ingest observation into ALL TRMs (best-effort)",
                "  quit                          Exit",
                "",
                "Notes:",
                "  - Use quotes if your text contains spaces.",
                "  - TRMs must have checkpoints present under the TRM root to be 'ready'.",
            ]
        )
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Standalone console for Dexter TRMs.")
    parser.add_argument("--device", default="cpu", help="Device for TRMs (cpu|cuda). Default: cpu")
    args = parser.parse_args()

    from core.trained_trm_loader import get_tool_trm, get_memory_trm, get_reasoning_trm

    tool = get_tool_trm(device=args.device)
    memory = get_memory_trm(device=args.device)
    reasoning = get_reasoning_trm(device=args.device)

    active = "reasoning"

    def _carry_steps(trm: Any) -> Optional[int]:
        try:
            return int(getattr(getattr(trm, "carry", None), "steps", 0))
        except Exception:
            return None

    def _status() -> None:
        print(
            _safe_json(
                {
                    "active": active,
                    "tool_ready": bool(getattr(tool, "is_ready", lambda: False)()),
                    "tool_carry_steps": _carry_steps(tool),
                    "memory_ready": bool(getattr(memory, "is_ready", lambda: False)()),
                    "memory_carry_steps": _carry_steps(memory),
                    "reasoning_ready": bool(getattr(reasoning, "is_ready", lambda: False)()),
                    "reasoning_carry_steps": _carry_steps(reasoning),
                }
            )
        )

    print("TRM Console (standalone)")
    _status()
    _print_help()

    while True:
        try:
            raw = input("trm> ").strip()
        except EOFError:
            print()
            return 0
        except KeyboardInterrupt:
            print()
            continue

        if not raw:
            continue

        parts = raw.split(" ", 1)
        cmd = parts[0].strip().lower()
        rest = parts[1].strip() if len(parts) > 1 else ""

        if cmd in ("quit", "exit"):
            return 0
        if cmd == "help":
            _print_help()
            continue
        if cmd == "status":
            _status()
            continue

        if cmd == "use":
            target = (rest or "").strip().lower()
            if target in ("tool", "memory", "reasoning"):
                active = target
                print(f"active={active}")
            else:
                print("use tool|memory|reasoning")
            continue

        if cmd == "reset":
            trm = {"tool": tool, "memory": memory, "reasoning": reasoning}.get(active)
            if not trm:
                print("No active TRM selected.")
                continue
            if hasattr(trm, "reset_carry"):
                try:
                    trm.reset_carry()
                    print(f"{active}: carry reset")
                except Exception as exc:
                    print(f"{active}: reset failed: {exc}")
            else:
                print(f"{active}: reset not supported")
            continue

        if cmd == "tool":
            if not tool.is_ready():
                print("tool TRM not ready (missing checkpoint?)")
                continue
            task = rest
            if not task:
                print("tool <task>")
                continue
            try:
                out = tool.select_tool(task, context=None, last_error=None)
            except Exception as exc:
                out = {"error": str(exc)}
            print(_safe_json(out))
            continue

        if cmd == "memory":
            if not memory.is_ready():
                print("memory TRM not ready (missing checkpoint?)")
                continue
            q = rest
            if not q:
                print("memory <query>")
                continue
            try:
                out = memory.query(q, memory_type="all")
            except Exception as exc:
                out = {"error": str(exc)}
            print(_safe_json(out))
            continue

        if cmd == "memory_type":
            if not memory.is_ready():
                print("memory TRM not ready (missing checkpoint?)")
                continue
            sub = rest.split(" ", 1)
            if len(sub) < 2:
                print("memory_type <type> <query>")
                continue
            mtype = sub[0].strip()
            q = sub[1].strip()
            try:
                out = memory.query(q, memory_type=mtype)
            except Exception as exc:
                out = {"error": str(exc)}
            print(_safe_json(out))
            continue

        if cmd in ("reason", "reasoning"):
            if not reasoning.is_ready():
                print("reasoning TRM not ready (missing checkpoint?)")
                continue
            prompt = rest
            if not prompt:
                print("reason <prompt>")
                continue
            try:
                out = reasoning.reason(prompt, context={"origin": "trm_console"})
            except Exception as exc:
                out = {"error": str(exc)}
            print(_safe_json(out))
            continue

        if cmd == "observe":
            # Minimal: observe into all TRMs, best-effort.
            # Format: observe <sender> <type> <msg>
            sub = rest.split(" ", 2)
            if len(sub) < 3:
                print("observe <sender> <type> <msg>")
                continue
            sender, msg_type, msg = sub[0], sub[1], sub[2]
            for name, trm in (("tool", tool), ("memory", memory), ("reasoning", reasoning)):
                if not getattr(trm, "is_ready", lambda: False)():
                    continue
                try:
                    trm.ingest_message(sender, msg, msg_type, metadata={"via": "trm_console"})
                except Exception:
                    pass
            print("ok")
            continue

        print("Unknown command. Type 'help'.")


if __name__ == "__main__":
    raise SystemExit(main())

