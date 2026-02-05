#!/usr/bin/env python3
"""
Librarian Health Check
Reports which skills are importable and which are failing.
"""

import os
from pathlib import Path
from typing import Any, Dict

__tool_prefix__ = "librarian"


def skills_health() -> Dict[str, Any]:
    """Return an import health report for all skills."""
    repo_root = Path(os.path.dirname(os.path.abspath(__file__))).parent
    from core.skill_librarian import SkillLibrarian
    lib = SkillLibrarian(repo_root)
    return lib.health_report()


def heal_dependencies() -> Dict[str, Any]:
    """
    Attempt to auto-install missing dependencies for skills that failed to import.
    Uses allow/deny lists from core_config.json (dependency_auto_install).
    """
    repo_root = Path(os.path.dirname(os.path.abspath(__file__))).parent
    from core.skill_librarian import SkillLibrarian
    from core.dependency_installer import install_for_missing_modules

    lib = SkillLibrarian(repo_root)
    report = lib.health_report()
    missing_modules = set()
    for info in report.get("details", {}).values():
        mod = info.get("missing_module")
        if mod:
            missing_modules.add(mod)

    cfg = lib.config.get("dependency_auto_install", {}) or {}
    allowlist = set(cfg.get("allowlist") or [])
    denylist = set(cfg.get("denylist") or [])
    result = install_for_missing_modules(missing_modules, allowlist=allowlist, denylist=denylist)
    return {
        "success": result.get("success", False),
        "missing_modules": sorted(missing_modules),
        "install_result": result,
    }


def run_test():
    return skills_health()
