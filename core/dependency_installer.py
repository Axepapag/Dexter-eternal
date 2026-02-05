import re
import subprocess
import sys
from typing import Dict, Iterable, List, Optional, Set

MODULE_TO_PACKAGE: Dict[str, str] = {
    "cv2": "opencv-python",
    "PIL": "Pillow",
    "bs4": "beautifulsoup4",
    "sklearn": "scikit-learn",
}


def module_to_package(module_name: str) -> str:
    return MODULE_TO_PACKAGE.get(module_name, module_name)


def extract_missing_module(error: Exception | str) -> Optional[str]:
    text = str(error)
    # ModuleNotFoundError: No module named 'xyz'
    match = re.search(r"No module named ['\"]([^'\"]+)['\"]", text)
    if match:
        return match.group(1)
    return None


def install_packages(packages: Iterable[str]) -> Dict[str, object]:
    pkgs = [p for p in packages if p]
    if not pkgs:
        return {"success": False, "error": "No packages to install", "installed": []}
    cmd = [sys.executable, "-m", "pip", "install", *pkgs]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        success = result.returncode == 0
        return {
            "success": success,
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "installed": pkgs if success else [],
        }
    except Exception as exc:
        return {"success": False, "error": str(exc), "installed": []}


def install_for_missing_modules(
    modules: Iterable[str],
    allowlist: Optional[Set[str]] = None,
    denylist: Optional[Set[str]] = None,
) -> Dict[str, object]:
    allowlist = allowlist or set()
    denylist = denylist or set()

    packages: List[str] = []
    skipped: List[str] = []
    for module in modules:
        pkg = module_to_package(module)
        if pkg in denylist or module in denylist:
            skipped.append(module)
            continue
        if allowlist and (pkg not in allowlist and module not in allowlist):
            skipped.append(module)
            continue
        packages.append(pkg)

    if not packages:
        return {"success": False, "error": "No allowed packages to install", "installed": [], "skipped": skipped}

    result = install_packages(packages)
    result["skipped"] = skipped
    return result
