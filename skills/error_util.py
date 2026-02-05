from functools import wraps
from typing import Any, Callable, Dict


class ToolError(Exception):
    def __init__(self, message: str, code: str = "TOOL_ERROR", context: Dict[str, Any] | None = None):
        super().__init__(message)
        self.code = code
        self.context = context or {}


def catch_errors(prefix: str) -> Callable:
    """
    Decorator to standardize tool error handling.
    Returns a dict with success=False and error metadata on exceptions.
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except ToolError as exc:
                return {
                    "success": False,
                    "error": str(exc),
                    "code": f"{prefix}_{exc.code}",
                    "context": getattr(exc, "context", {}),
                }
            except Exception as exc:
                return {
                    "success": False,
                    "error": str(exc),
                    "code": f"{prefix}_UNCAUGHT",
                }
        return wrapper
    return decorator
