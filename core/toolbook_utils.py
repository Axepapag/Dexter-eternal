#!/usr/bin/env python3
import inspect
import json
import os
import re
import sys
import types
from typing import Any, Dict, List, Optional, Set, Tuple, Union, get_args, get_origin

def _schema_for_annotation(annotation: Any) -> Dict[str, Any]:
    """Convert Python type annotation to JSON schema type."""
    if annotation is inspect._empty or annotation is Any:
        return {"type": "string"}

    origin = get_origin(annotation)
    args = get_args(annotation)

    union_type = getattr(types, "UnionType", None)
    if origin is Union or (union_type and origin is union_type):
        non_null = [arg for arg in args if arg is not type(None)]
        if len(non_null) == 1:
            return _schema_for_annotation(non_null[0])
        return {"type": "object"}

    if origin is list or annotation is list:
        item_schema = _schema_for_annotation(args[0]) if args else {}
        schema = {"type": "array"}
        if item_schema:
            schema["items"] = item_schema
        return schema

    if origin is dict or annotation is dict:
        return {"type": "object"}

    if annotation is str:
        return {"type": "string"}
    if annotation is int:
        return {"type": "integer"}
    if annotation is float:
        return {"type": "number"}
    if annotation is bool:
        return {"type": "boolean"}

    if origin is not None:
        if origin is list: return {"type": "array"}
        if origin is dict: return {"type": "object"}
        if origin is tuple: return {"type": "array"}
        return {"type": "object"}

    return {"type": "string"}

def build_tool_schema(name: str, func) -> Dict[str, Any]:
    """Build an OpenAI-style function schema for a tool."""
    signature = inspect.signature(func)
    description = inspect.getdoc(func) or ""
    type_hints = {}
    try:
        type_hints = getattr(func, "__annotations__", {}) or {}
    except Exception:
        type_hints = {}

    properties: Dict[str, Any] = {}
    required = []

    for param in signature.parameters.values():
        if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            continue

        annotation = type_hints.get(param.name, param.annotation)
        prop_schema = _schema_for_annotation(annotation)
        if param.default is not inspect._empty:
            prop_schema["default"] = param.default
        else:
            required.append(param.name)

        properties[param.name] = prop_schema

    params_schema: Dict[str, Any] = {"type": "object", "properties": properties}
    if required:
        params_schema["required"] = required

    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": params_schema,
        },
    }

def build_tool_template(tool_name: str, func, provided: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Build a template tool call for an agent to fill out."""
    template_args = dict(provided or {})
    try:
        sig = inspect.signature(func)
    except Exception:
        return {"name": tool_name, "arguments": template_args}
    for param in sig.parameters.values():
        if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            continue
        if param.default is inspect._empty and param.name not in template_args:
            template_args[param.name] = f"<{param.name}>"
    return {"name": tool_name, "arguments": template_args}

def build_toolbook(registry: Dict[str, Any], include_docstrings: bool = True, include_schemas: bool = True) -> str:
    """Consolidated tool catalog for the ToolAgent."""
    entries = []
    for tool_name in sorted(registry.keys()):
        entry_val = registry[tool_name]
        # Handle tuple format (module_name, file_path, func) from tool_executor.py
        if isinstance(entry_val, tuple) and len(entry_val) == 3:
            func = entry_val[2]
        else:
            func = entry_val
            
        entry: Dict[str, Any] = {"name": tool_name}
        if include_docstrings:
            entry["description"] = inspect.getdoc(func) or ""
        if include_schemas:
            schema = build_tool_schema(tool_name, func).get("function", {})
            entry["schema"] = schema
        entry["template"] = build_tool_template(tool_name, func)
        entries.append(entry)
    return json.dumps(entries, indent=2, ensure_ascii=True)
