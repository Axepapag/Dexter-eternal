import fnmatch
import json
import os
import re
import shutil
import string
from typing import Any, Dict, List, Optional
from error_util import ToolError, catch_errors

__tool_prefix__ = "file_ops"
def list_drives() -> List[str]:
    """List available drive roots on Windows."""
    drives = []
    for letter in string.ascii_uppercase:
        path = f"{letter}:\\"
        if os.path.exists(path):
            drives.append(path)
    return drives


def _resolve_path_arg(path: str, directory: Optional[str]) -> str:
    if directory and (not path or path == "."):
        return directory
    return path


def _list_directory(path: str, pattern: str = "*", recursive: bool = False,
                    include_hidden: bool = False) -> List[str]:
    if not os.path.exists(path):
        raise ToolError(f"Path not found: {path}", code="FS_NOT_FOUND", context={"path": path})
    
    if os.path.isfile(path):
        return [path]

    results: List[str] = []
    try:
        if recursive:
            for root, dirs, filenames in os.walk(path):
                if not include_hidden:
                    dirs[:] = [d for d in dirs if not d.startswith(".")]
                for filename in filenames:
                    if not include_hidden and filename.startswith("."):
                        continue
                    if fnmatch.fnmatch(filename, pattern):
                        results.append(os.path.join(root, filename))
        else:
            for item in os.listdir(path):
                if not include_hidden and item.startswith("."):
                    continue
                if fnmatch.fnmatch(item, pattern):
                    results.append(os.path.join(path, item))
    except PermissionError as e:
        raise ToolError(f"Permission denied accessing {path}", code="FS_PERMISSION_DENIED", context={"path": path}) from e
    except Exception as e:
        raise ToolError(f"Failed to list directory {path}: {str(e)}", code="FS_LIST_FAILED", context={"path": path}) from e

    return sorted(results)


@catch_errors("FS")
def read_file(path: Optional[str] = None, file_path: Optional[str] = None,
              encoding: str = "utf-8", binary: bool = False,
              lines: Optional[int] = None, start_line: int = 0,
              end_line: Optional[int] = None) -> Any:
    """Read file content."""
    target = path or file_path
    if not target:
        raise ToolError("Missing path", code="FS_MISSING_ARG")
    
    if not os.path.exists(target):
        raise ToolError(f"File not found: {target}", code="FS_NOT_FOUND", context={"path": target})
        
    try:
        if binary:
            with open(target, "rb") as fh:
                return fh.read()
        with open(target, "r", encoding=encoding) as fh:
            if lines:
                return "".join([fh.readline() for _ in range(lines)])
            if start_line or end_line is not None:
                all_lines = fh.readlines()
                end_line = end_line if end_line is not None else len(all_lines)
                return "".join(all_lines[start_line:end_line])
            return fh.read()
    except Exception as e:
        raise ToolError(f"Error reading file {target}: {str(e)}", code="FS_READ_ERROR", context={"path": target}) from e


@catch_errors("FS")
def write_file(path: Optional[str] = None, content: Any = "",
               file_path: Optional[str] = None, append: bool = False,
               encoding: str = "utf-8", binary: bool = False,
               overwrite: bool = False) -> Dict[str, Any]:
    """Write content to file. Refuses to overwrite existing files unless overwrite=True."""
    target = path or file_path
    if not target:
        raise ToolError("Missing path", code="FS_MISSING_ARG")
        
    if os.path.exists(target) and not append and not overwrite:
        return {
            "success": False,
            "error": "Refusing to overwrite existing file without overwrite=True",
            "path": target,
            "requires_overwrite": True,
            "code": "FS_OVERWRITE_DENIED"
        }
        
    parent_dir = os.path.dirname(target)
    if parent_dir and not os.path.exists(parent_dir):
        try:
            os.makedirs(parent_dir, exist_ok=True)
        except Exception as e:
            raise ToolError(f"Failed to create directory {parent_dir}: {str(e)}", code="FS_DIR_CREATE_FAILED") from e
            
    if isinstance(content, bytes):
        binary = True
    if isinstance(content, (dict, list)) and not binary:
        content = json.dumps(content, indent=2, ensure_ascii=False)
    
    if append:
        mode = "ab" if binary else "a"
    else:
        mode = "wb" if binary else "w"
        
    if binary and isinstance(content, str):
        content = content.encode(encoding)
        
    try:
        with open(target, mode, encoding=None if binary else encoding) as fh:
            fh.write(content)
    except Exception as e:
        raise ToolError(f"Error writing to file {target}: {str(e)}", code="FS_WRITE_ERROR") from e
        
    bytes_written = len(content) if isinstance(content, (bytes, bytearray)) else len(str(content))
    return {
        "success": True,
        "path": target,
        "bytes_written": bytes_written,
        "append": append,
        "overwrite": overwrite,
        "binary": binary,
    }


@catch_errors("FS")
def replace_text(path: str, search: str, replace: str,
                 count: int = 0, use_regex: bool = False,
                 flags: str = "", encoding: str = "utf-8") -> Dict[str, Any]:
    """Replace text in a file without rewriting unrelated content."""
    if not path:
        raise ToolError("Missing path", code="FS_MISSING_ARG")
    if search is None:
        raise ToolError("Missing search text", code="FS_MISSING_ARG")
    if not os.path.isfile(path):
        raise ToolError(f"File not found: {path}", code="FS_NOT_FOUND")
        
    with open(path, "r", encoding=encoding) as fh:
        content = fh.read()

    if use_regex:
        flag_map = {
            "i": re.IGNORECASE,
            "m": re.MULTILINE,
            "s": re.DOTALL,
        }
        re_flags = 0
        for ch in flags or "":
            re_flags |= flag_map.get(ch.lower(), 0)
        pattern = re.compile(search, re_flags)
        new_content, replacements = pattern.subn(replace, content, count=count if count > 0 else 0)
    else:
        if count and count > 0:
            new_content = content.replace(search, replace, count)
            occurrences = content.count(search)
            replacements = min(count, occurrences)
        else:
            new_content = content.replace(search, replace)
            replacements = content.count(search)

    if new_content == content:
        return {
            "success": True,
            "path": path,
            "changed": False,
            "replacements": 0,
        }

    with open(path, "w", encoding=encoding) as fh:
        fh.write(new_content)

    return {
        "success": True,
        "path": path,
        "changed": True,
        "replacements": replacements,
        "bytes_written": len(new_content.encode(encoding)),
    }


@catch_errors("FS")
def list_files(directory: str = ".", path: Optional[str] = None, pattern: str = "*",
               recursive: bool = False, include_hidden: bool = False) -> List[str]:
    """List files in directory."""
    if path and (directory == "." or directory is None):
        directory = path
    return _list_directory(directory, pattern=pattern, recursive=recursive, include_hidden=include_hidden)


@catch_errors("FS")
def list_dir(path: str = ".", directory: Optional[str] = None, pattern: str = "*",
             recursive: bool = False, include_hidden: bool = False) -> List[str]:
    """Alias for list_files (legacy name)."""
    return _list_directory(
        _resolve_path_arg(path, directory),
        pattern=pattern,
        recursive=recursive,
        include_hidden=include_hidden,
    )


@catch_errors("FS")
def list_directory(path: str = ".", directory: Optional[str] = None, pattern: str = "*",
                   recursive: bool = False, include_hidden: bool = False) -> List[str]:
    """Alias for list_files (legacy name)."""
    return _list_directory(
        _resolve_path_arg(path, directory),
        pattern=pattern,
        recursive=recursive,
        include_hidden=include_hidden,
    )


@catch_errors("FS")
def copy_file(src: str, dst: str, overwrite: bool = True) -> bool:
    """Copy a file."""
    if not overwrite and os.path.exists(dst):
        raise ToolError(f"Destination exists: {dst}", code="FS_EXISTS")
    shutil.copy2(src, dst)
    return True


@catch_errors("FS")
def move_file(src: str, dst: str) -> bool:
    """Move a file."""
    shutil.move(src, dst)
    return True


@catch_errors("FS")
def delete_file(path: str) -> bool:
    """Delete a file."""
    if not os.path.exists(path):
        raise ToolError(f"File not found: {path}", code="FS_NOT_FOUND")
    os.remove(path)
    return True


@catch_errors("FS")
def copy_directory(src: str, dst: str, overwrite: bool = False) -> bool:
    """Copy a directory tree."""
    if os.path.exists(dst):
        if overwrite:
            shutil.rmtree(dst)
        else:
            raise ToolError(f"Destination exists: {dst}", code="FS_EXISTS")
    shutil.copytree(src, dst)
    return True


@catch_errors("FS")
def move_directory(src: str, dst: str) -> bool:
    """Move a directory tree."""
    shutil.move(src, dst)
    return True


@catch_errors("FS")
def delete_directory(path: str) -> bool:
    """Delete a directory tree."""
    shutil.rmtree(path)
    return True



@catch_errors("FS")
def chunked_read(path: str, chunk_size: int = 8192) -> List[str]:
    """Read a large file in chunks."""
    chunks = []
    if not os.path.exists(path):
        raise ToolError(f"File not found: {path}", code="FS_NOT_FOUND")
    with open(path, "r", encoding="utf-8") as fh:
        while True:
            chunk = fh.read(chunk_size)
            if not chunk:
                break
            chunks.append(chunk)
    return chunks


@catch_errors("FS")
def chunked_write(path: str, chunks: List[str], mode: str = "w") -> bool:
    """Write chunks to a file."""
    with open(path, mode, encoding="utf-8") as fh:
        for chunk in chunks:
            fh.write(chunk)
    return True


@catch_errors("FS")
def search_files(path: str, pattern: str = "*", query: Optional[str] = None,
                  recursive: bool = True, max_results: int = 100) -> Dict[str, Any]:
    """Search for files by name and optionally by content."""
    if not os.path.exists(path):
        raise ToolError(f"Path not found: {path}", code="FS_NOT_FOUND")

    results: List[Dict[str, Any]] = []
    count = 0

    for root, dirs, files in os.walk(path):
        if not recursive and root != path:
            break
        for filename in files:
            if count >= max_results:
                break
            if fnmatch.fnmatch(filename, pattern):
                file_path = os.path.join(root, filename)
                entry = {"path": file_path, "matches": []}
                if query:
                    matches = _search_in_file(file_path, query)
                    if matches:
                        entry["matches"] = matches
                        results.append(entry)
                        count += 1
                else:
                    results.append(entry)
                    count += 1
        if count >= max_results:
            break

    return {"results": results, "truncated": count >= max_results}


def _search_in_file(path: str, query: str) -> List[Dict[str, Any]]:
    matches: List[Dict[str, Any]] = []
    try:
        with open(path, "r", encoding="utf-8") as fh:
            for line_num, line in enumerate(fh, 1):
                if query.lower() in line.lower():
                    matches.append({
                        "line_number": line_num,
                        "line_content": line.rstrip("\n"),
                        "match_position": line.lower().find(query.lower()),
                    })
    except (UnicodeDecodeError, PermissionError, OSError):
        # We might want to log this but for search, skipping unreadable files is common.
        pass
    return matches
