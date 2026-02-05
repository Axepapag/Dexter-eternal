#!/usr/bin/env python3
"""
UI Control for Dexter
Mouse, keyboard, and window automation
Enables Dexter to interact with desktop applications
"""

import os
import sys
import time
import subprocess
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path

__tool_prefix__ = "ui"

# Environment configuration
UI_FAILSAFE = os.getenv("UI_FAILSAFE", "false").lower() in ("1", "true", "yes")
UI_PAUSE = float(os.getenv("UI_PAUSE", "0.1"))

try:
    import pyautogui
    PYAUTOGUI_AVAILABLE = True
except ImportError:
    pyautogui = None
    PYAUTOGUI_AVAILABLE = False

# Common keyboard shortcuts
SHORTCUTS = {
    "copy": ["ctrl", "c"],
    "paste": ["ctrl", "v"],
    "cut": ["ctrl", "x"],
    "select_all": ["ctrl", "a"],
    "save": ["ctrl", "s"],
    "new_tab": ["ctrl", "t"],
    "close_tab": ["ctrl", "w"],
    "refresh": ["ctrl", "r"],
    "find": ["ctrl", "f"],
    "run_dialog": ["win", "r"],
    "desktop": ["win", "d"],
    "search": ["win", "s"],
    "task_manager": ["ctrl", "shift", "esc"],
    "screenshot": ["win", "shift", "s"],
}


def _ensure_pyautogui() -> Dict[str, Any]:
    """Check if pyautogui is available."""
    if not PYAUTOGUI_AVAILABLE:
        return {
            "success": False,
            "error": "pyautogui not installed. Run: pip install pyautogui",
            "install_command": "pip install pyautogui"
        }
    
    pyautogui.FAILSAFE = UI_FAILSAFE
    pyautogui.PAUSE = UI_PAUSE
    return {"success": True}


def ui_get_screen_size() -> Dict[str, Any]:
    """Get the screen resolution."""
    check = _ensure_pyautogui()
    if not check["success"]:
        return check
    
    size = pyautogui.size()
    return {
        "success": True,
        "width": size.width,
        "height": size.height,
        "resolution": f"{size.width}x{size.height}"
    }


def ui_get_mouse_position() -> Dict[str, Any]:
    """Get current mouse cursor position."""
    check = _ensure_pyautogui()
    if not check["success"]:
        return check
    
    pos = pyautogui.position()
    return {
        "success": True,
        "x": pos.x,
        "y": pos.y,
        "coordinates": f"({pos.x}, {pos.y})"
    }


def ui_move_mouse(x: int, y: int, duration: float = 0.5) -> Dict[str, Any]:
    """
    Move mouse to coordinates.
    
    Args:
        x: X coordinate
        y: Y coordinate
        duration: Movement duration in seconds
    """
    check = _ensure_pyautogui()
    if not check["success"]:
        return check
    
    try:
        pyautogui.moveTo(x, y, duration=duration)
        return {
            "success": True,
            "action": "move_to",
            "coordinates": f"({x}, {y})",
            "duration": duration
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


def ui_click(x: Optional[int] = None, y: Optional[int] = None, 
             button: str = "left", clicks: int = 1) -> Dict[str, Any]:
    """
    Click at position (or current position if not specified).
    
    Args:
        x: X coordinate (optional)
        y: Y coordinate (optional)
        button: Mouse button (left, right, middle)
        clicks: Number of clicks
    """
    check = _ensure_pyautogui()
    if not check["success"]:
        return check
    
    try:
        if x is not None and y is not None:
            pyautogui.click(x, y, button=button, clicks=clicks)
            location = f"({x}, {y})"
        else:
            pyautogui.click(button=button, clicks=clicks)
            pos = pyautogui.position()
            location = f"({pos.x}, {pos.y})"
        
        return {
            "success": True,
            "action": "click",
            "button": button,
            "clicks": clicks,
            "location": location
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


def ui_type(text: str, interval: float = 0.01) -> Dict[str, Any]:
    """
    Type text at current cursor position.
    
    Args:
        text: Text to type
        interval: Delay between keystrokes
    """
    check = _ensure_pyautogui()
    if not check["success"]:
        return check
    
    try:
        pyautogui.typewrite(text, interval=interval)
        return {
            "success": True,
            "action": "type",
            "characters": len(text),
            "text_preview": text[:50] + "..." if len(text) > 50 else text
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


def ui_press_key(key: str, presses: int = 1) -> Dict[str, Any]:
    """
    Press a key or key combination.
    
    Args:
        key: Key name (e.g., 'enter', 'tab', 'ctrl+c')
        presses: Number of times to press
    """
    check = _ensure_pyautogui()
    if not check["success"]:
        return check
    
    try:
        # Handle key combinations like "ctrl+c"
        if "+" in key:
            keys = key.split("+")
            for _ in range(presses):
                pyautogui.keyDown(keys[0])
                for k in keys[1:]:
                    pyautogui.keyDown(k)
                for k in reversed(keys[1:]):
                    pyautogui.keyUp(k)
                pyautogui.keyUp(keys[0])
        else:
            pyautogui.press(key, presses=presses)
        
        return {
            "success": True,
            "action": "key_press",
            "key": key,
            "presses": presses
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


def ui_hotkey(*keys: str) -> Dict[str, Any]:
    """
    Press a hotkey combination.
    
    Args:
        *keys: Keys to press simultaneously (e.g., "ctrl", "c")
    """
    check = _ensure_pyautogui()
    if not check["success"]:
        return check
    
    try:
        pyautogui.hotkey(*keys)
        return {
            "success": True,
            "action": "hotkey",
            "keys": list(keys)
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


def ui_shortcut(name: str) -> Dict[str, Any]:
    """
    Execute a named shortcut.
    
    Args:
        name: Shortcut name (copy, paste, save, etc.)
    """
    if name not in SHORTCUTS:
        return {
            "success": False,
            "error": f"Unknown shortcut: {name}",
            "available_shortcuts": list(SHORTCUTS.keys())
        }
    
    return ui_hotkey(*SHORTCUTS[name])


def ui_screenshot(filename: Optional[str] = None, region: Optional[Tuple[int, int, int, int]] = None) -> Dict[str, Any]:
    """
    Take a screenshot.
    
    Args:
        filename: Path to save screenshot (optional)
        region: (x, y, width, height) to capture region (optional)
    """
    check = _ensure_pyautogui()
    if not check["success"]:
        return check
    
    try:
        if region:
            screenshot = pyautogui.screenshot(region=region)
        else:
            screenshot = pyautogui.screenshot()
        
        if filename:
            screenshot.save(filename)
            return {
                "success": True,
                "action": "screenshot",
                "saved_to": filename,
                "size": screenshot.size
            }
        else:
            # Return image info
            return {
                "success": True,
                "action": "screenshot",
                "size": screenshot.size,
                "mode": screenshot.mode
            }
    except Exception as e:
        return {"success": False, "error": str(e)}


def ui_launch_app(app_name: str, wait_time: float = 2.0) -> Dict[str, Any]:
    """
    Launch an application.
    
    Args:
        app_name: Application name or command
        wait_time: Seconds to wait after launch
    """
    try:
        # Common app mappings for Windows
        app_map = {
            "notepad": "notepad",
            "calculator": "calc",
            "chrome": "chrome",
            "edge": "msedge",
            "firefox": "firefox",
            "explorer": "explorer",
            "task_manager": "taskmgr",
            "cmd": "cmd",
            "powershell": "powershell",
            "vscode": "code",
        }
        
        command = app_map.get(app_name.lower(), app_name)
        
        subprocess.Popen(command, shell=True)
        time.sleep(wait_time)
        
        return {
            "success": True,
            "action": "launch_app",
            "app": app_name,
            "command": command,
            "wait_time": wait_time
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


def ui_find_and_click(image_path: str, confidence: float = 0.9) -> Dict[str, Any]:
    """
    Find an image on screen and click it.
    Requires opencv-python.
    
    Args:
        image_path: Path to image to find
        confidence: Matching confidence (0-1)
    """
    check = _ensure_pyautogui()
    if not check["success"]:
        return check
    
    try:
        location = pyautogui.locateOnScreen(image_path, confidence=confidence)
        if location:
            center = pyautogui.center(location)
            pyautogui.click(center.x, center.y)
            return {
                "success": True,
                "action": "find_and_click",
                "found_at": f"({location.left}, {location.top})",
                "size": f"{location.width}x{location.height}"
            }
        else:
            return {
                "success": False,
                "error": f"Image not found on screen: {image_path}",
                "confidence_used": confidence
            }
    except Exception as e:
        return {"success": False, "error": str(e)}


def ui_scroll(direction: str = "down", amount: int = 3) -> Dict[str, Any]:
    """
    Scroll the mouse wheel.
    
    Args:
        direction: "up" or "down"
        amount: Number of scroll units
    """
    check = _ensure_pyautogui()
    if not check["success"]:
        return check
    
    try:
        scroll_amount = amount if direction == "down" else -amount
        pyautogui.scroll(scroll_amount)
        return {
            "success": True,
            "action": "scroll",
            "direction": direction,
            "amount": amount
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


# Backwards compatibility
get_screen_size = ui_get_screen_size
get_mouse_position = ui_get_mouse_position
move_mouse = ui_move_mouse
click = ui_click
type_text = ui_type
press_key = ui_press_key
hotkey = ui_hotkey
screenshot = ui_screenshot
launch_app = ui_launch_app
