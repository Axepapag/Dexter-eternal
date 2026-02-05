import os
from typing import Any, Dict, List, Optional

# Try importing playwright
try:
    from playwright.sync_api import sync_playwright
    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False

def _require_playwright():
    if not PLAYWRIGHT_AVAILABLE:
        raise RuntimeError("playwright library not found. Please install it via 'pip install playwright' and 'playwright install'")

def browser_run_workflow(steps: List[Dict[str, Any]], headless: bool = True, timeout_ms: int = 15000) -> Dict[str, Any]:
    """
    Executes a sequence of browser actions.
    Steps format: [{'action': 'goto', 'url': '...'}, {'action': 'click', 'selector': '...'}, ...]
    """
    _require_playwright()
    
    results = {"success": True, "logs": [], "screenshots": []}
    
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=headless)
            page = browser.new_page()
            page.set_default_timeout(timeout_ms)
            
            for i, step in enumerate(steps):
                action = step.get("action", "").lower()
                results["logs"].append(f"Executing step {i+1}: {action}")
                
                if action == "goto":
                    page.goto(step.get("url"), wait_until="domcontentloaded")
                
                elif action == "click":
                    if step.get("selector"):
                        page.click(step["selector"])
                    elif step.get("text"):
                        page.get_by_text(step["text"]).click()
                        
                elif action == "type":
                    page.fill(step["selector"], step.get("text", ""))
                        
                elif action == "press":
                    page.press(step["selector"], step.get("key", "Enter"))
                        
                elif action == "screenshot":
                    path = step.get("path", f"browser_step_{i}.png")
                    page.screenshot(path=path)
                    results["screenshots"].append(path)
                        
                elif action == "get_text":
                    text = page.inner_text(step["selector"])
                    results[f"step_{i}_text"] = text
                        
            browser.close()
            return results
            
    except Exception as e:
        return {"success": False, "error": str(e), "logs": results["logs"]}

def browser_open_url(url: str, headless: bool = False) -> Dict[str, Any]:
    """
    Simple helper to open a URL and take a screenshot.
    """
    return browser_run_workflow([
        {"action": "goto", "url": url},
        {"action": "screenshot", "path": "url_preview.png"}
    ], headless=headless)

def run_test(headless: bool = True) -> Dict[str, Any]:
    """
    Runs a simple test to verify browser automation capabilities by visiting example.com.
    """
    return browser_run_workflow([
        {"action": "goto", "url": "https://example.com"},
        {"action": "screenshot", "path": "browser_test_screenshot.png"}
    ], headless=headless)