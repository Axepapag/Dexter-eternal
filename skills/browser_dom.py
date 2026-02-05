from typing import Any, Dict, List, Optional

try:
    from playwright.sync_api import sync_playwright
    PLAYWRIGHT_AVAILABLE = True
except Exception:
    PLAYWRIGHT_AVAILABLE = False

__tool_prefix__ = "browser"


def _require_playwright() -> None:
    if not PLAYWRIGHT_AVAILABLE:
        raise RuntimeError(
            "playwright library not found. Install with 'pip install playwright' and run 'playwright install'."
        )


def _open_page(headless: bool, timeout_ms: int, session_path: Optional[str], url: Optional[str]):
    _require_playwright()
    playwright = sync_playwright().start()
    browser = playwright.chromium.launch(headless=headless)
    if session_path:
        context = browser.new_context(storage_state=session_path)
    else:
        context = browser.new_context()
    page = context.new_page()
    page.set_default_timeout(timeout_ms)
    if url:
        page.goto(url, wait_until="domcontentloaded")
    return playwright, browser, context, page


def _close(playwright, browser, context) -> None:
    try:
        context.close()
    except Exception:
        pass
    try:
        browser.close()
    except Exception:
        pass
    try:
        playwright.stop()
    except Exception:
        pass


def _run_steps(page, steps: Optional[List[Dict[str, Any]]]) -> List[str]:
    logs: List[str] = []
    for idx, step in enumerate(steps or []):
        action = (step.get("action") or "").lower()
        logs.append(f"step {idx + 1}: {action}")
        if action == "goto":
            page.goto(step.get("url"), wait_until="domcontentloaded")
        elif action == "click":
            if step.get("selector"):
                page.click(step["selector"])
            elif step.get("text"):
                page.get_by_text(step["text"]).click()
        elif action == "type":
            selector = step.get("selector")
            text = step.get("text", "")
            if selector:
                page.fill(selector, text)
        elif action == "press":
            selector = step.get("selector")
            key = step.get("key", "Enter")
            if selector:
                page.press(selector, key)
        elif action == "wait_for":
            selector = step.get("selector")
            text = step.get("text")
            if selector:
                page.wait_for_selector(selector, state=step.get("state", "visible"))
            elif text:
                page.get_by_text(text).wait_for()
        elif action == "screenshot":
            path = step.get("path", f"browser_step_{idx}.png")
            page.screenshot(path=path, full_page=step.get("full_page", True))
    return logs


def click(
    selector: Optional[str] = None,
    text: Optional[str] = None,
    url: Optional[str] = None,
    headless: bool = True,
    timeout_ms: int = 15000,
    session_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Click an element by selector or visible text."""
    if not selector and not text:
        return {"success": False, "error": "Missing selector or text"}
    playwright, browser, context, page = _open_page(headless, timeout_ms, session_path, url)
    try:
        if selector:
            page.click(selector)
        else:
            page.get_by_text(text).click()
        return {"success": True, "selector": selector, "text": text, "url": url}
    except Exception as exc:
        return {"success": False, "error": str(exc), "selector": selector, "text": text, "url": url}
    finally:
        _close(playwright, browser, context)


def type(
    selector: str,
    text: str,
    url: Optional[str] = None,
    headless: bool = True,
    timeout_ms: int = 15000,
    session_path: Optional[str] = None,
    clear: bool = True,
) -> Dict[str, Any]:
    """Type text into an element."""
    if not selector:
        return {"success": False, "error": "Missing selector"}
    playwright, browser, context, page = _open_page(headless, timeout_ms, session_path, url)
    try:
        if clear:
            page.fill(selector, text)
        else:
            page.type(selector, text)
        return {"success": True, "selector": selector, "text": text, "url": url}
    except Exception as exc:
        return {"success": False, "error": str(exc), "selector": selector, "url": url}
    finally:
        _close(playwright, browser, context)


def wait_for(
    selector: Optional[str] = None,
    text: Optional[str] = None,
    url: Optional[str] = None,
    state: str = "visible",
    headless: bool = True,
    timeout_ms: int = 15000,
    session_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Wait for an element or text to appear."""
    if not selector and not text:
        return {"success": False, "error": "Missing selector or text"}
    playwright, browser, context, page = _open_page(headless, timeout_ms, session_path, url)
    try:
        if selector:
            page.wait_for_selector(selector, state=state)
        else:
            page.get_by_text(text).wait_for()
        return {"success": True, "selector": selector, "text": text, "url": url}
    except Exception as exc:
        return {"success": False, "error": str(exc), "selector": selector, "text": text, "url": url}
    finally:
        _close(playwright, browser, context)


def download(
    selector: Optional[str] = None,
    text: Optional[str] = None,
    url: Optional[str] = None,
    save_path: Optional[str] = None,
    headless: bool = True,
    timeout_ms: int = 15000,
    session_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Click a download link/button and save the file."""
    if not selector and not text:
        return {"success": False, "error": "Missing selector or text"}
    playwright, browser, context, page = _open_page(headless, timeout_ms, session_path, url)
    try:
        with page.expect_download() as download_info:
            if selector:
                page.click(selector)
            else:
                page.get_by_text(text).click()
        download_obj = download_info.value
        if save_path:
            download_obj.save_as(save_path)
        return {
            "success": True,
            "save_path": save_path or download_obj.path(),
            "suggested_filename": download_obj.suggested_filename,
        }
    except Exception as exc:
        return {"success": False, "error": str(exc), "selector": selector, "text": text, "url": url}
    finally:
        _close(playwright, browser, context)


def screenshot(
    url: Optional[str] = None,
    path: str = "browser_screenshot.png",
    selector: Optional[str] = None,
    full_page: bool = True,
    headless: bool = True,
    timeout_ms: int = 15000,
    session_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Capture a screenshot of the page or a selector."""
    playwright, browser, context, page = _open_page(headless, timeout_ms, session_path, url)
    try:
        if selector:
            page.locator(selector).screenshot(path=path)
        else:
            page.screenshot(path=path, full_page=full_page)
        return {"success": True, "path": path, "selector": selector, "url": url}
    except Exception as exc:
        return {"success": False, "error": str(exc), "path": path, "url": url}
    finally:
        _close(playwright, browser, context)


def extract_table(
    selector: str,
    url: Optional[str] = None,
    headless: bool = True,
    timeout_ms: int = 15000,
    session_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Extract table headers and rows from a selector."""
    if not selector:
        return {"success": False, "error": "Missing selector"}
    playwright, browser, context, page = _open_page(headless, timeout_ms, session_path, url)
    try:
        headers = page.eval_on_selector_all(
            f"{selector} thead th",
            "els => els.map(e => e.innerText.trim())",
        )
        rows = page.eval_on_selector_all(
            f"{selector} tbody tr",
            "rows => rows.map(r => Array.from(r.querySelectorAll('th,td')).map(c => c.innerText.trim()))",
        )
        return {"success": True, "headers": headers, "rows": rows, "selector": selector}
    except Exception as exc:
        return {"success": False, "error": str(exc), "selector": selector}
    finally:
        _close(playwright, browser, context)


def session_save(
    save_path: str,
    url: Optional[str] = None,
    steps: Optional[List[Dict[str, Any]]] = None,
    headless: bool = True,
    timeout_ms: int = 15000,
) -> Dict[str, Any]:
    """Run steps and save browser storage state to a file."""
    if not save_path:
        return {"success": False, "error": "Missing save_path"}
    playwright, browser, context, page = _open_page(headless, timeout_ms, None, url)
    try:
        logs = _run_steps(page, steps)
        context.storage_state(path=save_path)
        return {"success": True, "session_path": save_path, "logs": logs}
    except Exception as exc:
        return {"success": False, "error": str(exc), "session_path": save_path}
    finally:
        _close(playwright, browser, context)


def session_load(
    session_path: str,
    url: Optional[str] = None,
    steps: Optional[List[Dict[str, Any]]] = None,
    headless: bool = True,
    timeout_ms: int = 15000,
) -> Dict[str, Any]:
    """Load browser storage state and optionally run steps."""
    if not session_path:
        return {"success": False, "error": "Missing session_path"}
    playwright, browser, context, page = _open_page(headless, timeout_ms, session_path, url)
    try:
        logs = _run_steps(page, steps)
        return {"success": True, "session_path": session_path, "logs": logs}
    except Exception as exc:
        return {"success": False, "error": str(exc), "session_path": session_path}
    finally:
        _close(playwright, browser, context)
