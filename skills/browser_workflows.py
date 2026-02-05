#!/usr/bin/env python3
"""
Browser Workflows for Dexter
Advanced web automation using Playwright
Handles search, navigation, data extraction, and form interaction
"""

import re
import time
from typing import Dict, Any, List, Optional, Tuple
from urllib.parse import quote_plus, urljoin
from dataclasses import dataclass

__tool_prefix__ = "browser"

# Try to import playwright
try:
    from playwright.sync_api import sync_playwright, Page, Browser
    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False
    sync_playwright = None


def _require_playwright() -> Dict[str, Any]:
    """Check if playwright is available."""
    if not PLAYWRIGHT_AVAILABLE:
        return {
            "success": False,
            "error": "playwright not installed",
            "install_commands": [
                "pip install playwright",
                "playwright install chromium"
            ]
        }
    return {"success": True}


def browser_search(query: str, engine: str = "duckduckgo", max_results: int = 5,
                  headless: bool = True, timeout_ms: int = 15000) -> Dict[str, Any]:
    """
    Search the web and return results.
    
    Args:
        query: Search query
        engine: Search engine (duckduckgo, bing)
        max_results: Maximum results to return
        headless: Run browser in headless mode
        timeout_ms: Page timeout in milliseconds
    """
    check = _require_playwright()
    if not check["success"]:
        return check
    
    engine = engine.lower().strip()
    if engine not in ("duckduckgo", "bing"):
        return {"success": False, "error": f"Unsupported engine: {engine}"}
    
    max_results = max(1, min(int(max_results), 20))
    
    # Configure based on engine
    if engine == "bing":
        url = f"https://www.bing.com/search?q={quote_plus(query)}"
        selectors = {
            "result": "li.b_algo",
            "title": "h2 a",
            "snippet": ".b_caption p"
        }
    else:  # duckduckgo
        url = f"https://duckduckgo.com/?q={quote_plus(query)}"
        selectors = {
            "result": "article[data-testid='result']",
            "title": "a[data-testid='result-title-a']",
            "snippet": "div[data-testid='result-snippet']"
        }
    
    results: List[Dict[str, Any]] = []
    
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=headless)
            page = browser.new_page()
            page.set_default_timeout(timeout_ms)
            
            try:
                page.goto(url, wait_until="domcontentloaded")
                
                # Wait for results to load
                page.wait_for_selector(selectors["result"], timeout=timeout_ms)
                
                items = page.locator(selectors["result"])
                total = items.count()
                
                for idx in range(min(total, max_results)):
                    try:
                        row = items.nth(idx)
                        title_el = row.locator(selectors["title"])
                        if title_el.count() == 0:
                            continue
                        
                        title = title_el.first.inner_text().strip()
                        href = title_el.first.get_attribute("href") or ""
                        
                        if engine == "bing":
                            href = urljoin("https://www.bing.com", href)
                        
                        snippet_el = row.locator(selectors["snippet"])
                        snippet = snippet_el.first.inner_text().strip() if snippet_el.count() else ""
                        
                        if title and href:
                            results.append({"title": title, "url": href, "snippet": snippet})
                    except Exception:
                        continue
                
            except Exception as e:
                browser.close()
                return {"success": False, "error": f"Navigation error: {str(e)}"}
            
            browser.close()
    
    except Exception as e:
        return {"success": False, "error": f"Browser error: {str(e)}"}
    
    return {
        "success": True,
        "query": query,
        "engine": engine,
        "count": len(results),
        "results": results
    }


def browser_navigate(url: str, wait_for: Optional[str] = None, 
                    headless: bool = True, timeout_ms: int = 30000) -> Dict[str, Any]:
    """
    Navigate to a URL and optionally wait for an element.
    
    Args:
        url: URL to navigate to
        wait_for: CSS selector to wait for (optional)
        headless: Run in headless mode
        timeout_ms: Timeout in milliseconds
    """
    check = _require_playwright()
    if not check["success"]:
        return check
    
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=headless)
            page = browser.new_page()
            page.set_default_timeout(timeout_ms)
            
            page.goto(url, wait_until="networkidle")
            
            if wait_for:
                page.wait_for_selector(wait_for, timeout=timeout_ms)
            
            title = page.title()
            final_url = page.url
            
            browser.close()
            
            return {
                "success": True,
                "url": final_url,
                "title": title,
                "waited_for": wait_for
            }
    
    except Exception as e:
        return {"success": False, "error": str(e)}


def browser_extract_text(url: str, selector: Optional[str] = None,
                        headless: bool = True) -> Dict[str, Any]:
    """
    Extract text content from a webpage.
    
    Args:
        url: URL to extract from
        selector: CSS selector to extract from (default: body)
        headless: Run in headless mode
    """
    check = _require_playwright()
    if not check["success"]:
        return check
    
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=headless)
            page = browser.new_page()
            
            page.goto(url, wait_until="domcontentloaded")
            
            if selector:
                element = page.locator(selector).first
                text = element.inner_text() if element.count() > 0 else ""
            else:
                text = page.locator("body").inner_text()
            
            browser.close()
            
            # Clean up text
            text = re.sub(r'\n+', '\n', text)
            text = re.sub(r'\s+', ' ', text)
            
            return {
                "success": True,
                "url": url,
                "text": text[:5000],  # Limit output
                "char_count": len(text)
            }
    
    except Exception as e:
        return {"success": False, "error": str(e)}


def browser_click(url: str, selector: str, headless: bool = True) -> Dict[str, Any]:
    """
    Navigate to URL and click an element.
    
    Args:
        url: URL to navigate to
        selector: CSS selector to click
        headless: Run in headless mode
    """
    check = _require_playwright()
    if not check["success"]:
        return check
    
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=headless)
            page = browser.new_page()
            
            page.goto(url, wait_until="domcontentloaded")
            page.click(selector)
            
            browser.close()
            
            return {
                "success": True,
                "url": url,
                "clicked": selector
            }
    
    except Exception as e:
        return {"success": False, "error": str(e)}


def browser_fill_form(url: str, fields: Dict[str, str], submit: bool = True,
                     headless: bool = True) -> Dict[str, Any]:
    """
    Fill a form on a webpage.
    
    Args:
        url: URL with form
        fields: Dict of {selector: value} to fill
        submit: Whether to submit the form
        headless: Run in headless mode
    """
    check = _require_playwright()
    if not check["success"]:
        return check
    
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=headless)
            page = browser.new_page()
            
            page.goto(url, wait_until="domcontentloaded")
            
            filled = []
            for selector, value in fields.items():
                page.fill(selector, value)
                filled.append(selector)
            
            if submit:
                # Try common submit methods
                try:
                    page.press("input[type='submit']", "Enter")
                except:
                    try:
                        page.click("button[type='submit']")
                    except:
                        pass
            
            browser.close()
            
            return {
                "success": True,
                "url": url,
                "fields_filled": filled,
                "submitted": submit
            }
    
    except Exception as e:
        return {"success": False, "error": str(e)}


def browser_screenshot(url: str, output_path: str, full_page: bool = False,
                      headless: bool = True) -> Dict[str, Any]:
    """
    Take a screenshot of a webpage.
    
    Args:
        url: URL to screenshot
        output_path: Path to save screenshot
        full_page: Capture full page or just viewport
        headless: Run in headless mode
    """
    check = _require_playwright()
    if not check["success"]:
        return check
    
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=headless)
            page = browser.new_page()
            
            page.goto(url, wait_until="networkidle")
            
            page.screenshot(path=output_path, full_page=full_page)
            
            browser.close()
            
            return {
                "success": True,
                "url": url,
                "saved_to": output_path,
                "full_page": full_page
            }
    
    except Exception as e:
        return {"success": False, "error": str(e)}


# Backwards compatibility
web_search = browser_search
navigate = browser_navigate
extract_text = browser_extract_text
