#!/usr/bin/env python3
"""
ResearchOps - Autonomous Research Capability
Allows Dexter Gliksbot to search for tools, niches, and opportunities.
"""

__tool_prefix__ = "research_ops"

import json
import os
import urllib.request
import urllib.parse
from html.parser import HTMLParser
from typing import Dict, Any, List

def ask_user(prompt: str) -> Dict[str, Any]:
    """
    Prompts the user for information via standard output.
    Returns a confirmation that the prompt was displayed.
    """
    print(f"[ResearchOps] Asking user: {prompt}", flush=True)
    # Note: In a non-interactive environment, we cannot wait for input().
    # We return a success status indicating the prompt was issued.
    return {
        "success": True,
        "action": "prompt_user",
        "prompt": prompt
    }

def _extract_text_from_html(html_content: str) -> str:
    """
    Helper function to strip HTML tags and extract visible text using standard library.
    """
    class TextExtractor(HTMLParser):
        def __init__(self):
            super().__init__()
            self.text = []
        def handle_data(self, data):
            self.text.append(data)
        def get_text(self):
            return " ".join(self.text)

    parser = TextExtractor()
    try:
        parser.feed(html_content)
        return parser.get_text()
    except Exception:
        return ""

def search(query: str) -> Dict[str, Any]:
    """
    Performs a web search to find relevant information on a topic.
    Uses standard library urllib to fetch results without external browser dependencies.
    """
    print(f"[Research] Searching for: {query}...", flush=True)
    
    try:
        # Construct search URL (using DuckDuckGo HTML version for simplicity)
        encoded_query = urllib.parse.quote_plus(query)
        url = f"https://duckduckgo.com/html/?q={encoded_query}"
        
        # Set a User-Agent to avoid basic blocking
        headers = {'User-Agent': 'Mozilla/5.0 (compatible; DexterGliksbot/1.0)'}
        req = urllib.request.Request(url, headers=headers)
        
        with urllib.request.urlopen(req, timeout=15) as response:
            html_content = response.read().decode('utf-8')
            
        # Extract text content
        raw_text = _extract_text_from_html(html_content)
        
        # Clean up the text: remove excessive whitespace and limit length
        snippet = " ".join(raw_text.split())[:4000]
        
        if not snippet:
            return {
                "success": False,
                "error": "Search returned no parseable content."
            }

        return {
            "success": True,
            "query": query,
            "summary": f"Retrieved search results for '{query}'.",
            "raw_data": snippet
        }
            
    except urllib.error.URLError as e:
        return {"success": False, "error": f"Network error: {str(e)}"}
    except Exception as e:
        return {"success": False, "error": f"Search failed: {str(e)}"}

def run_test(query: str = "current yields on US Treasuries German Bunds government bonds"):
    """
    Executes a search with a default query relevant to the current task context.
    """
    return search(query)