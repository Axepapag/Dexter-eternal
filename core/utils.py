#!/usr/bin/env python3
import json
from typing import Optional, Dict, Any

def extract_json(text: str) -> Optional[Dict[str, Any]]:
    """Robustly extracts JSON from LLM response, handling reasoning tokens and Python pitfalls."""
    try:
        clean = text.strip()
        # Remove <think> blocks if present
        if "</think>" in clean:
            clean = clean.split("</think>")[-1].strip()
        
        if "```json" in clean:
            clean = clean.split("```json")[1].split("```")[0].strip()
        elif "```" in clean:
            clean = clean.split("```")[1].split("```")[0].strip()
        
        # Find the first brace/bracket and last brace/bracket
        obj_start = clean.find('{')
        list_start = clean.find('[')
        
        if obj_start != -1 and (list_start == -1 or obj_start < list_start):
            start = obj_start
            end = clean.rfind('}')
        elif list_start != -1:
            start = list_start
            end = clean.rfind(']')
        else:
            return None

        if start != -1 and end != -1:
            clean = clean[start:end+1]
        
        # Fix common Python-to-JSON pitfalls
        clean = clean.replace(": True", ": true").replace(": False", ": false").replace(": None", ": null")
            
        return json.loads(clean)
    except Exception:
        return None
