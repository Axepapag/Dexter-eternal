import os
from typing import Dict, Any, List, Optional, Tuple

# Optional OCR dependency
try:
    import pytesseract  # type: ignore
    OCR_AVAILABLE = True
except Exception:
    pytesseract = None
    OCR_AVAILABLE = False


def _load_vision_deps() -> Tuple[Any, Any, Any, List[str]]:
    missing = []
    try:
        import cv2  # type: ignore
    except Exception:
        cv2 = None
        missing.append("opencv-python")
    try:
        import numpy as np  # type: ignore
    except Exception:
        np = None
        missing.append("numpy")
    try:
        import pyautogui  # type: ignore
    except Exception:
        pyautogui = None
        missing.append("pyautogui")
    return cv2, np, pyautogui, missing


def _configure_tesseract() -> None:
    """Point pytesseract at a known Tesseract binary when available."""
    if not OCR_AVAILABLE:
        return
    cmd = os.getenv("TESSERACT_CMD")
    if cmd and os.path.exists(cmd):
        pytesseract.pytesseract.tesseract_cmd = cmd
        return
    candidates = [
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
    ]
    for path in candidates:
        if os.path.exists(path):
            pytesseract.pytesseract.tesseract_cmd = path
            return

def run_test(profile: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Process the provided profile information to configure the scanning parameters 
    for the analysis engine and verify system status.
    """
    if profile is None:
        profile = {}

    # Load dependencies to check availability
    cv2, np, pyautogui, missing = _load_vision_deps()

    # Apply configuration from profile if available
    if "tesseract_cmd" in profile:
        os.environ["TESSERACT_CMD"] = profile["tesseract_cmd"]

    # Configure Tesseract path
    _configure_tesseract()

    # Return status report
    return {
        "status": "configured",
        "missing_dependencies": missing,
        "ocr_available": OCR_AVAILABLE,
        "profile_applied": profile
    }

def vision_analyze(operation: str, image_path: str = None, target_image: str = None, 
                   confidence: float = 0.8, region: Dict[str, int] = None, text_query: str = None) -> Dict[str, Any]:
    """Perform vision analysis"""
    cv2, np, pyautogui, missing = _load_vision_deps()
    if missing:
        raise ValueError(f"Missing dependencies: {', '.join(missing)}. Install them to use vision_ops.")
    _configure_tesseract()
    # Get the main image (screenshot or file)
    if image_path:
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not load image from {image_path}")
    else:
        # Take a screenshot if no image path provided
        if region:
            shot = pyautogui.screenshot(region=(region['x'], region['y'], region['width'], region['height']))
        else:
            shot = pyautogui.screenshot()
        img = cv2.cvtColor(np.array(shot), cv2.COLOR_RGB2BGR)

    if operation == "find_image":
        if not target_image:
            raise ValueError("target_image required for find_image")
        
        template = cv2.imread(target_image)
        if template is None:
            raise ValueError(f"Could not load target image from {target_image}")
        
        result = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
        
        if max_val >= confidence:
            h, w = template.shape[:2]
            center_x = max_loc[0] + w // 2
            center_y = max_loc[1] + h // 2
            return {
                "found": True,
                "confidence": float(max_val),
                "location": {"x": max_loc[0], "y": max_loc[1]},
                "center": {"x": center_x, "y": center_y},
                "size": {"width": w, "height": h}
            }
        else:
            return {"found": False, "confidence": float(max_val)}

    elif operation == "detect_text":
        if not OCR_AVAILABLE:
            raise ValueError("OCR not available (pytesseract missing)")
        
        # Convert to RGB for pytesseract
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        data = pytesseract.image_to_data(rgb_img, output_type=pytesseract.Output.DICT)
        
        found_text = []
        n_boxes = len(data['text'])
        for i in range(n_boxes):
            if int(data['conf'][i]) > 60:  # Confidence threshold
                text = data['text'][i].strip()
                if text:
                    if text_query and text_query.lower() not in text.lower():
                        continue
                    found_text.append({
                        "text": text,
                        "confidence": data['conf'][i],
                        "box": {
                            "x": data['left'][i],
                            "y": data['top'][i],
                            "width": data['width'][i],
                            "height": data['height'][i]
                        }
                    })
        return {"text_blocks": found_text}

    elif operation == "get_pixel_color":
        if region and region.get('x') is not None and region.get('y') is not None:
            # If region provided, use relative coordinates, else center
            x, y = region['x'], region['y']
        else:
            h, w = img.shape[:2]
            x, y = w // 2, h // 2
        
        if 0 <= y < img.shape[0] and 0 <= x < img.shape[1]:
            color = img[y, x]
            return {
                "bgr": [int(c) for c in color],
                "rgb": [int(color[2]), int(color[1]), int(color[0])],
                "coordinates": {"x": x, "y": y}
            }
        else:
            raise ValueError("Coordinates out of bounds")

    else:
        raise ValueError(f"Unknown operation: {operation}")