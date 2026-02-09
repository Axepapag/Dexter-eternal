#!/usr/bin/env python3
"""
Dexter Dependency Checker
Verifies all required Python packages are installed before starting Dexter.
"""

import sys
import subprocess

# Critical dependencies required for Dexter to run
CRITICAL_DEPS = [
    ('fastapi', 'FastAPI web framework'),
    ('uvicorn', 'ASGI server for FastAPI'),
    ('psutil', 'Process and system utilities'),
    ('pydantic', 'Data validation'),
]

# Important dependencies (system will run but with reduced functionality)
IMPORTANT_DEPS = [
    ('torch', 'PyTorch for TRM models'),
    ('aiohttp', 'Async HTTP client'),
    ('numpy', 'Numerical computing'),
]

# Optional dependencies
OPTIONAL_DEPS = [
    ('playwright', 'Browser automation'),
    ('opencv-python', 'Computer vision'),
    ('pyautogui', 'GUI automation'),
]

def check_import(module_name):
    """Try to import a module and return True if successful."""
    try:
        __import__(module_name.replace('-', '_'))
        return True
    except ImportError:
        return False

def main():
    print("=" * 60)
    print("  DEXTER DEPENDENCY CHECKER")
    print("=" * 60)
    print()
    
    critical_missing = []
    important_missing = []
    optional_missing = []
    
    # Check critical dependencies
    print("Checking CRITICAL dependencies...")
    for module, description in CRITICAL_DEPS:
        if check_import(module):
            print(f"  ✓ {module:20s} - {description}")
        else:
            print(f"  ✗ {module:20s} - {description} [MISSING]")
            critical_missing.append(module)
    print()
    
    # Check important dependencies
    print("Checking IMPORTANT dependencies...")
    for module, description in IMPORTANT_DEPS:
        if check_import(module):
            print(f"  ✓ {module:20s} - {description}")
        else:
            print(f"  ⚠ {module:20s} - {description} [MISSING]")
            important_missing.append(module)
    print()
    
    # Check optional dependencies
    print("Checking OPTIONAL dependencies...")
    for module, description in OPTIONAL_DEPS:
        if check_import(module):
            print(f"  ✓ {module:20s} - {description}")
        else:
            print(f"  - {module:20s} - {description} [NOT INSTALLED]")
            optional_missing.append(module)
    print()
    
    # Summary
    print("=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    
    if critical_missing:
        print(f"\n❌ CRITICAL: {len(critical_missing)} required packages missing!")
        print(f"   Missing: {', '.join(critical_missing)}")
        print("\n   Dexter CANNOT start without these packages.")
        print("\n   To install all dependencies:")
        print("   pip install -r requirements.txt")
        print("\n   Or install just the missing packages:")
        print(f"   pip install {' '.join(critical_missing)}")
        return 1
    
    if important_missing:
        print(f"\n⚠️  WARNING: {len(important_missing)} important packages missing.")
        print(f"   Missing: {', '.join(important_missing)}")
        print("\n   Dexter will start but some features may not work.")
        print("\n   To install:")
        print(f"   pip install {' '.join(important_missing)}")
    
    if optional_missing:
        print(f"\nℹ️  INFO: {len(optional_missing)} optional packages not installed.")
        print("   This is OK - these are only needed for specific features.")
    
    if not critical_missing and not important_missing and not optional_missing:
        print("\n✅ ALL DEPENDENCIES INSTALLED!")
        print("   Dexter is ready to start.")
    elif not critical_missing and not important_missing:
        print("\n✅ ALL REQUIRED DEPENDENCIES INSTALLED!")
        print("   Dexter is ready to start.")
    elif not critical_missing:
        print("\n✅ CRITICAL DEPENDENCIES INSTALLED!")
        print("   Dexter can start (recommend installing important packages).")
    
    print()
    return 0 if not critical_missing else 1

if __name__ == "__main__":
    sys.exit(main())
