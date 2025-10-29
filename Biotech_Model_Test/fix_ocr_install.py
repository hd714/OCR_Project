"""
Fix OCR Installation Issues
Run this to properly install EasyOCR and PaddleOCR
"""

import subprocess
import sys
from pathlib import Path

def run_command(cmd):
    """Run a command and show output"""
    print(f"\n>>> Running: {cmd}")
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.stdout:
            print(result.stdout)
        if result.stderr:
            print(f"Warnings/Errors: {result.stderr}")
        return result.returncode == 0
    except Exception as e:
        print(f"Failed: {e}")
        return False

def fix_easyocr():
    """Fix EasyOCR installation"""
    print("\n" + "="*60)
    print("FIXING EASYOCR")
    print("="*60)
    
    # Uninstall and reinstall
    commands = [
        f"{sys.executable} -m pip uninstall -y easyocr",
        f"{sys.executable} -m pip install easyocr==1.7.1",
        # Download models manually if needed
        f'{sys.executable} -c "import easyocr; reader = easyocr.Reader([\'en\'], gpu=False, verbose=True)"'
    ]
    
    for cmd in commands:
        if not run_command(cmd):
            print(f"Command failed, but continuing...")

def fix_paddleocr():
    """Fix PaddleOCR installation"""
    print("\n" + "="*60)
    print("FIXING PADDLEOCR")
    print("="*60)
    
    # Uninstall and reinstall in correct order
    commands = [
        f"{sys.executable} -m pip uninstall -y paddlepaddle paddleocr",
        f"{sys.executable} -m pip install paddlepaddle==2.6.1",
        f"{sys.executable} -m pip install paddleocr==2.7.3",
        # Test import and download models
        f'{sys.executable} -c "from paddleocr import PaddleOCR; ocr = PaddleOCR(use_angle_cls=True, lang=\'en\', use_gpu=False, show_log=True)"'
    ]
    
    for cmd in commands:
        if not run_command(cmd):
            print(f"Command failed, but continuing...")

def download_models_manually():
    """Download models manually if automatic download fails"""
    print("\n" + "="*60)
    print("DOWNLOADING MODELS MANUALLY")
    print("="*60)
    
    print("\nTrying to download EasyOCR models...")
    try:
        import easyocr
        reader = easyocr.Reader(['en'], gpu=False, verbose=True, download_enabled=True)
        print("✓ EasyOCR models downloaded")
    except Exception as e:
        print(f"✗ EasyOCR model download failed: {e}")
        print("\nYou may need to:")
        print("1. Check your internet connection")
        print("2. Try with a VPN if models are blocked in your region")
        print("3. Download models manually from: https://github.com/JaidedAI/EasyOCR/releases")
    
    print("\nTrying to download PaddleOCR models...")
    try:
        from paddleocr import PaddleOCR
        ocr = PaddleOCR(use_angle_cls=True, lang='en', use_gpu=False, show_log=True)
        # Run a test to trigger model download
        import numpy as np
        test_image = np.ones((100, 300, 3), dtype=np.uint8) * 255
        result = ocr.ocr(test_image)
        print("✓ PaddleOCR models downloaded")
    except Exception as e:
        print(f"✗ PaddleOCR model download failed: {e}")
        print("\nYou may need to:")
        print("1. Check your internet connection")
        print("2. Set proxy if behind firewall:")
        print("   set HTTP_PROXY=http://your-proxy:port")
        print("   set HTTPS_PROXY=http://your-proxy:port")

def main():
    print("="*60)
    print("OCR INSTALLATION FIX UTILITY")
    print("="*60)
    
    print("\nThis will reinstall EasyOCR and PaddleOCR properly.")
    response = input("Continue? (y/n): ")
    
    if response.lower() != 'y':
        print("Cancelled.")
        return
    
    # Fix installations
    fix_easyocr()
    fix_paddleocr()
    
    # Try to download models
    download_models_manually()
    
    print("\n" + "="*60)
    print("INSTALLATION COMPLETE")
    print("="*60)
    print("\nNow try running: python test_ocr.py")
    print("\nIf models still don't download automatically:")
    print("1. Check firewall/proxy settings")
    print("2. Try running with admin privileges")
    print("3. Download models manually and place in ~/.EasyOCR/model/")

if __name__ == "__main__":
    main()