"""
Setup script for Ultimate OCR Comparison
Ensures all required packages are installed
"""

import subprocess
import sys

print("""
╔══════════════════════════════════════════════════════════╗
║            🔧 SETTING UP OCR COMPARISON 🔧              ║
╚══════════════════════════════════════════════════════════╝
""")

packages = [
    "pillow",
    "pytesseract", 
    "easyocr",
    "numpy",
]

print("📦 Installing required packages...\n")

for package in packages:
    print(f"Installing {package}...")
    subprocess.run([sys.executable, "-m", "pip", "install", package], 
                   capture_output=True, text=True)
    print(f"  ✅ {package} installed\n")

print("✨ Setup complete!")
print("\nTo run the comparison:")
print("  1. Easy mode: python run_comparison.py")
print("  2. Direct: python ultimate_ocr_comparison.py image1.jpg image2.jpg")
print("\n🚀 Ready to compare OCR engines!")
