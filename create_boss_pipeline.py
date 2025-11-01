print("Creating Boss Pipeline files...")

# Create boss_pipeline.py
boss_pipeline_code = """#!/usr/bin/env python3
'''Boss-Friendly OCR Pipeline'''

import sys
import os
import json
import time
from pathlib import Path
from datetime import datetime
import webbrowser
from typing import Dict, List, Any

# Add your existing code to path
sys.path.insert(0, str(Path(__file__).parent / "Biotech_Model_Test"))
sys.path.insert(0, str(Path(__file__).parent / "Biotech_Model_Test" / "src"))

# Import your existing components
from main import OCRPipeline
from text_parsers.parse_pdfplumber import PDFPlumberParser
from text_parsers.parse_pypdf import PyPDFParser
from local_ocr.ocr_tesseract import TesseractOCR
from local_ocr.ocr_easyocr import EasyOCROCR

class BossPipeline:
    def __init__(self):
        self.results_dir = Path("boss_results")
        self.results_dir.mkdir(exist_ok=True)
        
        self.current_results = {
            'timestamp': datetime.now().isoformat(),
            'files_processed': []
        }

# ... rest of the code would go here
# For now, let's create a minimal working version

def main():
    print("Boss OCR Pipeline Starting...")
    print("Run with --interactive or provide a file path")

if __name__ == "__main__":
    main()
"""

with open('boss_pipeline.py', 'w', encoding='utf-8') as f:
    f.write(boss_pipeline_code)
print("✅ Created boss_pipeline.py")

# Create run_boss.bat
bat_content = """@echo off
echo ========================================
echo     BOSS OCR PIPELINE
echo ========================================
echo.

if "%1"=="" (
    echo Starting interactive mode...
    python boss_pipeline.py --interactive
) else (
    echo Processing: %1
    python boss_pipeline.py %1
)

pause
"""

with open('run_boss.bat', 'w', encoding='utf-8') as f:
    f.write(bat_content)
print("✅ Created run_boss.bat")

# Create test_boss.py
test_content = """from pathlib import Path

print("Testing Boss Pipeline Setup...")
print("="*50)

files_to_check = [
    "boss_pipeline.py",
    "Biotech_Model_Test/base_ocr.py",
]

for file in files_to_check:
    if Path(file).exists():
        print(f"✅ Found: {file}")
    else:
        print(f"❌ Missing: {file}")
"""

with open('test_boss.py', 'w', encoding='utf-8') as f:
    f.write(test_content)
print("✅ Created test_boss.py")

print("\n✅ All files created successfully!")
print("\nNow run: python test_boss.py")
