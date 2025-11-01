#!/usr/bin/env python3
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
