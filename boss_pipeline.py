#!/usr/bin/env python3
"""
Boss-Friendly OCR Pipeline
Input any file -> Process with multiple engines -> View results in HTML dashboard
"""

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
            "timestamp": datetime.now().isoformat(),
            "files_processed": []
        }
    
    def process_file(self, file_path):
        """Process a single file with all available methods"""
        file_path = Path(file_path)
        print(f"\n{'='*60}")
        print(f"Processing: {file_path.name}")
        print(f"{'='*60}")
        
        result = {
            "filename": file_path.name,
            "file_type": file_path.suffix.lower(),
            "timestamp": datetime.now().isoformat(),
            "processing_results": {},
            "comparisons": {}
        }
        
        # Run Tesseract
        print("\nRunning Tesseract...")
        try:
            tesseract = TesseractOCR(tesseract_cmd=r"C:\Program Files\Tesseract-OCR\tesseract.exe")
            tess_result = tesseract.process(file_path)
            result["processing_results"]["tesseract"] = {
                "method": "Tesseract",
                "word_count": tess_result.word_count,
                "processing_time": tess_result.processing_time
            }
            print(f"  Extracted {tess_result.word_count} words")
        except Exception as e:
            print(f"  Failed: {e}")
        
        return result

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Boss OCR Pipeline")
    parser.add_argument("file", nargs="?", help="File to process")
    args = parser.parse_args()
    
    pipeline = BossPipeline()
    
    if args.file:
        result = pipeline.process_file(args.file)
        print("\nProcessing complete!")
    else:
        print("Usage: python boss_pipeline.py <file>")

if __name__ == "__main__":
    main()
