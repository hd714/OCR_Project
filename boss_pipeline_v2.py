#!/usr/bin/env python3
"""Boss OCR Pipeline with HTML Dashboard"""

import sys
import os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "Biotech_Model_Test"))
sys.path.insert(0, str(Path(__file__).parent / "Biotech_Model_Test" / "src"))

from datetime import datetime
import webbrowser
import json

# Import your components
from local_ocr.ocr_tesseract import TesseractOCR
from local_ocr.ocr_easyocr import EasyOCROCR

class BossPipeline:
    def __init__(self):
        self.results_dir = Path("boss_results")
        self.results_dir.mkdir(exist_ok=True)
        
    def process_file(self, file_path):
        file_path = Path(file_path)
        print(f"\n{'='*60}")
        print(f"Processing: {file_path.name}")
        print(f"{'='*60}")
        
        results = {}
        
        # Run Tesseract
        print("\nTesseract OCR...")
        try:
            tesseract = TesseractOCR(tesseract_cmd=r"C:\Program Files\Tesseract-OCR\tesseract.exe")
            tess_result = tesseract.process(file_path)
            results["tesseract"] = {
                "text": tess_result.text[:1000],
                "words": tess_result.word_count,
                "time": tess_result.processing_time,
                "confidence": tess_result.confidence
            }
            print(f"  Extracted {tess_result.word_count} words in {tess_result.processing_time:.2f}s")
        except Exception as e:
            print(f"  Failed: {e}")
            results["tesseract"] = {"error": str(e)}
        
        # Run EasyOCR
        print("\nEasyOCR...")
        try:
            easyocr = EasyOCROCR(use_gpu=False)
            easy_result = easyocr.process(file_path)
            results["easyocr"] = {
                "text": easy_result.text[:1000],
                "words": easy_result.word_count,
                "time": easy_result.processing_time,
                "confidence": easy_result.confidence
            }
            print(f"  Extracted {easy_result.word_count} words in {easy_result.processing_time:.2f}s")
        except Exception as e:
            print(f"  Failed: {e}")
            results["easyocr"] = {"error": str(e)}
        
        # Generate HTML
        self.generate_html(file_path.name, results)
        
        return results
    
    def generate_html(self, filename, results):
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>OCR Results - {filename}</title>
            <style>
                body {{ font-family: Arial; margin: 20px; background: #f5f5f5; }}
                .header {{ background: linear-gradient(135deg, #667eea, #764ba2); color: white; padding: 30px; border-radius: 10px; }}
                .engine {{ background: white; padding: 20px; margin: 20px 0; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
                .metric {{ display: inline-block; margin: 10px 20px; }}
                .metric-value {{ font-size: 24px; font-weight: bold; color: #667eea; }}
                .text-preview {{ background: #f8f9fa; padding: 15px; border-radius: 5px; font-family: monospace; font-size: 12px; max-height: 200px; overflow-y: auto; }}
                .winner {{ background: #d4f8d4; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>OCR Pipeline Results</h1>
                <p>File: {filename}</p>
                <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
        """
        
        for engine, result in results.items():
            if "error" not in result:
                html += f"""
                <div class="engine">
                    <h2>{engine.upper()}</h2>
                    <div class="metric">
                        <div>Words Extracted</div>
                        <div class="metric-value">{result.get('words', 0)}</div>
                    </div>
                    <div class="metric">
                        <div>Processing Time</div>
                        <div class="metric-value">{result.get('time', 0):.2f}s</div>
                    </div>
                    <div class="metric">
                        <div>Confidence</div>
                        <div class="metric-value">{result.get('confidence', 0):.2%}</div>
                    </div>
                    <h3>Text Preview:</h3>
                    <div class="text-preview">{result.get('text', 'No text extracted')}</div>
                </div>
                """
        
        html += "</body></html>"
        
        # Save and open
        output_file = self.results_dir / f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html)
        
        print(f"\nHTML Dashboard saved: {output_file}")
        webbrowser.open(str(output_file))

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Boss OCR Pipeline")
    parser.add_argument("file", help="File to process")
    args = parser.parse_args()
    
    if not Path(args.file).exists():
        print(f"Error: File not found: {args.file}")
        return
    
    pipeline = BossPipeline()
    pipeline.process_file(args.file)
    print("\nDone! Check the HTML dashboard in your browser.")

if __name__ == "__main__":
    main()
