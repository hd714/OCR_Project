"""
Enhanced Boss Pipeline - OCR vs Text Parser Comparison
Simple interface for comparing OCR engines with PDF text parsers
Generates beautiful HTML dashboard automatically
"""

import sys
import os
from pathlib import Path
from datetime import datetime
import time
import webbrowser
import json
from typing import Dict, Optional, List
import traceback

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "Biotech_Model_Test"))
sys.path.insert(0, str(Path(__file__).parent / "Biotech_Model_Test" / "src"))

# Import the comparison module
from ocr_vs_parser_comparison import OCRvsParserComparison

# For additional processing
try:
    from pdf2image import convert_from_path
    PDF2IMAGE_AVAILABLE = True
except ImportError:
    PDF2IMAGE_AVAILABLE = False


class EnhancedBossPipeline:
    """Enhanced pipeline with comprehensive PDF analysis"""

    def __init__(self):
        """Initialize the enhanced pipeline"""
        self.output_dir = Path("boss_results")
        self.output_dir.mkdir(exist_ok=True)
        self.comparison_tool = OCRvsParserComparison(output_dir=str(self.output_dir))

    def process_document(self, file_path: str, auto_open: bool = True) -> Dict:
        """
        Process a document with comprehensive analysis

        Args:
            file_path: Path to PDF or image file
            auto_open: Whether to automatically open the HTML report

        Returns:
            Dictionary with processing results
        """
        file_path = Path(file_path)

        if not file_path.exists():
            print(f"[ERROR] File not found: {file_path}")
            return {"error": "File not found"}

        print(f"\n{'='*70}")
        print(f"ENHANCED BOSS PIPELINE - DOCUMENT ANALYSIS")
        print(f"{'='*70}")
        print(f"Processing: {file_path.name}")
        print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*70}\n")

        start_time = time.time()

        try:
            # Check file type
            if file_path.suffix.lower() == '.pdf':
                print("Detected PDF file - Running comprehensive comparison...")
                report_path = self.comparison_tool.compare_pdf(file_path)

                # Generate enhanced report with additional insights
                enhanced_report = self._generate_enhanced_report(file_path, report_path)

                processing_time = time.time() - start_time

                print(f"\n{'='*70}")
                print(f"[SUCCESS] ANALYSIS COMPLETE")
                print(f"Total time: {processing_time:.2f} seconds")
                print(f"Report saved: {enhanced_report}")
                print(f"{'='*70}\n")

                if auto_open:
                    webbrowser.open(f"file://{enhanced_report.absolute()}")
                    print("Opening report in browser...")

                return {
                    "success": True,
                    "file": str(file_path),
                    "report": str(enhanced_report),
                    "processing_time": processing_time
                }

            elif file_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']:
                print("Detected image file - Running OCR comparison...")
                return self._process_image(file_path, auto_open)

            else:
                print(f"[ERROR] Unsupported file type: {file_path.suffix}")
                return {"error": f"Unsupported file type: {file_path.suffix}"}

        except Exception as e:
            print(f"[ERROR] Error during processing: {e}")
            traceback.print_exc()
            return {"error": str(e)}

    def _process_image(self, image_path: Path, auto_open: bool) -> Dict:
        """Process image files with OCR engines"""
        from Biotech_Model_Test.src.local_ocr.ocr_tesseract import TesseractOCR
        from Biotech_Model_Test.src.local_ocr.ocr_easyocr import EasyOCROCR as EasyOCREngine
        from Biotech_Model_Test.src.local_ocr.ocr_deepseek import DeepSeekOCR

        results = {}

        # Initialize OCR engines
        tesseract = TesseractOCR()
        easyocr = EasyOCREngine(use_gpu=False)
        deepseek = DeepSeekOCR(use_gpu=False)

        print("Running Tesseract OCR...")
        tesseract_result = tesseract.process(image_path)
        results['Tesseract'] = tesseract_result

        print("Running EasyOCR...")
        easyocr_result = easyocr.process(image_path)
        results['EasyOCR'] = easyocr_result

        print("Running DeepSeek OCR...")
        deepseek_result = deepseek.process(image_path)
        results['DeepSeek'] = deepseek_result

        # Generate HTML report
        report_path = self._generate_image_report(image_path, results)

        if auto_open:
            webbrowser.open(f"file://{report_path.absolute()}")

        return {
            "success": True,
            "file": str(image_path),
            "report": str(report_path),
            "results": {
                "tesseract_words": tesseract_result.word_count,
                "easyocr_words": easyocr_result.word_count,
                "deepseek_words": deepseek_result.word_count
            }
        }

    def _generate_enhanced_report(self, pdf_path: Path, original_report: Path) -> Path:
        """Generate an enhanced report with additional insights"""
        # For now, return the original report
        # In the future, you could add more analysis here
        return original_report

    def _generate_image_report(self, image_path: Path, results: Dict) -> Path:
        """Generate HTML report for image OCR comparison"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = self.output_dir / f"image_ocr_{timestamp}.html"

        html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>OCR Comparison - {image_path.name}</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 2rem;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 20px;
            padding: 2rem;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
        }}
        h1 {{
            text-align: center;
            color: #333;
            margin-bottom: 2rem;
        }}
        .results-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
            gap: 2rem;
        }}
        .result-card {{
            background: #f8f9fa;
            border-radius: 12px;
            padding: 1.5rem;
            border: 2px solid #e9ecef;
        }}
        .metrics {{
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 1rem;
            margin: 1rem 0;
        }}
        .metric {{
            background: white;
            padding: 0.75rem;
            border-radius: 8px;
            text-align: center;
        }}
        .text-preview {{
            background: white;
            border: 1px solid #dee2e6;
            border-radius: 8px;
            padding: 1rem;
            max-height: 300px;
            overflow-y: auto;
            font-family: monospace;
            font-size: 0.9rem;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🔍 OCR Comparison Results</h1>
        <p style="text-align: center; color: #666;">
            File: {image_path.name} | Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        </p>
        <div class="results-grid">
"""

        for engine_name, result in results.items():
            html += f"""
            <div class="result-card">
                <h2>{engine_name}</h2>
                <div class="metrics">
                    <div class="metric">
                        <strong>Words</strong><br>{result.word_count:,}
                    </div>
                    <div class="metric">
                        <strong>Characters</strong><br>{result.char_count:,}
                    </div>
                    <div class="metric">
                        <strong>Time</strong><br>{result.processing_time:.2f}s
                    </div>
                    <div class="metric">
                        <strong>Confidence</strong><br>{result.confidence:.1%}
                    </div>
                </div>
                <h3>Extracted Text</h3>
                <div class="text-preview">{result.text[:1000]}...</div>
            </div>
"""

        html += """
        </div>
    </div>
</body>
</html>
"""

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(html)

        return report_path

    def batch_process(self, folder_path: str, file_pattern: str = "*.pdf") -> List[Dict]:
        """
        Process multiple documents in a folder

        Args:
            folder_path: Path to folder containing documents
            file_pattern: Glob pattern for files to process (default: *.pdf)

        Returns:
            List of processing results
        """
        folder = Path(folder_path)
        if not folder.exists():
            print(f"[ERROR] Folder not found: {folder}")
            return []

        files = list(folder.glob(file_pattern))
        print(f"\nFound {len(files)} files matching pattern: {file_pattern}")

        results = []
        for i, file_path in enumerate(files, 1):
            print(f"\n[{i}/{len(files)}] Processing {file_path.name}...")
            result = self.process_document(str(file_path), auto_open=False)
            results.append(result)

        # Generate summary report
        self._generate_batch_summary(results)

        return results

    def _generate_batch_summary(self, results: List[Dict]) -> Path:
        """Generate a summary report for batch processing"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        summary_path = self.output_dir / f"batch_summary_{timestamp}.html"

        successful = [r for r in results if r.get("success")]
        failed = [r for r in results if r.get("error")]

        html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Batch Processing Summary</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            padding: 2rem;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        }}
        .container {{
            max-width: 1000px;
            margin: 0 auto;
            background: white;
            border-radius: 20px;
            padding: 2rem;
        }}
        .stats {{
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 1rem;
            margin: 2rem 0;
        }}
        .stat {{
            background: #f8f9fa;
            padding: 1rem;
            border-radius: 8px;
            text-align: center;
        }}
        .success {{ color: #28a745; }}
        .failed {{ color: #dc3545; }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 2rem;
        }}
        th, td {{
            padding: 0.75rem;
            text-align: left;
            border-bottom: 1px solid #dee2e6;
        }}
        th {{
            background: #f8f9fa;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>📊 Batch Processing Summary</h1>
        <div class="stats">
            <div class="stat">
                <h3>Total Files</h3>
                <p style="font-size: 2rem;">{len(results)}</p>
            </div>
            <div class="stat">
                <h3 class="success">Successful</h3>
                <p style="font-size: 2rem;" class="success">{len(successful)}</p>
            </div>
            <div class="stat">
                <h3 class="failed">Failed</h3>
                <p style="font-size: 2rem;" class="failed">{len(failed)}</p>
            </div>
        </div>
        <table>
            <thead>
                <tr>
                    <th>File</th>
                    <th>Status</th>
                    <th>Processing Time</th>
                    <th>Report</th>
                </tr>
            </thead>
            <tbody>
"""

        for result in results:
            status = "✅ Success" if result.get("success") else f"❌ {result.get('error', 'Failed')}"
            file_name = Path(result.get("file", "Unknown")).name if result.get("file") else "Unknown"
            time_str = f"{result.get('processing_time', 0):.2f}s" if result.get('processing_time') else "N/A"
            report_link = f'<a href="{Path(result.get("report", "")).name}">View</a>' if result.get("report") else "N/A"

            html += f"""
                <tr>
                    <td>{file_name}</td>
                    <td>{status}</td>
                    <td>{time_str}</td>
                    <td>{report_link}</td>
                </tr>
"""

        html += """
            </tbody>
        </table>
    </div>
</body>
</html>
"""

        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write(html)

        print(f"\nBatch summary saved to: {summary_path}")
        webbrowser.open(f"file://{summary_path.absolute()}")

        return summary_path


def main():
    """Main execution function for the enhanced boss pipeline"""
    import argparse

    # Banner
    print("""
========================================================================

                     ENHANCED BOSS PIPELINE
                 OCR vs Text Parser Comparison Tool

========================================================================
    """)

    parser = argparse.ArgumentParser(
        description="Enhanced Boss Pipeline - Compare OCR engines with text parsers",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        "input_path",
        type=str,
        help="Path to PDF/image file or folder for batch processing"
    )

    parser.add_argument(
        "--batch",
        action="store_true",
        help="Process all PDFs in the given folder"
    )

    parser.add_argument(
        "--pattern",
        type=str,
        default="*.pdf",
        help="File pattern for batch processing (default: *.pdf)"
    )

    parser.add_argument(
        "--no-open",
        action="store_true",
        help="Don't automatically open the HTML report"
    )

    args = parser.parse_args()

    # Initialize pipeline
    pipeline = EnhancedBossPipeline()

    # Process based on mode
    if args.batch:
        # Batch processing mode
        print(f"\nStarting batch processing of folder: {args.input_path}")
        results = pipeline.batch_process(args.input_path, args.pattern)
        print(f"\n[COMPLETE] Batch processing complete! Processed {len(results)} files.")
    else:
        # Single file processing mode
        result = pipeline.process_document(
            args.input_path,
            auto_open=not args.no_open
        )

        if result.get("success"):
            print("\n[SUCCESS] Your document has been analyzed.")
            print(f"Report location: {result.get('report')}")
        else:
            print(f"\n[ERROR] Processing failed: {result.get('error')}")

    print("\nThank you for using the Enhanced Boss Pipeline!")

    return 0


if __name__ == "__main__":
    sys.exit(main())