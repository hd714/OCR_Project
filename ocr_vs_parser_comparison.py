"""
OCR vs Text Parser Comparison Pipeline
Compares OCR engines (Tesseract, EasyOCR) with text parsers (PyPDF2, PDFPlumber)
Generates comprehensive HTML dashboard with side-by-side comparisons
"""

import sys
import os
from pathlib import Path
from datetime import datetime
import time
import json
import difflib
import webbrowser
from typing import Dict, List, Tuple, Optional
import traceback

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "Biotech_Model_Test"))
sys.path.insert(0, str(Path(__file__).parent / "Biotech_Model_Test" / "src"))

# Import base OCR framework
from Biotech_Model_Test.base_ocr import BaseOCR, OCRResult

# Import text parsers
from Biotech_Model_Test.src.text_parsers.parse_pypdf import PyPDFParser
from Biotech_Model_Test.src.text_parsers.parse_pdfplumber import PDFPlumberParser

# Import OCR engines
from Biotech_Model_Test.src.local_ocr.ocr_tesseract import TesseractOCR
from Biotech_Model_Test.src.local_ocr.ocr_easyocr import EasyOCROCR as EasyOCREngine

# PDF to image conversion
try:
    from pdf2image import convert_from_path
    PDF2IMAGE_AVAILABLE = True
except ImportError:
    PDF2IMAGE_AVAILABLE = False
    print("Warning: pdf2image not available. Install with: pip install pdf2image")

from PIL import Image
import numpy as np

class OCRvsParserComparison:
    """Compare OCR engines against text parsers for PDF analysis"""

    def __init__(self, output_dir: str = "comparison_results"):
        """Initialize comparison pipeline"""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # Initialize text parsers
        self.pypdf_parser = PyPDFParser()
        self.pdfplumber_parser = PDFPlumberParser(
            extract_tables=True,
            extract_metadata=True
        )

        # Initialize OCR engines
        self.tesseract_ocr = TesseractOCR()
        self.easyocr_engine = EasyOCREngine(use_gpu=False)

        # Results storage
        self.results = {}

    def extract_with_parsers(self, pdf_path: Path) -> Dict[str, OCRResult]:
        """Extract text using PDF parsers"""
        results = {}

        # PyPDF2 extraction
        print("Extracting with PyPDF2...")
        try:
            pypdf_result = self.pypdf_parser.process(pdf_path)
            results['PyPDF2'] = pypdf_result
            print(f"  - PyPDF2: {pypdf_result.word_count} words extracted")
        except Exception as e:
            print(f"  - PyPDF2 failed: {e}")
            results['PyPDF2'] = OCRResult(
                text="",
                confidence=0,
                processing_time=0,
                word_count=0,
                char_count=0,
                model_name="PyPDF2",
                errors=[str(e)]
            )

        # PDFPlumber extraction
        print("Extracting with PDFPlumber...")
        try:
            plumber_result = self.pdfplumber_parser.process(pdf_path)
            results['PDFPlumber'] = plumber_result
            print(f"  - PDFPlumber: {plumber_result.word_count} words extracted")

            # Extract table info if available
            if plumber_result.metadata and 'tables_extracted' in plumber_result.metadata:
                tables = plumber_result.metadata.get('tables_extracted', [])
                if tables:
                    print(f"  - PDFPlumber found {len(tables)} tables")
        except Exception as e:
            print(f"  - PDFPlumber failed: {e}")
            results['PDFPlumber'] = OCRResult(
                text="",
                confidence=0,
                processing_time=0,
                word_count=0,
                char_count=0,
                model_name="PDFPlumber",
                errors=[str(e)]
            )

        return results

    def extract_with_ocr(self, pdf_path: Path) -> Dict[str, OCRResult]:
        """Extract text using OCR engines"""
        results = {}

        if not PDF2IMAGE_AVAILABLE:
            print("Warning: pdf2image not available, skipping OCR extraction")
            return results

        print("Converting PDF to images...")
        try:
            # Convert PDF to images
            images = convert_from_path(str(pdf_path))
            print(f"  - Converted to {len(images)} images")

            # Combine all pages into one temporary image for OCR
            # (In production, you might want to process each page separately)
            if images:
                # Save first page for OCR (for demo purposes)
                # In production, you'd process all pages
                temp_image_path = self.output_dir / "temp_page.png"
                images[0].save(temp_image_path)

                # Tesseract OCR
                print("Running Tesseract OCR...")
                try:
                    tesseract_result = self.tesseract_ocr.process(temp_image_path)
                    results['Tesseract'] = tesseract_result
                    print(f"  - Tesseract: {tesseract_result.word_count} words extracted")
                except Exception as e:
                    print(f"  - Tesseract failed: {e}")
                    results['Tesseract'] = OCRResult(
                        text="",
                        confidence=0,
                        processing_time=0,
                        word_count=0,
                        char_count=0,
                        model_name="Tesseract",
                        errors=[str(e)]
                    )

                # EasyOCR
                print("Running EasyOCR...")
                try:
                    easyocr_result = self.easyocr_engine.process(temp_image_path)
                    results['EasyOCR'] = easyocr_result
                    print(f"  - EasyOCR: {easyocr_result.word_count} words extracted")
                except Exception as e:
                    print(f"  - EasyOCR failed: {e}")
                    results['EasyOCR'] = OCRResult(
                        text="",
                        confidence=0,
                        processing_time=0,
                        word_count=0,
                        char_count=0,
                        model_name="EasyOCR",
                        errors=[str(e)]
                    )

                # Clean up temp file
                if temp_image_path.exists():
                    temp_image_path.unlink()

        except Exception as e:
            print(f"Failed to convert PDF to images: {e}")

        return results

    def calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two texts"""
        if not text1 or not text2:
            return 0.0

        # Use sequence matcher for similarity
        return difflib.SequenceMatcher(None, text1, text2).ratio()

    def generate_html_report(self,
                            pdf_path: Path,
                            parser_results: Dict[str, OCRResult],
                            ocr_results: Dict[str, OCRResult]) -> Path:
        """Generate comprehensive HTML comparison report"""

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = self.output_dir / f"comparison_{timestamp}.html"

        # Calculate similarities between parsers and OCR
        similarities = {}
        for parser_name, parser_result in parser_results.items():
            for ocr_name, ocr_result in ocr_results.items():
                key = f"{parser_name}_vs_{ocr_name}"
                similarities[key] = self.calculate_similarity(
                    parser_result.text[:1000],  # Compare first 1000 chars
                    ocr_result.text[:1000]
                )

        # Generate HTML
        html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>OCR vs Text Parser Comparison - {pdf_path.name}</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}

        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 2rem;
        }}

        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            border-radius: 20px;
            padding: 2rem;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
        }}

        h1 {{
            color: #333;
            text-align: center;
            margin-bottom: 1rem;
            font-size: 2.5rem;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
        }}

        .subtitle {{
            text-align: center;
            color: #666;
            margin-bottom: 2rem;
            font-size: 1.1rem;
        }}

        .comparison-grid {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 2rem;
            margin-bottom: 2rem;
        }}

        .method-section {{
            background: #f8f9fa;
            border-radius: 12px;
            padding: 1.5rem;
            border: 2px solid #e9ecef;
        }}

        .method-section h2 {{
            color: #495057;
            margin-bottom: 1rem;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }}

        .parser-section h2 {{
            color: #0066cc;
        }}

        .ocr-section h2 {{
            color: #ff6b6b;
        }}

        .method-card {{
            background: white;
            border-radius: 8px;
            padding: 1rem;
            margin-bottom: 1rem;
            border: 1px solid #dee2e6;
            transition: transform 0.2s;
        }}

        .method-card:hover {{
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        }}

        .method-name {{
            font-weight: bold;
            color: #333;
            font-size: 1.2rem;
            margin-bottom: 0.5rem;
        }}

        .metrics {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 0.5rem;
            margin: 0.5rem 0;
        }}

        .metric {{
            background: #f1f3f5;
            padding: 0.5rem;
            border-radius: 6px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}

        .metric-label {{
            font-size: 0.85rem;
            color: #666;
        }}

        .metric-value {{
            font-weight: bold;
            color: #333;
        }}

        .text-preview {{
            background: #f8f9fa;
            border: 1px solid #dee2e6;
            border-radius: 6px;
            padding: 1rem;
            margin-top: 1rem;
            max-height: 200px;
            overflow-y: auto;
            font-family: 'Monaco', 'Courier New', monospace;
            font-size: 0.85rem;
            line-height: 1.5;
            white-space: pre-wrap;
            word-break: break-word;
        }}

        .similarity-matrix {{
            margin: 2rem 0;
            background: white;
            border-radius: 12px;
            padding: 1.5rem;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }}

        .similarity-matrix h2 {{
            color: #333;
            margin-bottom: 1rem;
            text-align: center;
        }}

        .similarity-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 1rem;
        }}

        .similarity-card {{
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            border-radius: 8px;
            padding: 1rem;
            text-align: center;
        }}

        .similarity-comparison {{
            font-size: 0.9rem;
            color: #666;
            margin-bottom: 0.5rem;
        }}

        .similarity-score {{
            font-size: 2rem;
            font-weight: bold;
            color: #333;
        }}

        .high-similarity {{
            color: #28a745;
        }}

        .medium-similarity {{
            color: #ffc107;
        }}

        .low-similarity {{
            color: #dc3545;
        }}

        .summary {{
            background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
            border-radius: 12px;
            padding: 1.5rem;
            margin-top: 2rem;
            text-align: center;
        }}

        .summary h2 {{
            color: #333;
            margin-bottom: 1rem;
        }}

        .best-method {{
            background: white;
            border-radius: 8px;
            padding: 1rem;
            margin-top: 1rem;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }}

        .error {{
            background: #fee;
            color: #c00;
            padding: 0.5rem;
            border-radius: 4px;
            margin-top: 0.5rem;
        }}

        .success-badge {{
            background: #d4edda;
            color: #155724;
            padding: 0.25rem 0.5rem;
            border-radius: 4px;
            font-size: 0.85rem;
            display: inline-block;
        }}

        .warning-badge {{
            background: #fff3cd;
            color: #856404;
            padding: 0.25rem 0.5rem;
            border-radius: 4px;
            font-size: 0.85rem;
            display: inline-block;
        }}

        @media (max-width: 768px) {{
            .comparison-grid {{
                grid-template-columns: 1fr;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>📊 OCR vs Text Parser Comparison</h1>
        <p class="subtitle">
            <strong>File:</strong> {pdf_path.name} |
            <strong>Date:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        </p>

        <div class="comparison-grid">
            <!-- Text Parsers Section -->
            <div class="method-section parser-section">
                <h2>📄 Text Parsers (Direct Extraction)</h2>
                {self._generate_method_cards(parser_results, is_parser=True)}
            </div>

            <!-- OCR Engines Section -->
            <div class="method-section ocr-section">
                <h2>👁️ OCR Engines (Image-based)</h2>
                {self._generate_method_cards(ocr_results, is_parser=False)}
            </div>
        </div>

        <!-- Similarity Matrix -->
        <div class="similarity-matrix">
            <h2>🎯 Similarity Analysis</h2>
            <div class="similarity-grid">
                {self._generate_similarity_cards(similarities)}
            </div>
        </div>

        <!-- Summary and Recommendations -->
        <div class="summary">
            <h2>📈 Summary & Recommendations</h2>
            {self._generate_summary(parser_results, ocr_results, similarities)}
        </div>
    </div>
</body>
</html>
"""

        # Write HTML file
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(html)

        return report_path

    def _generate_method_cards(self, results: Dict[str, OCRResult], is_parser: bool) -> str:
        """Generate HTML cards for each method"""
        cards_html = ""

        for method_name, result in results.items():
            status_badge = ""
            if result.errors:
                status_badge = '<span class="warning-badge">⚠️ Error</span>'
            elif result.word_count > 0:
                status_badge = '<span class="success-badge">✅ Success</span>'
            else:
                status_badge = '<span class="warning-badge">⚠️ No text</span>'

            # Text preview (first 500 chars)
            text_preview = result.text[:500] + "..." if len(result.text) > 500 else result.text

            cards_html += f"""
            <div class="method-card">
                <div class="method-name">{method_name} {status_badge}</div>
                <div class="metrics">
                    <div class="metric">
                        <span class="metric-label">Words:</span>
                        <span class="metric-value">{result.word_count:,}</span>
                    </div>
                    <div class="metric">
                        <span class="metric-label">Characters:</span>
                        <span class="metric-value">{result.char_count:,}</span>
                    </div>
                    <div class="metric">
                        <span class="metric-label">Time:</span>
                        <span class="metric-value">{result.processing_time:.2f}s</span>
                    </div>
                    <div class="metric">
                        <span class="metric-label">Confidence:</span>
                        <span class="metric-value">{result.confidence:.1%}</span>
                    </div>
                </div>
                {"<div class='error'>" + ", ".join(result.errors) + "</div>" if result.errors else ""}
                <div class="text-preview">{text_preview}</div>
            </div>
            """

        return cards_html

    def _generate_similarity_cards(self, similarities: Dict[str, float]) -> str:
        """Generate similarity comparison cards"""
        cards_html = ""

        for comparison, score in similarities.items():
            # Determine color based on score
            if score > 0.8:
                color_class = "high-similarity"
            elif score > 0.5:
                color_class = "medium-similarity"
            else:
                color_class = "low-similarity"

            methods = comparison.replace("_vs_", " vs ")

            cards_html += f"""
            <div class="similarity-card">
                <div class="similarity-comparison">{methods}</div>
                <div class="similarity-score {color_class}">{score:.1%}</div>
            </div>
            """

        return cards_html

    def _generate_summary(self,
                         parser_results: Dict[str, OCRResult],
                         ocr_results: Dict[str, OCRResult],
                         similarities: Dict[str, float]) -> str:
        """Generate summary and recommendations"""

        # Find best methods
        all_results = {**parser_results, **ocr_results}
        best_by_words = max(all_results.items(), key=lambda x: x[1].word_count)
        fastest = min(all_results.items(), key=lambda x: x[1].processing_time)

        # Calculate averages
        parser_avg_words = sum(r.word_count for r in parser_results.values()) / len(parser_results) if parser_results else 0
        ocr_avg_words = sum(r.word_count for r in ocr_results.values()) / len(ocr_results) if ocr_results else 0

        summary_html = f"""
        <div class="best-method">
            <h3>🏆 Best Methods</h3>
            <p><strong>Most Text Extracted:</strong> {best_by_words[0]} ({best_by_words[1].word_count:,} words)</p>
            <p><strong>Fastest Processing:</strong> {fastest[0]} ({fastest[1].processing_time:.3f} seconds)</p>

            <h3 style="margin-top: 1rem;">📊 Analysis</h3>
            <p><strong>Text Parsers Average:</strong> {parser_avg_words:.0f} words</p>
            <p><strong>OCR Engines Average:</strong> {ocr_avg_words:.0f} words</p>

            <h3 style="margin-top: 1rem;">💡 Recommendations</h3>
        """

        # Generate recommendations
        if parser_avg_words > ocr_avg_words * 2:
            summary_html += "<p>✅ <strong>Use Text Parsers:</strong> This PDF contains native text. Text parsers are extracting significantly more content.</p>"
        elif ocr_avg_words > parser_avg_words * 2:
            summary_html += "<p>✅ <strong>Use OCR:</strong> This appears to be a scanned document. OCR engines are extracting more content.</p>"
        else:
            summary_html += "<p>✅ <strong>Hybrid Approach:</strong> Consider combining both methods for comprehensive extraction.</p>"

        summary_html += "</div>"

        return summary_html

    def compare_pdf(self, pdf_path: Path) -> Path:
        """Main comparison pipeline"""
        print(f"\n{'='*60}")
        print(f"Comparing extraction methods for: {pdf_path.name}")
        print(f"{'='*60}\n")

        # Run text parsers
        print("Stage 1: Text Parser Extraction")
        parser_results = self.extract_with_parsers(pdf_path)

        # Run OCR engines
        print("\nStage 2: OCR Engine Extraction")
        ocr_results = self.extract_with_ocr(pdf_path)

        # Generate HTML report
        print("\nStage 3: Generating HTML Report")
        report_path = self.generate_html_report(pdf_path, parser_results, ocr_results)

        print(f"\n[SUCCESS] Comparison complete!")
        print(f"Report saved to: {report_path}")

        return report_path


def main():
    """Main execution function"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Compare OCR engines with text parsers for PDF extraction"
    )
    parser.add_argument(
        "pdf_path",
        type=str,
        help="Path to PDF file to analyze"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="comparison_results",
        help="Directory to save results (default: comparison_results)"
    )
    parser.add_argument(
        "--open",
        action="store_true",
        help="Open HTML report in browser after generation"
    )

    args = parser.parse_args()

    # Validate PDF path
    pdf_path = Path(args.pdf_path)
    if not pdf_path.exists():
        print(f"Error: PDF file not found: {pdf_path}")
        sys.exit(1)

    if pdf_path.suffix.lower() != '.pdf':
        print(f"Error: File must be a PDF: {pdf_path}")
        sys.exit(1)

    # Run comparison
    comparison = OCRvsParserComparison(output_dir=args.output_dir)
    report_path = comparison.compare_pdf(pdf_path)

    # Open in browser if requested
    if args.open:
        webbrowser.open(f"file://{report_path.absolute()}")
        print(f"📂 Opening report in browser...")

    return 0


if __name__ == "__main__":
    sys.exit(main())