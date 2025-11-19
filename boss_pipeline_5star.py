"""
Enhanced Boss Pipeline with 5-Star OCR Models
Tests all new models and generates beautiful HTML comparison
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

# Import base OCR framework
from Biotech_Model_Test.base_ocr import BaseOCR, OCRResult, OCRBenchmarker

# Import text parsers
from Biotech_Model_Test.src.text_parsers.parse_pypdf import PyPDFParser
from Biotech_Model_Test.src.text_parsers.parse_pdfplumber import PDFPlumberParser

# Import existing OCR engines
try:
    from Biotech_Model_Test.src.local_ocr.ocr_tesseract import TesseractOCR
except:
    TesseractOCR = None

try:
    from Biotech_Model_Test.src.local_ocr.ocr_easyocr import EasyOCROCR as EasyOCREngine
except:
    EasyOCREngine = None

try:
    from Biotech_Model_Test.src.local_ocr.ocr_paddleocr import PaddleOCROCR
except:
    PaddleOCROCR = None

try:
    from Biotech_Model_Test.src.local_ocr.ocr_deepseek import DeepSeekOCR
except:
    DeepSeekOCR = None

# Import 5-Star OCR models
from Biotech_Model_Test.src.local_ocr.ocr_trocr import TrOCR, TrOCRLarge
from Biotech_Model_Test.src.local_ocr.ocr_layoutlm import LayoutLMv3OCR, LayoutLMv3Tables
from Biotech_Model_Test.src.local_ocr.ocr_nougat import NougatOCR
from Biotech_Model_Test.src.local_ocr.ocr_got import GOTOCR
from Biotech_Model_Test.src.local_ocr.ocr_surya import SuryaOCR

# PDF to image conversion
try:
    from pdf2image import convert_from_path
    PDF2IMAGE_AVAILABLE = True
except ImportError:
    PDF2IMAGE_AVAILABLE = False
    print("Warning: pdf2image not available. Install with: pip install pdf2image")

from PIL import Image
import tempfile


class Boss5StarPipeline:
    """Enhanced pipeline with 5-star OCR models"""

    def __init__(self, use_gpu: bool = False):
        """Initialize the 5-star pipeline"""
        self.output_dir = Path("boss_5star_results")
        self.output_dir.mkdir(exist_ok=True)
        self.use_gpu = use_gpu

        # Initialize text parsers (for PDFs)
        self.pypdf_parser = PyPDFParser()
        self.pdfplumber_parser = PDFPlumberParser(
            extract_tables=True,
            extract_metadata=True
        )

        # Track results
        self.benchmarker = OCRBenchmarker()

    def process_document(self, file_path: str, models: List[str] = None, auto_open: bool = True) -> Dict:
        """
        Process a document with 5-star models

        Args:
            file_path: Path to PDF or image file
            models: List of models to use (None = use all)
            auto_open: Whether to automatically open the HTML report

        Returns:
            Dictionary with processing results
        """
        file_path = Path(file_path)

        if not file_path.exists():
            print(f"[ERROR] File not found: {file_path}")
            return {"error": "File not found"}

        print(f"\n{'='*70}")
        print(f"5-STAR OCR MODEL COMPARISON PIPELINE")
        print(f"{'='*70}")
        print(f"Document: {file_path.name}")
        print(f"GPU Mode: {'Enabled' if self.use_gpu else 'Disabled (Mock Mode)'}")
        print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*70}\n")

        start_time = time.time()
        results = {}

        # Convert PDF to image if needed
        if file_path.suffix.lower() == '.pdf':
            print(" Converting PDF to image for OCR processing...")
            image_path = self._pdf_to_image(file_path)
            if not image_path:
                return {"error": "Failed to convert PDF to image"}

            # Also run text parsers for comparison
            print("\n Running text parsers on PDF...")
            results['parsers'] = self._run_parsers(file_path)
        else:
            image_path = file_path

        # Select models to run
        if models:
            selected_models = models
        else:
            selected_models = ['trocr', 'layoutlm', 'nougat', 'got', 'surya',
                              'tesseract', 'easyocr', 'paddle', 'deepseek']

        # Run OCR models
        print("\nRunning OCR models...")
        results['ocr'] = self._run_ocr_models(image_path, selected_models)

        # Generate HTML report
        report_path = self._generate_html_report(file_path, results)

        processing_time = time.time() - start_time

        print(f"\n{'='*70}")
        print(f"[SUCCESS] ANALYSIS COMPLETE")
        print(f"Total time: {processing_time:.2f} seconds")
        print(f"Report saved: {report_path}")
        print(f"{'='*70}\n")

        # Print benchmark summary
        print(self.benchmarker.get_comparison_table())

        if auto_open:
            webbrowser.open(f"file://{report_path.absolute()}")
            print("\nOpening report in browser...")

        return {
            "success": True,
            "file": str(file_path),
            "report": str(report_path),
            "processing_time": processing_time,
            "models_run": len(results.get('ocr', {}))
        }

    def _pdf_to_image(self, pdf_path: Path) -> Optional[Path]:
        """Convert first page of PDF to image"""
        if not PDF2IMAGE_AVAILABLE:
            print("  ⚠️ pdf2image not installed, skipping conversion")
            return None

        try:
            images = convert_from_path(pdf_path, first_page=1, last_page=1)
            if images:
                # Save to temp file
                temp_dir = self.output_dir / "temp"
                temp_dir.mkdir(exist_ok=True)
                image_path = temp_dir / f"{pdf_path.stem}_page1.png"
                images[0].save(image_path, "PNG")
                print(f"   Converted to: {image_path.name}")
                return image_path
        except Exception as e:
            print(f"   Conversion failed: {e}")
        return None

    def _run_parsers(self, pdf_path: Path) -> Dict[str, OCRResult]:
        """Run text parsers on PDF"""
        results = {}

        # PyPDF2
        try:
            print("   PyPDF2...")
            result = self.pypdf_parser.process(pdf_path)
            results['PyPDF2'] = result
            self.benchmarker.add_result(result)
            print(f"     - {result.word_count} words extracted")
        except Exception as e:
            print(f"     - Failed: {e}")

        # PDFPlumber
        try:
            print("   PDFPlumber...")
            result = self.pdfplumber_parser.process(pdf_path)
            results['PDFPlumber'] = result
            self.benchmarker.add_result(result)
            print(f"     - {result.word_count} words extracted")
        except Exception as e:
            print(f"     - Failed: {e}")

        return results

    def _run_ocr_models(self, image_path: Path, selected_models: List[str]) -> Dict[str, OCRResult]:
        """Run selected OCR models"""
        results = {}

        # 5-Star Models
        models = {
            'trocr': (TrOCR, " TrOCR (Microsoft)", {"use_gpu": self.use_gpu}),
            'trocr_large': (TrOCRLarge, " TrOCR-Large", {"use_gpu": self.use_gpu}),
            'layoutlm': (LayoutLMv3OCR, " LayoutLMv3", {"use_gpu": self.use_gpu}),
            'layoutlm_tables': (LayoutLMv3Tables, " LayoutLMv3-Tables", {"use_gpu": self.use_gpu}),
            'nougat': (NougatOCR, " Nougat (Meta)", {"use_gpu": self.use_gpu}),
            'got': (GOTOCR, " GOT-OCR2.0", {"use_gpu": self.use_gpu}),
            'surya': (SuryaOCR, " Surya", {"use_gpu": self.use_gpu}),
        }

        # Add existing models if available
        if TesseractOCR:
            models['tesseract'] = (TesseractOCR, "Tesseract", {})
        if EasyOCREngine:
            models['easyocr'] = (EasyOCREngine, "EasyOCR", {"use_gpu": self.use_gpu})
        if PaddleOCROCR:
            models['paddle'] = (PaddleOCROCR, "PaddleOCR", {"use_gpu": self.use_gpu})
        if DeepSeekOCR:
            models['deepseek'] = (DeepSeekOCR, "DeepSeek", {"use_gpu": self.use_gpu})

        # Run selected models
        for model_key, (model_class, display_name, kwargs) in models.items():
            # Check if model should be run
            should_run = any(key in model_key for key in selected_models)
            if not should_run:
                continue

            try:
                print(f"  {display_name}...")
                model = model_class(**kwargs)
                result = model.process(image_path)
                results[display_name] = result
                self.benchmarker.add_result(result)

                if result.errors:
                    print(f"      Errors: {result.errors[0][:50]}")
                else:
                    print(f"      {result.word_count} words in {result.processing_time:.2f}s")

            except Exception as e:
                print(f"      Failed: {str(e)[:50]}")
                results[display_name] = OCRResult(
                    model_name=display_name,
                    errors=[str(e)],
                    processing_time=0.0
                )

        return results

    def _generate_html_report(self, file_path: Path, results: Dict) -> Path:
        """Generate beautiful HTML report"""

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = self.output_dir / f"5star_comparison_{timestamp}.html"

        # Prepare data
        ocr_results = results.get('ocr', {})
        parser_results = results.get('parsers', {})
        all_results = {**parser_results, **ocr_results}

        # Find best performer
        best_performer = None
        best_time = float('inf')
        for name, result in all_results.items():
            if not result.errors and result.processing_time < best_time:
                best_time = result.processing_time
                best_performer = name

        # Generate HTML
        html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>5-Star OCR Comparison - {file_path.name}</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}

        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 2rem;
        }}

        .container {{
            max-width: 1600px;
            margin: 0 auto;
            background: white;
            border-radius: 20px;
            padding: 2rem;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
        }}

        h1 {{
            color: #333;
            text-align: center;
            margin-bottom: 0.5rem;
            font-size: 2.5rem;
        }}

        .subtitle {{
            text-align: center;
            color: #666;
            margin-bottom: 2rem;
            font-size: 1.1rem;
        }}

        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1rem;
            margin-bottom: 2rem;
            padding: 1rem;
            background: #f8f9fa;
            border-radius: 12px;
        }}

        .stat-card {{
            background: white;
            padding: 1rem;
            border-radius: 8px;
            text-align: center;
            border: 2px solid #e9ecef;
        }}

        .stat-value {{
            font-size: 2rem;
            font-weight: bold;
            color: #667eea;
        }}

        .stat-label {{
            color: #666;
            font-size: 0.9rem;
            margin-top: 0.25rem;
        }}

        .models-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(350px, 1fr));
            gap: 1.5rem;
            margin-bottom: 2rem;
        }}

        .model-card {{
            background: white;
            border-radius: 12px;
            padding: 1.5rem;
            border: 2px solid #e9ecef;
            transition: all 0.3s;
        }}

        .model-card:hover {{
            transform: translateY(-4px);
            box-shadow: 0 8px 20px rgba(0,0,0,0.1);
            border-color: #667eea;
        }}

        .model-card.five-star {{
            background: linear-gradient(135deg, #fff 0%, #f8f9ff 100%);
            border-color: #ffd700;
        }}

        .model-card.best {{
            border-color: #28a745;
            background: linear-gradient(135deg, #fff 0%, #f0fff4 100%);
        }}

        .model-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 1rem;
        }}

        .model-name {{
            font-size: 1.3rem;
            font-weight: bold;
            color: #333;
        }}

        .model-badge {{
            padding: 0.25rem 0.75rem;
            border-radius: 20px;
            font-size: 0.8rem;
            font-weight: bold;
        }}

        .badge-star {{
            background: linear-gradient(135deg, #ffd700 0%, #ffed4e 100%);
            color: #333;
        }}

        .badge-success {{
            background: #28a745;
            color: white;
        }}

        .badge-error {{
            background: #dc3545;
            color: white;
        }}

        .metrics {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 0.5rem;
            margin: 1rem 0;
        }}

        .metric {{
            background: #f8f9fa;
            padding: 0.5rem;
            border-radius: 6px;
            display: flex;
            justify-content: space-between;
        }}

        .metric-label {{
            color: #666;
            font-size: 0.85rem;
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
            max-height: 150px;
            overflow-y: auto;
            font-family: 'Monaco', 'Courier New', monospace;
            font-size: 0.85rem;
            line-height: 1.5;
        }}

        .legend {{
            display: flex;
            justify-content: center;
            gap: 2rem;
            margin: 2rem 0;
            padding: 1rem;
            background: #f8f9fa;
            border-radius: 8px;
        }}

        .legend-item {{
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }}

        .legend-color {{
            width: 20px;
            height: 20px;
            border-radius: 4px;
            border: 2px solid;
        }}

        .performance-chart {{
            background: white;
            padding: 2rem;
            border-radius: 12px;
            border: 2px solid #e9ecef;
            margin-bottom: 2rem;
        }}

        .chart-title {{
            font-size: 1.5rem;
            font-weight: bold;
            color: #333;
            margin-bottom: 1rem;
            text-align: center;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1> 5-Star OCR Model Comparison </h1>
        <p class="subtitle">Document: {file_path.name} | Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>

        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-value">{len(all_results)}</div>
                <div class="stat-label">Models Tested</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{len([r for r in ocr_results.values() if '' in r.model_name])}</div>
                <div class="stat-label">5-Star Models</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{best_performer or 'N/A'}</div>
                <div class="stat-label">Fastest Model</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{'GPU' if self.use_gpu else 'CPU/Mock'}</div>
                <div class="stat-label">Processing Mode</div>
            </div>
        </div>

        <div class="legend">
            <div class="legend-item">
                <div class="legend-color" style="border-color: #ffd700; background: #fffacd;"></div>
                <span>5-Star Model</span>
            </div>
            <div class="legend-item">
                <div class="legend-color" style="border-color: #28a745; background: #d4edda;"></div>
                <span>Best Performer</span>
            </div>
            <div class="legend-item">
                <div class="legend-color" style="border-color: #e9ecef; background: white;"></div>
                <span>Standard Model</span>
            </div>
        </div>

        <div class="models-grid">
"""

        # Add model cards
        for name, result in all_results.items():
            is_five_star = '' in name
            is_best = name == best_performer
            has_error = bool(result.errors)

            card_class = "model-card"
            if is_five_star:
                card_class += " five-star"
            if is_best:
                card_class += " best"

            badge_html = ""
            if is_five_star:
                badge_html = '<span class="model-badge badge-star"> 5-STAR</span>'
            elif has_error:
                badge_html = '<span class="model-badge badge-error">ERROR</span>'
            elif is_best:
                badge_html = '<span class="model-badge badge-success">FASTEST</span>'

            text_preview = result.text[:200].replace('\n', ' ') if result.text else "No text extracted"
            if len(text_preview) > 200:
                text_preview = text_preview[:197] + "..."

            html += f"""
            <div class="{card_class}">
                <div class="model-header">
                    <div class="model-name">{name}</div>
                    {badge_html}
                </div>

                <div class="metrics">
                    <div class="metric">
                        <span class="metric-label">Time</span>
                        <span class="metric-value">{result.processing_time:.3f}s</span>
                    </div>
                    <div class="metric">
                        <span class="metric-label">Words</span>
                        <span class="metric-value">{result.word_count:,}</span>
                    </div>
                    <div class="metric">
                        <span class="metric-label">Chars</span>
                        <span class="metric-value">{result.char_count:,}</span>
                    </div>
                    <div class="metric">
                        <span class="metric-label">Memory</span>
                        <span class="metric-value">{result.memory_usage_mb:.1f}MB</span>
                    </div>
                </div>

                <div class="text-preview">
                    {text_preview if not has_error else f"Error: {result.errors[0][:100]}"}
                </div>
            </div>
"""

        html += """
        </div>
    </div>
</body>
</html>
"""

        # Save HTML
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(html)

        return report_path


def main():
    """Main function with CLI"""
    import argparse

    parser = argparse.ArgumentParser(description="5-Star OCR Model Pipeline")
    parser.add_argument("file", help="Path to PDF or image file")
    parser.add_argument("--gpu", action="store_true", help="Use GPU (if available)")
    parser.add_argument("--models", nargs="+", help="Specific models to run")
    parser.add_argument("--no-browser", action="store_true", help="Don't open browser")

    args = parser.parse_args()

    # Run pipeline
    pipeline = Boss5StarPipeline(use_gpu=args.gpu)
    result = pipeline.process_document(
        args.file,
        models=args.models,
        auto_open=not args.no_browser
    )

    # Exit with appropriate code
    sys.exit(0 if result.get("success") else 1)


if __name__ == "__main__":
    main()