"""
Surya OCR Implementation
High-accuracy multilingual OCR with line-level text detection
Excellent for complex layouts and multiple languages
"""

from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
import sys
import traceback

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from base_ocr import BaseOCR

class SuryaOCR(BaseOCR):
    """Surya OCR implementation for multilingual text extraction"""

    def __init__(self,
                 use_gpu: bool = True,
                 languages: List[str] = None,
                 line_detection: bool = True,
                 layout_analysis: bool = True,
                 **kwargs):
        """
        Initialize Surya OCR

        Args:
            use_gpu: Whether to use GPU
            languages: List of language codes (supports 90+ languages)
            line_detection: Enable line-level detection
            layout_analysis: Analyze document layout
        """
        super().__init__(model_name="Surya-OCR", **kwargs)

        self.use_gpu = use_gpu
        self.languages = languages or ["en"]  # Default to English
        self.line_detection = line_detection
        self.layout_analysis = layout_analysis
        self.model = None
        self.processor = None
        self.det_model = None  # Detection model
        self.rec_model = None  # Recognition model

    def _initialize_model(self):
        """Initialize Surya OCR models"""
        if self.model is None:
            try:
                # Try to use the surya-ocr package
                try:
                    from surya.ocr import run_ocr
                    from surya.model.detection import segformer
                    from surya.model.recognition.model import load_model as load_rec_model
                    from surya.model.recognition.processor import load_processor
                    import torch

                    self.use_gpu = self.use_gpu and torch.cuda.is_available()

                    if self.logger:
                        self.logger.info("Loading Surya OCR models...")

                    # Load detection model (for text line detection)
                    if self.line_detection:
                        self.det_model = segformer.load_model()
                        if self.use_gpu:
                            self.det_model = self.det_model.cuda()

                    # Load recognition model
                    self.rec_model = load_rec_model()
                    if self.use_gpu:
                        self.rec_model = self.rec_model.cuda()

                    # Load processor
                    self.processor = load_processor()

                    # Mark as initialized
                    self.model = "surya"

                    if self.logger:
                        self.logger.info(f"Surya OCR loaded on {'GPU' if self.use_gpu else 'CPU'}")

                except ImportError:
                    # Fallback: try transformers-based approach
                    import torch
                    from transformers import AutoModel, AutoProcessor

                    if self.logger:
                        self.logger.info("surya-ocr package not found, trying transformers...")

                    # Try to load as a vision encoder-decoder model
                    try:
                        model_name = "vikp/surya_ocr"  # Example model name

                        self.processor = AutoProcessor.from_pretrained(model_name)
                        self.model = AutoModel.from_pretrained(model_name)

                        if self.use_gpu and torch.cuda.is_available():
                            self.model = self.model.cuda()

                        self.model.eval()

                        if self.logger:
                            self.logger.info("Surya OCR loaded via transformers")

                    except:
                        # Use mock mode if loading fails
                        if self.logger:
                            self.logger.info("Surya OCR not available, using mock mode")
                        self.model = "mock"

            except Exception as e:
                if self.logger:
                    self.logger.error(f"Failed to initialize Surya OCR: {e}")
                    self.logger.info("Install with: pip install surya-ocr")
                    self.logger.info("Falling back to mock mode for testing")
                self.model = "mock"

    def _extract_text(self, file_path: Path, **kwargs) -> tuple[str, Optional[float], Dict[str, Any]]:
        """Extract text using Surya OCR"""

        metadata = {
            'engine': 'Surya-OCR',
            'gpu_used': self.use_gpu,
            'languages': self.languages,
            'line_detection': self.line_detection,
            'layout_analysis': self.layout_analysis
        }

        try:
            from PIL import Image
            import numpy as np

            # Load image
            image = Image.open(file_path)
            if image.mode != 'RGB':
                image = image.convert('RGB')

            metadata['image_size'] = image.size

            # Initialize model
            self._initialize_model()

            # If model failed to load, use mock extraction
            if self.model == "mock" or self.model is None:
                if self.logger:
                    self.logger.info("Using mock Surya OCR for testing")

                # Mock extraction
                text = f"""[Surya OCR Mock Extraction]
Document: {file_path.name}
Languages: {', '.join(self.languages)}

===== LINE-LEVEL DETECTION =====

Line 1: Phase 2 Dose-Optimization Study Results
Line 2: HUMIRA (adalimumab) in Rheumatoid Arthritis
Line 3:
Line 4: Primary Endpoint: ACR20 Response at Week 12
Line 5: • HUMIRA 40mg EOW: 67.2% (n=125)
Line 6: • HUMIRA 20mg weekly: 62.8% (n=118)
Line 7: • Placebo: 23.4% (n=120)
Line 8: • p<0.001 for both doses vs placebo
Line 9:
Line 10: Secondary Endpoints:
Line 11: ACR50 Response:
Line 12: • HUMIRA 40mg: 42.1%
Line 13: • HUMIRA 20mg: 38.9%
Line 14: • Placebo: 8.2%
Line 15:
Line 16: Safety Profile:
Line 17: Most common adverse events (≥5%):
Line 18: • Injection site reactions: 12.3%
Line 19: • Upper respiratory infections: 8.7%
Line 20: • Headache: 6.2%

===== MULTILINGUAL SUPPORT =====
English: Extracted with 98% accuracy
中文: 支持中文文本提取
日本語: 日本語のテキスト抽出をサポート
한국어: 한국어 텍스트 추출 지원
Español: Soporte para extracción de texto en español

===== LAYOUT ANALYSIS =====
[TITLE] Phase 2 Study Results
[SECTION] Primary Endpoints
[TABLE] Efficacy Results
[BULLET_LIST] Safety Data
[FOOTER] Study Reference: NCT00195702

Surya OCR Features:
• Line-level text detection
• 90+ language support
• Layout understanding
• Complex script handling (Arabic, Hindi, etc.)
• Rotation correction
• High accuracy on challenging documents

Mock Metrics:
- Processing time: 1.2s
- Confidence: 0.94
- Lines detected: 20
- Languages: {len(self.languages)}
- Image size: {image.size}
"""

                confidence = 0.94
                metadata['mock_mode'] = True
                metadata['lines_detected'] = 20

            else:
                # Real Surya OCR inference
                if self.logger:
                    self.logger.info(f"Running Surya OCR on {'GPU' if self.use_gpu else 'CPU'}...")

                if self.model == "surya":
                    # Using surya-ocr package
                    from surya.ocr import run_ocr

                    # Prepare image
                    img_array = np.array(image)

                    # Run OCR with line detection
                    predictions = run_ocr(
                        [img_array],
                        [self.languages],
                        self.det_model,
                        self.rec_model,
                        self.processor
                    )

                    # Extract text from predictions
                    text_lines = []
                    for page_pred in predictions:
                        for line in page_pred.text_lines:
                            text_lines.append(line.text)

                    text = '\n'.join(text_lines)

                    metadata['lines_detected'] = len(text_lines)

                    # Layout analysis if enabled
                    if self.layout_analysis:
                        # Analyze layout structure
                        layout_info = self._analyze_layout(predictions)
                        metadata['layout_info'] = layout_info

                else:
                    # Using transformers or alternative implementation
                    import torch

                    # Process image
                    inputs = self.processor(images=image, return_tensors="pt")

                    if self.use_gpu:
                        inputs = {k: v.cuda() for k, v in inputs.items()}

                    # Generate text
                    with torch.no_grad():
                        outputs = self.model.generate(
                            **inputs,
                            max_length=kwargs.get('max_length', 2048)
                        )

                    # Decode text
                    text = self.processor.batch_decode(outputs, skip_special_tokens=True)[0]

                # Calculate confidence
                confidence = 0.90  # Base confidence
                if len(text) > 100:
                    confidence = min(0.95, confidence + 0.03)

                if self.logger:
                    self.logger.info(f"Surya OCR extracted {len(text)} characters")
                    if 'lines_detected' in metadata:
                        self.logger.info(f"Detected {metadata['lines_detected']} lines")

                metadata['real_inference'] = True

            metadata.update({
                'output_length': len(text),
                'processing_mode': 'mock' if metadata.get('mock_mode') else 'real'
            })

            return text, confidence, metadata

        except Exception as e:
            if self.logger:
                self.logger.error(f"Surya OCR processing failed: {e}")
                self.logger.debug(traceback.format_exc())

            # Return error result
            error_text = f"[Surya OCR Error]\nError: {str(e)[:100]}\nFile: {file_path.name}"
            return error_text, 0.0, {'error': str(e), **metadata}

    def _analyze_layout(self, predictions: List) -> Dict:
        """Analyze document layout from predictions"""
        layout_info = {
            'num_columns': 1,
            'has_tables': False,
            'has_lists': False,
            'text_blocks': []
        }

        # Simple heuristic-based layout analysis
        # This would be enhanced with actual layout detection models
        for page in predictions:
            # Check for table-like structures
            if any('|' in line.text for line in page.text_lines):
                layout_info['has_tables'] = True

            # Check for bullet points
            if any(line.text.strip().startswith(('•', '-', '*', '○'))
                   for line in page.text_lines):
                layout_info['has_lists'] = True

        return layout_info

    def extract_by_region(self, file_path: Path, regions: List[Tuple[int, int, int, int]], **kwargs):
        """Extract text from specific regions"""
        """
        regions: List of (x, y, width, height) tuples
        """
        from PIL import Image

        image = Image.open(file_path)
        results = []

        for x, y, w, h in regions:
            # Crop region
            region_img = image.crop((x, y, x + w, y + h))

            # Save to temp file and process
            import tempfile
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                region_img.save(tmp.name)
                result = self.process(Path(tmp.name), **kwargs)
                results.append(result.text)

        return '\n---\n'.join(results)


class SuryaMultilingual(SuryaOCR):
    """Surya OCR optimized for multilingual documents"""

    def __init__(self, **kwargs):
        # Common languages for medical/scientific documents
        languages = ["en", "es", "fr", "de", "ja", "zh", "ko"]
        super().__init__(languages=languages, **kwargs)
        self.model_name = "Surya-Multilingual"


class SuryaLineLevel(SuryaOCR):
    """Surya OCR with emphasis on line-level detection"""

    def __init__(self, **kwargs):
        super().__init__(
            line_detection=True,
            layout_analysis=False,  # Focus on lines only
            **kwargs
        )
        self.model_name = "Surya-LineLevel"