# Local OCR implementations
from .ocr_tesseract import TesseractOCR, TesseractAdvancedOCR
from .ocr_easyocr import EasyOCROCR, EasyOCRMultilingual
from .ocr_paddleocr import PaddleOCROCR, PaddleOCRAdvanced
from .ocr_deepseek import DeepSeekOCR, DeepSeekOCRAdvanced

__all__ = [
    'TesseractOCR',
    'TesseractAdvancedOCR', 
    'EasyOCROCR',
    'EasyOCRMultilingual',
    'PaddleOCROCR',
    'PaddleOCRAdvanced',
    'DeepSeekOCR',
    'DeepSeekOCRAdvanced'
]
