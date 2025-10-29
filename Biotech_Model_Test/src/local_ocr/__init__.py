# Local OCR implementations
from .ocr_tesseract import TesseractOCR, TesseractAdvancedOCR
from .ocr_easyocr import EasyOCROCR, EasyOCRMultilingual
from .ocr_paddleocr import PaddleOCROCR, PaddleOCRAdvanced

__all__ = [
    'TesseractOCR',
    'TesseractAdvancedOCR', 
    'EasyOCROCR',
    'EasyOCRMultilingual',
    'PaddleOCROCR',
    'PaddleOCRAdvanced'
]
