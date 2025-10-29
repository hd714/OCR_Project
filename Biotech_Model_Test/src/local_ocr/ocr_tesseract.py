"""
Tesseract OCR Implementation
Fast, reliable baseline OCR engine
"""

from pathlib import Path
from typing import Optional, Dict, Any, List
import pytesseract
from PIL import Image
import cv2
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from base_ocr import BaseOCR
import os

class TesseractOCR(BaseOCR):
    """Tesseract OCR implementation with preprocessing options"""
    
    def __init__(self, 
                 tesseract_cmd: Optional[str] = None,
                 lang: str = 'eng',
                 config: str = '',
                 preprocess: bool = False,
                 **kwargs):
        """
        Initialize Tesseract OCR
        
        Args:
            tesseract_cmd: Path to tesseract executable
            lang: Language(s) to use (e.g., 'eng', 'eng+fra')
            config: Additional Tesseract config parameters
            preprocess: Whether to apply image preprocessing
        """
        super().__init__(model_name="Tesseract", **kwargs)
        
        if tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
        elif os.path.exists(r"C:\Program Files\Tesseract-OCR\tesseract.exe"):
            pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
            
        self.lang = lang
        self.config = config or '--psm 6'
        self.preprocess = preprocess
        
        self.preprocessing_methods = {
            'grayscale': self._convert_to_grayscale,
            'threshold': self._apply_threshold,
            'denoise': self._denoise,
            'deskew': self._deskew,
            'resize': self._resize
        }
        
    def _convert_to_grayscale(self, image: np.ndarray) -> np.ndarray:
        """Convert image to grayscale"""
        if len(image.shape) == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return image
    
    def _apply_threshold(self, image: np.ndarray) -> np.ndarray:
        """Apply adaptive threshold to improve text clarity"""
        gray = self._convert_to_grayscale(image)
        return cv2.adaptiveThreshold(
            gray, 255, 
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 11, 2
        )
    
    def _denoise(self, image: np.ndarray) -> np.ndarray:
        """Remove noise from image"""
        return cv2.medianBlur(image, 3)
    
    def _deskew(self, image: np.ndarray) -> np.ndarray:
        """Correct image skew"""
        gray = self._convert_to_grayscale(image)
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        lines = cv2.HoughLines(edges, 1, np.pi/180, 200)
        
        if lines is not None:
            angles = []
            for rho, theta in lines[:, 0]:
                angle = (theta * 180 / np.pi) - 90
                if -45 <= angle <= 45:
                    angles.append(angle)
            
            if angles:
                median_angle = np.median(angles)
                if abs(median_angle) > 0.5:
                    return self._rotate_image(image, median_angle)
        
        return image
    
    def _rotate_image(self, image: np.ndarray, angle: float) -> np.ndarray:
        """Rotate image by given angle"""
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(image, M, (w, h), 
                                 flags=cv2.INTER_CUBIC, 
                                 borderMode=cv2.BORDER_REPLICATE)
        return rotated
    
    def _resize(self, image: np.ndarray, scale_factor: float = 2.0) -> np.ndarray:
        """Resize image to improve OCR accuracy"""
        h, w = image.shape[:2]
        new_h, new_w = int(h * scale_factor), int(w * scale_factor)
        return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    
    def _preprocess_image(self, image: np.ndarray, 
                         methods: List[str] = None) -> np.ndarray:
        """
        Apply preprocessing methods to improve OCR accuracy
        
        Args:
            image: Input image
            methods: List of preprocessing methods to apply
        
        Returns:
            Preprocessed image
        """
        if methods is None:
            methods = ['grayscale', 'denoise']
        
        processed = image.copy()
        for method in methods:
            if method in self.preprocessing_methods:
                processed = self.preprocessing_methods[method](processed)
                if self.logger:
                    self.logger.debug(f"Applied {method} preprocessing")
        
        return processed
    
    def _extract_text(self, file_path: Path, **kwargs) -> tuple[str, Optional[float], Dict[str, Any]]:
        """
        Extract text using Tesseract OCR
        
        Args:
            file_path: Path to image or PDF file
            **kwargs: Additional parameters
                - preprocess_methods: List of preprocessing methods
                - page_segmentation_mode: PSM value (0-13)
                - ocr_engine_mode: OEM value (0-3)
        
        Returns:
            Tuple of (text, confidence, metadata)
        """
        metadata = {
            'engine': 'Tesseract',
            'lang': self.lang,
            'config': self.config
        }
        
        try:
            image = cv2.imread(str(file_path))
            if image is None:
                pil_image = Image.open(file_path)
                if pil_image.mode != 'RGB':
                    pil_image = pil_image.convert('RGB')
                image = np.array(pil_image)
                if len(image.shape) == 2:
                    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
                elif image.shape[2] == 4:
                    image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
            
            if self.preprocess:
                preprocess_methods = kwargs.get('preprocess_methods')
                image = self._preprocess_image(image, preprocess_methods)
                metadata['preprocessing'] = preprocess_methods or ['grayscale', 'denoise']
            
            config = self.config
            if 'page_segmentation_mode' in kwargs:
                config = f'--psm {kwargs["page_segmentation_mode"]} ' + config
            if 'ocr_engine_mode' in kwargs:
                config = f'--oem {kwargs["ocr_engine_mode"]} ' + config
            
            text = pytesseract.image_to_string(
                image, 
                lang=self.lang,
                config=config
            ).strip()
            
            data = pytesseract.image_to_data(
                image, 
                lang=self.lang,
                config=config,
                output_type=pytesseract.Output.DICT
            )
            
            confidences = []
            for i, conf in enumerate(data['conf']):
                if int(conf) > 0:
                    confidences.append(int(conf))
            
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0
            
            metadata.update({
                'total_words': len(text.split()),
                'avg_confidence': avg_confidence,
                'min_confidence': min(confidences) if confidences else 0,
                'max_confidence': max(confidences) if confidences else 0,
                'image_shape': image.shape,
                'preprocessing_applied': self.preprocess
            })
            
            return text, avg_confidence / 100, metadata
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"Tesseract OCR failed: {e}")
            raise

class TesseractAdvancedOCR(TesseractOCR):
    """Advanced Tesseract with multiple pass processing and voting"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model_name = "Tesseract-Advanced"
        
    def _extract_text(self, file_path: Path, **kwargs) -> tuple[str, Optional[float], Dict[str, Any]]:
        """
        Extract text using multiple preprocessing strategies and voting
        
        This advanced version tries different preprocessing combinations
        and picks the best result based on confidence scores
        """
        strategies = [
            ['grayscale'],
            ['grayscale', 'threshold'],
            ['grayscale', 'denoise', 'threshold'],
            ['resize', 'grayscale', 'threshold'],
            []
        ]
        
        results = []
        metadata = {'strategies_tried': len(strategies)}
        
        for strategy in strategies:
            try:
                kwargs['preprocess_methods'] = strategy
                text, conf, meta = super()._extract_text(file_path, **kwargs)
                results.append({
                    'text': text,
                    'confidence': conf or 0,
                    'strategy': strategy,
                    'metadata': meta
                })
            except Exception as e:
                if self.logger:
                    self.logger.warning(f"Strategy {strategy} failed: {e}")
                continue
        
        if not results:
            raise ValueError("All preprocessing strategies failed")
        
        best_result = max(results, key=lambda x: x['confidence'])
        
        metadata.update({
            'best_strategy': best_result['strategy'],
            'all_confidences': [r['confidence'] for r in results],
            'confidence_variance': np.var([r['confidence'] for r in results]),
            **best_result['metadata']
        })
        
        return best_result['text'], best_result['confidence'], metadata