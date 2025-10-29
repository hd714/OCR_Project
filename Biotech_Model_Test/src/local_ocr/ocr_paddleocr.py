
"""
Fixed PaddleOCR Implementation
"""

from pathlib import Path
from typing import Optional, Dict, Any, List, Union, Tuple
import numpy as np
import cv2
from PIL import Image
import sys
import traceback
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from base_ocr import BaseOCR

class PaddleOCROCR(BaseOCR):
    """Fixed PaddleOCR implementation"""
    
    def __init__(self,
                 lang: str = 'en',
                 use_gpu: bool = False,  # Default to CPU
                 use_angle_cls: bool = True,
                 **kwargs):
        super().__init__(model_name="PaddleOCR", **kwargs)
        
        self.lang = lang
        self.use_gpu = use_gpu
        self.use_angle_cls = use_angle_cls
        self.ocr_engine = None
        
    def _initialize_engine(self):
        """Initialize PaddleOCR with better error handling"""
        if self.ocr_engine is None:
            try:
                from paddleocr import PaddleOCR
                
                if self.logger:
                    self.logger.info(f"Initializing PaddleOCR for language: {self.lang}")
                
                # Create OCR with explicit settings
                self.ocr_engine = PaddleOCR(
                    use_angle_cls=self.use_angle_cls,
                    lang=self.lang,
                    use_gpu=self.use_gpu,
                    show_log=False,  # Disable verbose logging
                    det_model_dir=None,  # Use default models
                    rec_model_dir=None,
                    cls_model_dir=None,
                    det_db_thresh=0.3,  # Detection threshold
                    det_db_box_thresh=0.5,  # Box threshold
                    det_db_unclip_ratio=1.5,
                    use_space_char=True,
                    drop_score=0.5,  # Confidence threshold
                    det_limit_side_len=960,  # Max side length
                    rec_batch_num=6,
                    max_text_length=25,
                    rec_algorithm='SVTR_LCNet',
                    use_dilation=False,
                    det_db_score_mode='fast'
                )
                
                if self.logger:
                    self.logger.info("PaddleOCR initialized successfully")
                    
            except Exception as e:
                if self.logger:
                    self.logger.error(f"Failed to initialize PaddleOCR: {e}")
                    self.logger.error("Make sure PaddleOCR is properly installed:")
                    self.logger.error("  pip install paddlepaddle paddleocr")
                raise
    
    def _load_image(self, file_path: Path) -> np.ndarray:
        """Load image for PaddleOCR"""
        # PaddleOCR prefers OpenCV format (BGR)
        image = cv2.imread(str(file_path))
        
        if image is None:
            # Fallback to PIL
            try:
                pil_image = Image.open(file_path)
                if pil_image.mode != 'RGB':
                    pil_image = pil_image.convert('RGB')
                image = np.array(pil_image)
                # Convert RGB to BGR for PaddleOCR
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            except Exception as e:
                raise ValueError(f"Could not load image {file_path}: {e}")
        
        return image
    
    def _process_result(self, result: List) -> Tuple[str, float, List[Dict]]:
        """Process PaddleOCR result"""
        if not result:
            return "", 0.0, []
        
        # Handle different result formats
        if isinstance(result, list) and len(result) > 0:
            # Check if it's wrapped in an extra list
            if isinstance(result[0], list):
                result = result[0]
        
        if not result:
            return "", 0.0, []
        
        text_parts = []
        confidences = []
        detections = []
        
        for item in result:
            if item and len(item) >= 2:
                try:
                    bbox = item[0]
                    text_info = item[1]
                    
                    if isinstance(text_info, tuple) and len(text_info) >= 2:
                        text = str(text_info[0])
                        confidence = float(text_info[1])
                        
                        if text.strip():  # Only add non-empty text
                            text_parts.append(text)
                            confidences.append(confidence)
                            detections.append({
                                'text': text,
                                'confidence': confidence,
                                'bbox': bbox
                            })
                except Exception as e:
                    if self.logger:
                        self.logger.warning(f"Error processing detection: {e}")
                    continue
        
        # Combine text
        text = ' '.join(text_parts)
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0
        
        return text, avg_confidence, detections
    
    def _extract_text(self, file_path: Path, **kwargs) -> tuple[str, Optional[float], Dict[str, Any]]:
        """Extract text with better error handling"""
        
        # Initialize engine
        self._initialize_engine()
        
        metadata = {
            'engine': 'PaddleOCR',
            'language': self.lang,
            'gpu_used': self.use_gpu,
            'angle_classification': self.use_angle_cls
        }
        
        try:
            # Load image
            image = self._load_image(file_path)
            metadata['image_shape'] = image.shape
            
            if self.logger:
                self.logger.info(f"Processing image with PaddleOCR...")
            
            # Process with PaddleOCR
            # Use image path for more stable processing
            result = self.ocr_engine.ocr(str(file_path), cls=self.use_angle_cls)
            
            # Process results
            text, confidence, detections = self._process_result(result)
            
            if not text and self.logger:
                self.logger.warning("PaddleOCR found no text, trying with adjusted parameters...")
                
                # Try with different settings
                try:
                    # Reinitialize with lower thresholds
                    from paddleocr import PaddleOCR
                    ocr_permissive = PaddleOCR(
                        use_angle_cls=False,  # Disable angle classification
                        lang=self.lang,
                        use_gpu=self.use_gpu,
                        show_log=False,
                        det_db_thresh=0.1,  # Much lower threshold
                        det_db_box_thresh=0.3,
                        drop_score=0.3,  # Accept lower confidence
                        det_limit_side_len=1280,  # Larger size limit
                        use_dilation=True  # Enable dilation
                    )
                    
                    result = ocr_permissive.ocr(str(file_path), cls=False)
                    text, confidence, detections = self._process_result(result)
                    
                    if text:
                        metadata['used_permissive_settings'] = True
                        
                except Exception as e:
                    if self.logger:
                        self.logger.warning(f"Permissive attempt failed: {e}")
            
            metadata.update({
                'num_detections': len(detections),
                'avg_confidence': confidence,
                'min_confidence': min([d['confidence'] for d in detections]) if detections else 0,
                'max_confidence': max([d['confidence'] for d in detections]) if detections else 0
            })
            
            return text, confidence, metadata
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"PaddleOCR processing failed: {e}")
                self.logger.error(f"Traceback: {traceback.format_exc()}")
            
            # Return empty result instead of raising
            return "", 0, {'error': str(e), **metadata}


class PaddleOCRAdvanced(PaddleOCROCR):
    """Advanced version with rotation detection"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model_name = "PaddleOCR-Advanced"
    
    def _extract_text(self, file_path: Path, **kwargs) -> tuple[str, Optional[float], Dict[str, Any]]:
        """Try multiple orientations"""
        # For now, just use the base implementation
        # You can extend this to try multiple rotations if needed
        return super()._extract_text(file_path, **kwargs)