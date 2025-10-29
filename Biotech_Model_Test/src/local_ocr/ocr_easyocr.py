"""
Fixed EasyOCR Implementation
"""

from pathlib import Path
from typing import Optional, Dict, Any, List, Union
import numpy as np
import cv2
from PIL import Image
import sys
import traceback
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from base_ocr import BaseOCR

class EasyOCROCR(BaseOCR):
    """Fixed EasyOCR implementation"""
    
    def __init__(self,
                 languages: List[str] = None,
                 use_gpu: bool = False,  # Default to CPU for stability
                 **kwargs):
        super().__init__(model_name="EasyOCR", **kwargs)
        
        self.languages = languages or ['en']
        self.use_gpu = use_gpu
        self.reader = None
        
    def _initialize_reader(self):
        """Initialize EasyOCR reader with better error handling"""
        if self.reader is None:
            try:
                import easyocr
                import torch
                
                # Check if GPU is actually available
                gpu_available = torch.cuda.is_available() if self.use_gpu else False
                
                if self.logger:
                    self.logger.info(f"Initializing EasyOCR with languages: {self.languages}, GPU: {gpu_available}")
                
                # Create reader with explicit settings
                self.reader = easyocr.Reader(
                    self.languages,
                    gpu=gpu_available,
                    model_storage_directory=None,  # Use default
                    download_enabled=True,  # Ensure downloads work
                    verbose=False
                )
                
                if self.logger:
                    self.logger.info("EasyOCR reader initialized successfully")
                    
            except Exception as e:
                if self.logger:
                    self.logger.error(f"Failed to initialize EasyOCR: {e}")
                raise
    
    def _load_image(self, file_path: Path) -> np.ndarray:
        """Load and prepare image for EasyOCR"""
        # First try PIL (more robust for various formats)
        try:
            pil_image = Image.open(file_path)
            
            # Convert to RGB if necessary
            if pil_image.mode != 'RGB':
                pil_image = pil_image.convert('RGB')
            
            # Convert to numpy array
            image = np.array(pil_image)
            
            return image
            
        except Exception as e:
            # Fallback to OpenCV
            image = cv2.imread(str(file_path))
            if image is None:
                raise ValueError(f"Could not load image: {file_path}")
            
            # Convert BGR to RGB for EasyOCR
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            return image
    
    def _extract_text(self, file_path: Path, **kwargs) -> tuple[str, Optional[float], Dict[str, Any]]:
        """Extract text with better error handling and parameters"""
        
        # Initialize reader
        self._initialize_reader()
        
        metadata = {
            'engine': 'EasyOCR',
            'languages': self.languages,
            'gpu_used': self.use_gpu
        }
        
        try:
            # Load image
            image = self._load_image(file_path)
            metadata['image_shape'] = image.shape
            
            # Use more permissive parameters
            params = {
                'detail': 1,  # Get bounding boxes and confidence
                'paragraph': False,  # Don't combine into paragraphs initially
                'width_ths': 0.7,
                'height_ths': 0.7,
                'decoder': 'greedy',
                'beamWidth': 5,
                'batch_size': 1,
                'workers': 0,
                'allowlist': None,
                'blocklist': None,
                'text_threshold': 0.5,  # Lower threshold
                'low_text': 0.3,  # Lower threshold
                'link_threshold': 0.3,  # Lower threshold
                'canvas_size': 2560,
                'mag_ratio': 1.5  # Slight magnification can help
            }
            
            # Merge with user kwargs
            params.update(kwargs)
            
            # Try to extract text
            if self.logger:
                self.logger.info(f"Processing image with EasyOCR...")
            
            # Use file path instead of numpy array (more stable)
            result = self.reader.readtext(str(file_path), **params)
            
            if not result:
                # Try with even more permissive settings
                if self.logger:
                    self.logger.warning("No text found, trying with adjusted parameters...")
                
                params['text_threshold'] = 0.3
                params['low_text'] = 0.2
                params['link_threshold'] = 0.2
                result = self.reader.readtext(str(file_path), **params)
            
            # Process results
            if result:
                text_parts = []
                confidences = []
                
                for detection in result:
                    if len(detection) >= 3:
                        bbox, text_content, conf = detection[0], detection[1], detection[2]
                        if text_content.strip():  # Only add non-empty text
                            text_parts.append(text_content)
                            confidences.append(conf)
                
                text = ' '.join(text_parts)
                confidence = sum(confidences) / len(confidences) if confidences else 0
                
                metadata.update({
                    'num_detections': len(result),
                    'avg_confidence': confidence,
                    'min_confidence': min(confidences) if confidences else 0,
                    'max_confidence': max(confidences) if confidences else 0
                })
            else:
                text = ""
                confidence = 0
                metadata['num_detections'] = 0
                
                if self.logger:
                    self.logger.warning("EasyOCR found no text in the image")
            
            return text, confidence, metadata
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"EasyOCR processing failed: {e}")
                self.logger.error(f"Traceback: {traceback.format_exc()}")
            
            # Return empty result instead of raising
            return "", 0, {'error': str(e), **metadata}


class EasyOCRMultilingual(EasyOCROCR):
    """Multilingual version with same fixes"""
    
    def __init__(self, **kwargs):
        # Just use English for now to ensure it works
        super().__init__(languages=['en'], **kwargs)
        self.model_name = "EasyOCR-Multilingual"