"""
CLIP Vision Embedder
Uses OpenAI CLIP for multimodal embeddings (text + image)
This is for creating embeddings for vector database storage
"""

from pathlib import Path
from typing import Optional, Dict, Any, List, Union, Tuple
import torch
import numpy as np
from PIL import Image
import sys
import traceback

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from base_ocr import BaseOCR

class CLIPVisionEmbedder(BaseOCR):
    """CLIP implementation for multimodal embeddings"""
    
    def __init__(self,
                 model_name: str = "openai/clip-vit-base-patch32",
                 use_gpu: bool = True,
                 **kwargs):
        """
        Initialize CLIP model
        
        Args:
            model_name: HuggingFace model name or OpenAI model name
            use_gpu: Whether to use GPU
        """
        super().__init__(model_name="CLIP", **kwargs)
        
        self.model_name_clip = model_name
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.model = None
        self.processor = None
        self.tokenizer = None
        
    def _initialize_model(self):
        """Initialize CLIP model"""
        if self.model is None:
            try:
                from transformers import CLIPProcessor, CLIPModel, CLIPTokenizer
                
                if self.logger:
                    self.logger.info(f"Loading CLIP model: {self.model_name_clip}")
                
                # Load model and processor
                self.model = CLIPModel.from_pretrained(self.model_name_clip)
                self.processor = CLIPProcessor.from_pretrained(self.model_name_clip)
                self.tokenizer = CLIPTokenizer.from_pretrained(self.model_name_clip)
                
                if self.use_gpu:
                    self.model = self.model.cuda()
                    if self.logger:
                        self.logger.info("CLIP model loaded on GPU")
                else:
                    if self.logger:
                        self.logger.info("CLIP model loaded on CPU")
                
                # Set to eval mode
                self.model.eval()
                
            except ImportError:
                raise ImportError("Please install transformers: pip install transformers")
            except Exception as e:
                if self.logger:
                    self.logger.error(f"Failed to initialize CLIP: {e}")
                raise
    
    def get_image_embedding(self, image_path: Path) -> np.ndarray:
        """Get image embedding using CLIP"""
        self._initialize_model()
        
        try:
            # Load image
            image = Image.open(image_path)
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Process image
            inputs = self.processor(images=image, return_tensors="pt")
            
            if self.use_gpu:
                inputs = {k: v.cuda() if torch.is_tensor(v) else v for k, v in inputs.items()}
            
            # Get image features
            with torch.no_grad():
                image_features = self.model.get_image_features(**inputs)
                
                # Normalize
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            
            # Convert to numpy
            embedding = image_features.cpu().numpy().squeeze()
            
            return embedding
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"Failed to get image embedding: {e}")
            raise
    
    def get_text_embedding(self, text: str) -> np.ndarray:
        """Get text embedding using CLIP"""
        self._initialize_model()
        
        try:
            # Tokenize text
            inputs = self.tokenizer(
                text,
                padding=True,
                truncation=True,
                max_length=77,  # CLIP's max length
                return_tensors="pt"
            )
            
            if self.use_gpu:
                inputs = {k: v.cuda() if torch.is_tensor(v) else v for k, v in inputs.items()}
            
            # Get text features
            with torch.no_grad():
                text_features = self.model.get_text_features(**inputs)
                
                # Normalize
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            
            # Convert to numpy
            embedding = text_features.cpu().numpy().squeeze()
            
            return embedding
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"Failed to get text embedding: {e}")
            raise
    
    def get_multimodal_embedding(self, 
                                 image_path: Optional[Path] = None,
                                 text: Optional[str] = None,
                                 fusion_method: str = "concatenate") -> np.ndarray:
        """
        Get combined multimodal embedding
        
        Args:
            image_path: Path to image
            text: Text content
            fusion_method: How to combine embeddings
                - "concatenate": Concatenate embeddings
                - "average": Average embeddings
                - "weighted": Weighted combination
        """
        embeddings = []
        
        if image_path:
            image_emb = self.get_image_embedding(image_path)
            embeddings.append(image_emb)
        
        if text:
            text_emb = self.get_text_embedding(text)
            embeddings.append(text_emb)
        
        if not embeddings:
            raise ValueError("Either image_path or text must be provided")
        
        if len(embeddings) == 1:
            return embeddings[0]
        
        # Combine embeddings based on method
        if fusion_method == "concatenate":
            return np.concatenate(embeddings)
        elif fusion_method == "average":
            return np.mean(embeddings, axis=0)
        elif fusion_method == "weighted":
            # Give more weight to text for documents
            weights = [0.3, 0.7] if image_path and text else [0.5, 0.5]
            return np.average(embeddings, axis=0, weights=weights)
        else:
            raise ValueError(f"Unknown fusion method: {fusion_method}")
    
    def _extract_text(self, file_path: Path, **kwargs) -> tuple[str, Optional[float], Dict[str, Any]]:
        """
        This method creates embeddings rather than extracting text
        Override to fit the BaseOCR interface
        """
        metadata = {
            'engine': 'CLIP',
            'model': self.model_name_clip,
            'gpu_used': self.use_gpu
        }
        
        try:
            # Get image embedding
            image_embedding = self.get_image_embedding(file_path)
            
            # If text is provided, create multimodal embedding
            text = kwargs.get('text', '')
            if text:
                text_embedding = self.get_text_embedding(text)
                multimodal_embedding = self.get_multimodal_embedding(
                    image_path=file_path,
                    text=text,
                    fusion_method=kwargs.get('fusion_method', 'concatenate')
                )
                
                metadata['embeddings'] = {
                    'image_embedding_shape': image_embedding.shape,
                    'text_embedding_shape': text_embedding.shape,
                    'multimodal_embedding_shape': multimodal_embedding.shape,
                    'fusion_method': kwargs.get('fusion_method', 'concatenate')
                }
            else:
                metadata['embeddings'] = {
                    'image_embedding_shape': image_embedding.shape
                }
            
            # Return embeddings in metadata
            metadata['image_embedding'] = image_embedding.tolist()
            if text:
                metadata['text_embedding'] = text_embedding.tolist()
                metadata['multimodal_embedding'] = multimodal_embedding.tolist()
            
            # For compatibility with BaseOCR, return empty text
            return "", None, metadata
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"CLIP embedding failed: {e}")
                self.logger.error(f"Traceback: {traceback.format_exc()}")
            
            return "", 0, {'error': str(e), **metadata}
    
    def compute_similarity(self, 
                          embedding1: np.ndarray, 
                          embedding2: np.ndarray) -> float:
        """Compute cosine similarity between two embeddings"""
        from numpy.linalg import norm
        
        # Normalize
        embedding1 = embedding1 / norm(embedding1)
        embedding2 = embedding2 / norm(embedding2)
        
        # Compute cosine similarity
        similarity = np.dot(embedding1, embedding2)
        
        return float(similarity)