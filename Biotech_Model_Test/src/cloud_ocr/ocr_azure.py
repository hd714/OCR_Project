"""
Azure Computer Vision OCR Implementation
Uses Azure's Document Intelligence API for high-quality OCR
"""

import os
import time
from pathlib import Path
from typing import Optional, Dict, Any
from PIL import Image
import sys
import traceback

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from base_ocr import BaseOCR

class AzureOCR(BaseOCR):
    """Azure Document Intelligence OCR implementation"""
    
    def __init__(self,
                 endpoint: str = None,
                 api_key: str = None,
                 model: str = "prebuilt-read",
                 **kwargs):
        """
        Initialize Azure OCR
        
        Args:
            endpoint: Azure endpoint URL
            api_key: Azure API key
            model: Azure model to use (prebuilt-read, prebuilt-document, etc.)
        """
        super().__init__(model_name="Azure-OCR", **kwargs)
        
        # Use provided credentials or environment variables
        self.endpoint = endpoint or os.getenv("AZURE_ENDPOINT", "https://biotech-ocr.cognitiveservices.azure.com/")
        self.api_key = api_key or os.getenv("AZURE_KEY")

        self.model = model
        self.client = None
        
    def _initialize_client(self):
        """Initialize Azure client"""
        if self.client is None:
            try:
                from azure.ai.formrecognizer import DocumentAnalysisClient
                from azure.core.credentials import AzureKeyCredential
                
                self.client = DocumentAnalysisClient(
                    endpoint=self.endpoint,
                    credential=AzureKeyCredential(self.api_key)
                )
                
                if self.logger:
                    self.logger.info("Azure Document Intelligence client initialized")
                    
            except ImportError:
                raise ImportError("Please install azure-ai-formrecognizer: pip install azure-ai-formrecognizer")
            except Exception as e:
                if self.logger:
                    self.logger.error(f"Failed to initialize Azure client: {e}")
                raise
    
    def _preprocess_image(self, file_path: Path) -> Path:
        """Resize image if needed for Azure limits"""
        if file_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
            try:
                image = Image.open(file_path)
                max_size = (4200, 4200)  # Azure's max dimensions
                
                if image.size[0] > max_size[0] or image.size[1] > max_size[1]:
                    image.thumbnail(max_size, Image.Resampling.LANCZOS)
                    temp_path = file_path.parent / f"temp_{file_path.name}"
                    image.save(temp_path, quality=95)
                    return temp_path
            except Exception as e:
                if self.logger:
                    self.logger.warning(f"Image preprocessing failed: {e}")
        
        return file_path
    
    def _extract_text(self, file_path: Path, **kwargs) -> tuple[str, Optional[float], Dict[str, Any]]:
        """Extract text using Azure Document Intelligence"""
        
        # Initialize client
        self._initialize_client()
        
        metadata = {
            'engine': 'Azure Document Intelligence',
            'model': self.model,
            'endpoint': self.endpoint
        }
        
        try:
            # Preprocess image if needed
            processed_path = self._preprocess_image(file_path)
            
            # Analyze document
            with open(processed_path, "rb") as f:
                if self.logger:
                    self.logger.info(f"Analyzing document with Azure {self.model}...")
                
                poller = self.client.begin_analyze_document(
                    self.model, 
                    f,
                    **kwargs  # Pass any additional Azure-specific parameters
                )
                result = poller.result()
            
            # Clean up temp file if created
            if processed_path != file_path:
                processed_path.unlink()
            
            # Extract text from results
            text_parts = []
            confidences = []
            
            # Process by pages
            for page_num, page in enumerate(result.pages, 1):
                page_text = []
                
                # Extract lines
                for line in page.lines:
                    page_text.append(line.content)
                    
                    # Get confidence if available
                    if hasattr(line, 'confidence') and line.confidence:
                        confidences.append(line.confidence)
                
                text_parts.append('\n'.join(page_text))
                
                metadata[f'page_{page_num}_lines'] = len(page.lines)
                
                # Extract tables if present
                if hasattr(page, 'tables'):
                    metadata[f'page_{page_num}_tables'] = len(page.tables)
            
            # Combine all pages
            text = '\n\n'.join(text_parts)
            
            # Calculate average confidence
            avg_confidence = sum(confidences) / len(confidences) if confidences else None
            
            # Add metadata
            metadata.update({
                'total_pages': len(result.pages),
                'avg_confidence': avg_confidence,
                'min_confidence': min(confidences) if confidences else None,
                'max_confidence': max(confidences) if confidences else None,
                'document_type': result.doc_type if hasattr(result, 'doc_type') else None,
                'has_tables': any(hasattr(page, 'tables') and page.tables for page in result.pages),
                'has_key_value_pairs': hasattr(result, 'key_value_pairs') and bool(result.key_value_pairs)
            })
            
            # Extract key-value pairs if available
            if hasattr(result, 'key_value_pairs') and result.key_value_pairs:
                kv_pairs = {}
                for kv in result.key_value_pairs:
                    if kv.key and kv.value:
                        key = kv.key.content if hasattr(kv.key, 'content') else str(kv.key)
                        value = kv.value.content if hasattr(kv.value, 'content') else str(kv.value)
                        kv_pairs[key] = value
                metadata['key_value_pairs'] = kv_pairs
            
            return text, avg_confidence, metadata
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"Azure OCR processing failed: {e}")
                self.logger.error(f"Traceback: {traceback.format_exc()}")
            
            # Return empty result with error
            return "", 0, {'error': str(e), **metadata}
