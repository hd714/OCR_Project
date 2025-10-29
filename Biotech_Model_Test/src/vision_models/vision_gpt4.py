"""
GPT-4 Vision OCR Implementation
Uses OpenAI's GPT-4V for advanced document understanding
"""

import os
import base64
from pathlib import Path
from typing import Optional, Dict, Any
from PIL import Image
import sys
import traceback
import json

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from base_ocr import BaseOCR

class GPT4VisionOCR(BaseOCR):
    """GPT-4 Vision implementation for OCR and document understanding"""
    
    def __init__(self,
                 api_key: str = None,
                 model: str = "gpt-4-vision-preview",
                 max_tokens: int = 4096,
                 **kwargs):
        """
        Initialize GPT-4 Vision
        
        Args:
            api_key: OpenAI API key
            model: Model to use (gpt-4-vision-preview)
            max_tokens: Maximum tokens in response
        """
        super().__init__(model_name="GPT-4-Vision", **kwargs)
        
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model
        self.max_tokens = max_tokens
        self.client = None
        
    def _initialize_client(self):
        """Initialize OpenAI client"""
        if self.client is None:
            try:
                from openai import OpenAI
                
                self.client = OpenAI(api_key=self.api_key)
                
                if self.logger:
                    self.logger.info("OpenAI GPT-4V client initialized")
                    
            except ImportError:
                raise ImportError("Please install openai: pip install openai")
            except Exception as e:
                if self.logger:
                    self.logger.error(f"Failed to initialize OpenAI client: {e}")
                raise
    
    def _encode_image(self, image_path: Path) -> str:
        """Encode image to base64"""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    
    def _extract_text(self, file_path: Path, **kwargs) -> tuple[str, Optional[float], Dict[str, Any]]:
        """Extract text using GPT-4 Vision"""
        
        # Initialize client
        self._initialize_client()
        
        metadata = {
            'engine': 'GPT-4 Vision',
            'model': self.model
        }
        
        try:
            # Prepare the prompt
            prompt = kwargs.get('prompt', """Please extract all text from this image. 
            If there are tables, preserve their structure. 
            If there are multiple sections, maintain the hierarchy.
            Include all visible text, headers, footers, captions, and labels.
            For charts or figures, describe them briefly and extract any text labels.""")
            
            # Encode image
            base64_image = self._encode_image(file_path)
            
            if self.logger:
                self.logger.info(f"Sending image to GPT-4V...")
            
            # Create the request
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}",
                                    "detail": kwargs.get("detail", "high")  # "low", "high", or "auto"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=self.max_tokens,
                temperature=kwargs.get("temperature", 0)
            )
            
            # Extract text from response
            text = response.choices[0].message.content
            
            # GPT-4V doesn't provide explicit confidence scores
            confidence = None
            
            # Add metadata
            metadata.update({
                'usage': {
                    'prompt_tokens': response.usage.prompt_tokens,
                    'completion_tokens': response.usage.completion_tokens,
                    'total_tokens': response.usage.total_tokens
                },
                'finish_reason': response.choices[0].finish_reason,
                'model_used': response.model,
                'detail_level': kwargs.get("detail", "high")
            })
            
            # If structured extraction requested
            if kwargs.get('extract_structured', False):
                structured_prompt = """Also provide a JSON summary with:
                - document_type
                - key_sections
                - tables_found
                - figures_found
                - key_information"""
                
                structure_response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": structured_prompt},
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/jpeg;base64,{base64_image}",
                                        "detail": "low"
                                    }
                                }
                            ]
                        }
                    ],
                    max_tokens=1024,
                    temperature=0,
                    response_format={"type": "json_object"}
                )
                
                try:
                    structured_data = json.loads(structure_response.choices[0].message.content)
                    metadata['structured_extraction'] = structured_data
                except:
                    pass
            
            return text, confidence, metadata
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"GPT-4V processing failed: {e}")
                self.logger.error(f"Traceback: {traceback.format_exc()}")
            
            return "", 0, {'error': str(e), **metadata}