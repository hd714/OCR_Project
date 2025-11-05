"""
DeepSeek-OCR Implementation
State-of-the-art OCR using DeepSeek's vision-language model
"""

from pathlib import Path
from typing import Optional, Dict, Any
import torch
from PIL import Image
import sys
import traceback
import time
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from base_ocr import BaseOCR

class DeepSeekOCR(BaseOCR):
    """DeepSeek-OCR implementation for high-quality text extraction"""
    
    def __init__(self,
                 model_name: str = "deepseek-ai/DeepSeek-OCR",
                 use_gpu: bool = True,
                 load_in_8bit: bool = False,
                 **kwargs):
        """
        Initialize DeepSeek-OCR
        
        Args:
            model_name: HuggingFace model name
            use_gpu: Whether to use GPU
            load_in_8bit: Use 8-bit quantization to save memory
        """
        super().__init__(model_name="DeepSeek-OCR", **kwargs)
        
        self.model_name_hf = model_name
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.load_in_8bit = load_in_8bit
        self.model = None
        self.processor = None
        self.tokenizer = None
        
    def _initialize_model(self):
        """Initialize DeepSeek model and processor"""
        if self.model is None:
            try:
                if self.logger:
                    self.logger.info(f"Loading DeepSeek-OCR model: {self.model_name_hf}")
                
                # Load tokenizer and processor
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.model_name_hf,
                    trust_remote_code=True
                )
                
                # Try to load processor, fallback to tokenizer if not available
                try:
                    self.processor = AutoProcessor.from_pretrained(
                        self.model_name_hf,
                        trust_remote_code=True
                    )
                except:
                    self.processor = self.tokenizer
                
                # Load model with appropriate configuration
                model_kwargs = {
                    "trust_remote_code": True,
                    "torch_dtype": torch.float16 if self.use_gpu else torch.float32
                }
                
                if self.load_in_8bit:
                    model_kwargs["load_in_8bit"] = True
                    model_kwargs["device_map"] = "auto"
                
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name_hf,
                    **model_kwargs
                )
                
                if self.use_gpu and not self.load_in_8bit:
                    self.model = self.model.cuda()
                
                self.model.eval()
                
                if self.logger:
                    device = "GPU (8-bit)" if self.load_in_8bit else ("GPU" if self.use_gpu else "CPU")
                    self.logger.info(f"DeepSeek-OCR model loaded on {device}")
                
            except ImportError:
                raise ImportError("Please install required packages: pip install transformers torch")
            except Exception as e:
                if self.logger:
                    self.logger.error(f"Failed to initialize DeepSeek-OCR: {e}")
                raise
    
    def _extract_text(self, file_path: Path, **kwargs) -> tuple[str, Optional[float], Dict[str, Any]]:
        """Extract text using DeepSeek-OCR"""
        
        # Initialize model
        self._initialize_model()
        
        metadata = {
            'engine': 'DeepSeek-OCR',
            'model': self.model_name_hf,
            'gpu_used': self.use_gpu,
            '8bit_quantization': self.load_in_8bit
        }
        
        try:
            # Load image
            image = Image.open(file_path)
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            metadata['image_size'] = image.size
            
            # Prepare prompt for OCR
            prompt = kwargs.get('prompt', "Extract all text from this image, preserving the layout and structure. Include all visible text, numbers, and special characters.")
            
            if self.logger:
                self.logger.info(f"Processing with DeepSeek-OCR...")
            
            # Process image and text
            if hasattr(self.processor, 'image_processor'):
                # If processor has image processing capability
                inputs = self.processor(
                    text=prompt,
                    images=image,
                    return_tensors="pt"
                )
            else:
                # Fallback to basic tokenization
                inputs = self.tokenizer(
                    prompt,
                    return_tensors="pt",
                    padding=True,
                    truncation=True
                )
                
                # Add image handling if model supports it
                if hasattr(self.model, 'vision_tower'):
                    # Model has vision capabilities
                    # This is a simplified approach - actual implementation may vary
                    inputs['pixel_values'] = self._process_image(image)
            
            if self.use_gpu and not self.load_in_8bit:
                inputs = {k: v.cuda() if torch.is_tensor(v) else v for k, v in inputs.items()}
            
            # Generate text
            with torch.no_grad():
                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=kwargs.get('max_new_tokens', 2048),
                    temperature=kwargs.get('temperature', 0.1),
                    do_sample=kwargs.get('do_sample', False),
                    num_beams=kwargs.get('num_beams', 1),
                    early_stopping=True
                )
            
            # Decode output
            if hasattr(output_ids, 'shape') and len(output_ids.shape) > 1:
                # Remove input tokens from output
                output_ids = output_ids[:, inputs['input_ids'].shape[-1]:]
            
            text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
            
            # Clean up the text
            text = text.strip()
            
            # DeepSeek doesn't provide explicit confidence scores
            confidence = None
            
            metadata.update({
                'prompt_used': prompt[:100] + "...",
                'output_length': len(text),
                'max_new_tokens': kwargs.get('max_new_tokens', 2048)
            })
            
            return text, confidence, metadata
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"DeepSeek-OCR processing failed: {e}")
                self.logger.error(f"Traceback: {traceback.format_exc()}")
            
            return "", 0, {'error': str(e), **metadata}
    
    def _process_image(self, image: Image.Image) -> torch.Tensor:
        """Process image for vision model input"""
        # This is a simplified image processing
        # Actual implementation depends on model requirements
        import torchvision.transforms as transforms
        
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        return transform(image).unsqueeze(0)


class DeepSeekOCRAdvanced(DeepSeekOCR):
    """Advanced version with specialized prompts for different document types"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model_name = "DeepSeek-OCR-Advanced"
    
    def _extract_text(self, file_path: Path, **kwargs) -> tuple[str, Optional[float], Dict[str, Any]]:
        """Extract text with document-type specific prompts"""
        
        # Determine document type from filename or metadata
        file_name = file_path.name.lower()
        
        if 'clinical' in file_name or 'trial' in file_name:
            kwargs['prompt'] = """Extract all text from this clinical trial document. 
            Pay special attention to:
            - Drug names and dosages
            - Efficacy percentages and p-values
            - Tables with adverse events
            - Patient demographics and statistics
            Preserve all numbers and maintain table structure."""
        
        elif 'poster' in file_name:
            kwargs['prompt'] = """Extract all text from this medical poster.
            Include:
            - Title and authors
            - All section headings
            - Body text and bullet points
            - Figure captions
            - Tables and their contents
            - References
            Maintain the hierarchical structure."""
        
        else:
            kwargs['prompt'] = """Extract all text from this document.
            Preserve:
            - Complete text content
            - Numbers and special characters
            - Table structures
            - Layout and formatting
            Be thorough and accurate."""
        
        return super()._extract_text(file_path, **kwargs)