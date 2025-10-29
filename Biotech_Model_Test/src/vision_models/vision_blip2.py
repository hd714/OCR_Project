"""
BLIP-2 Vision OCR Implementation
Uses Salesforce BLIP-2 for vision-language tasks
"""

from pathlib import Path
from typing import Optional, Dict, Any
import torch
from PIL import Image
import sys
import traceback

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from base_ocr import BaseOCR

class BLIP2VisionOCR(BaseOCR):
    """BLIP-2 implementation for vision-language understanding"""
    
    def __init__(self,
                 model_name_hf: str = "Salesforce/blip2-opt-2.7b",
                 use_gpu: bool = True,
                 load_in_8bit: bool = False,
                 **kwargs):
        """
        Initialize BLIP-2 model
        
        Args:
            model_name_hf: HuggingFace model name
            use_gpu: Whether to use GPU
            load_in_8bit: Use 8-bit quantization
        """
        super().__init__(model_name="BLIP-2", **kwargs)
        
        self.model_name_hf = model_name_hf
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.load_in_8bit = load_in_8bit
        self.processor = None
        self.model = None
        
    def _initialize_model(self):
        """Initialize BLIP-2 model and processor"""
        if self.model is None:
            try:
                from transformers import Blip2Processor, Blip2ForConditionalGeneration
                
                if self.logger:
                    self.logger.info(f"Loading BLIP-2 model: {self.model_name_hf}")
                
                # Load processor
                self.processor = Blip2Processor.from_pretrained(self.model_name_hf)
                
                # Load model with appropriate settings
                dtype = torch.float16 if self.use_gpu else torch.float32
                
                if self.load_in_8bit:
                    self.model = Blip2ForConditionalGeneration.from_pretrained(
                        self.model_name_hf,
                        load_in_8bit=True,
                        device_map="auto"
                    )
                else:
                    self.model = Blip2ForConditionalGeneration.from_pretrained(
                        self.model_name_hf,
                        torch_dtype=dtype
                    )
                    
                    if self.use_gpu:
                        self.model = self.model.cuda()
                
                if self.logger:
                    device = "GPU (8-bit)" if self.load_in_8bit else ("GPU" if self.use_gpu else "CPU")
                    self.logger.info(f"BLIP-2 model loaded on {device}")
                
            except ImportError:
                raise ImportError("Please install transformers: pip install transformers")
            except Exception as e:
                if self.logger:
                    self.logger.error(f"Failed to initialize BLIP-2: {e}")
                raise
    
    def _extract_text(self, file_path: Path, **kwargs) -> tuple[str, Optional[float], Dict[str, Any]]:
        """Extract text using BLIP-2"""
        
        # Initialize model
        self._initialize_model()
        
        metadata = {
            'engine': 'BLIP-2',
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
            
            # BLIP-2 works best with specific prompts
            prompts = [
                "Question: What text is visible in this image? Answer:",
                "Extract all text from this document:",
                "Read the text in this image:",
                "Question: What does this document say? Answer:"
            ]
            
            all_extractions = []
            
            for prompt in prompts:
                if self.logger:
                    self.logger.info(f"Processing with BLIP-2, prompt: {prompt[:50]}...")
                
                # Process inputs
                inputs = self.processor(images=image, text=prompt, return_tensors="pt")
                
                if self.use_gpu and not self.load_in_8bit:
                    inputs = {k: v.cuda() if torch.is_tensor(v) else v for k, v in inputs.items()}
                
                # Generate output
                with torch.no_grad():
                    generated_ids = self.model.generate(
                        **inputs,
                        max_new_tokens=kwargs.get('max_new_tokens', 1024),
                        temperature=kwargs.get('temperature', 0.1),
                        do_sample=kwargs.get('do_sample', False),
                        num_beams=kwargs.get('num_beams', 3)
                    )
                
                # Decode output
                generated_text = self.processor.batch_decode(
                    generated_ids,
                    skip_special_tokens=True
                )[0].strip()
                
                all_extractions.append(generated_text)
            
            # Combine all extractions (you can also choose the best one)
            if kwargs.get('combine_extractions', True):
                # Remove duplicates and combine
                unique_extractions = list(set(all_extractions))
                text = '\n\n'.join(unique_extractions)
            else:
                # Use the longest extraction
                text = max(all_extractions, key=len)
            
            # No explicit confidence from BLIP-2
            confidence = None
            
            metadata.update({
                'num_prompts_used': len(prompts),
                'output_length': len(text),
                'all_extractions_lengths': [len(e) for e in all_extractions]
            })
            
            # Optional: Get image caption for context
            if kwargs.get('include_caption', True):
                caption_inputs = self.processor(images=image, return_tensors="pt")
                
                if self.use_gpu and not self.load_in_8bit:
                    caption_inputs = {k: v.cuda() if torch.is_tensor(v) else v for k, v in caption_inputs.items()}
                
                with torch.no_grad():
                    caption_ids = self.model.generate(
                        **caption_inputs,
                        max_new_tokens=50
                    )
                
                caption = self.processor.batch_decode(
                    caption_ids,
                    skip_special_tokens=True
                )[0].strip()
                
                metadata['image_caption'] = caption
            
            return text, confidence, metadata
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"BLIP-2 processing failed: {e}")
                self.logger.error(f"Traceback: {traceback.format_exc()}")
            
            return "", 0, {'error': str(e), **metadata}