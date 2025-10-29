"""
LLaMA Vision (LLaVA) OCR Implementation
Uses LLaVA models for vision-language understanding
"""

from pathlib import Path
from typing import Optional, Dict, Any
import torch
from PIL import Image
import sys
import traceback

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from base_ocr import BaseOCR

class LLaMAVisionOCR(BaseOCR):
    """LLaVA implementation for multimodal document understanding"""
    
    def __init__(self,
                 model_name_hf: str = "llava-hf/llava-1.5-7b-hf",
                 use_gpu: bool = True,
                 load_in_8bit: bool = False,
                 **kwargs):
        """
        Initialize LLaVA model
        
        Args:
            model_name_hf: HuggingFace model name
            use_gpu: Whether to use GPU
            load_in_8bit: Use 8-bit quantization to save memory
        """
        super().__init__(model_name="LLaMA-Vision", **kwargs)
        
        self.model_name_hf = model_name_hf
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.load_in_8bit = load_in_8bit
        self.processor = None
        self.model = None
        
    def _initialize_model(self):
        """Initialize LLaVA model and processor"""
        if self.model is None:
            try:
                from transformers import LlavaProcessor, LlavaForConditionalGeneration, BitsAndBytesConfig
                
                if self.logger:
                    self.logger.info(f"Loading LLaVA model: {self.model_name_hf}")
                
                # Quantization config if needed
                quantization_config = None
                if self.load_in_8bit:
                    quantization_config = BitsAndBytesConfig(
                        load_in_8bit=True,
                        bnb_8bit_compute_dtype=torch.float16
                    )
                
                # Load processor
                self.processor = LlavaProcessor.from_pretrained(self.model_name_hf)
                
                # Load model
                if quantization_config:
                    self.model = LlavaForConditionalGeneration.from_pretrained(
                        self.model_name_hf,
                        quantization_config=quantization_config,
                        device_map="auto"
                    )
                else:
                    self.model = LlavaForConditionalGeneration.from_pretrained(
                        self.model_name_hf,
                        torch_dtype=torch.float16 if self.use_gpu else torch.float32
                    )
                    
                    if self.use_gpu:
                        self.model = self.model.cuda()
                
                if self.logger:
                    device = "GPU (8-bit)" if self.load_in_8bit else ("GPU" if self.use_gpu else "CPU")
                    self.logger.info(f"LLaVA model loaded on {device}")
                
            except ImportError:
                raise ImportError("Please install transformers and bitsandbytes: pip install transformers bitsandbytes")
            except Exception as e:
                if self.logger:
                    self.logger.error(f"Failed to initialize LLaVA: {e}")
                raise
    
    def _extract_text(self, file_path: Path, **kwargs) -> tuple[str, Optional[float], Dict[str, Any]]:
        """Extract text using LLaVA"""
        
        # Initialize model
        self._initialize_model()
        
        metadata = {
            'engine': 'LLaVA',
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
            
            # Prepare prompt
            prompt = kwargs.get('prompt', """USER: <image>
Extract all text from this document image. Include:
- All visible text content
- Table structures (preserve formatting)
- Headers and sections
- Any captions or labels
- Brief descriptions of charts/figures and their text
ASSISTANT:""")
            
            if self.logger:
                self.logger.info(f"Processing with LLaVA...")
            
            # Process inputs
            inputs = self.processor(text=prompt, images=image, return_tensors="pt")
            
            if self.use_gpu and not self.load_in_8bit:
                inputs = {k: v.cuda() if torch.is_tensor(v) else v for k, v in inputs.items()}
            
            # Generate output
            with torch.no_grad():
                generate_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=kwargs.get('max_new_tokens', 2048),
                    temperature=kwargs.get('temperature', 0.1),
                    do_sample=kwargs.get('do_sample', False),
                    num_beams=kwargs.get('num_beams', 1)
                )
            
            # Decode output
            output = self.processor.batch_decode(
                generate_ids[:, inputs['input_ids'].shape[1]:],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )[0]
            
            text = output.strip()
            
            # No explicit confidence from LLaVA
            confidence = None
            
            metadata.update({
                'prompt_used': prompt[:100] + "...",
                'output_length': len(text),
                'max_new_tokens': kwargs.get('max_new_tokens', 2048)
            })
            
            # Optional: Extract structured information
            if kwargs.get('extract_structured', False):
                structure_prompt = """USER: <image>
Analyze this document and provide:
1. Document type
2. Main sections
3. Key information
4. Number of tables/figures
ASSISTANT:"""
                
                struct_inputs = self.processor(text=structure_prompt, images=image, return_tensors="pt")
                
                if self.use_gpu and not self.load_in_8bit:
                    struct_inputs = {k: v.cuda() if torch.is_tensor(v) else v for k, v in struct_inputs.items()}
                
                with torch.no_grad():
                    struct_ids = self.model.generate(
                        **struct_inputs,
                        max_new_tokens=512,
                        temperature=0.1,
                        do_sample=False
                    )
                
                structured_output = self.processor.batch_decode(
                    struct_ids[:, struct_inputs['input_ids'].shape[1]:],
                    skip_special_tokens=True
                )[0]
                
                metadata['structured_analysis'] = structured_output
            
            return text, confidence, metadata
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"LLaVA processing failed: {e}")
                self.logger.error(f"Traceback: {traceback.format_exc()}")
            
            return "", 0, {'error': str(e), **metadata}