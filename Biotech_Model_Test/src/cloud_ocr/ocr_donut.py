"""
Donut (Document Understanding Transformer) Implementation
State-of-the-art end-to-end document understanding without OCR
"""

from pathlib import Path
from typing import Optional, Dict, Any
import torch
from PIL import Image
import sys
import traceback

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from base_ocr import BaseOCR

class DonutOCR(BaseOCR):
    """Donut implementation for document understanding"""
    
    def __init__(self,
                 model_name_hf: str = "naver-clova-ix/donut-base-finetuned-docvqa",
                 use_gpu: bool = True,
                 **kwargs):
        """
        Initialize Donut model
        
        Args:
            model_name_hf: HuggingFace model name
            use_gpu: Whether to use GPU
        """
        super().__init__(model_name="Donut", **kwargs)
        
        self.model_name_hf = model_name_hf
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.processor = None
        self.model = None
        
    def _initialize_model(self):
        """Initialize Donut model and processor"""
        if self.model is None:
            try:
                from transformers import DonutProcessor, VisionEncoderDecoderModel
                
                if self.logger:
                    self.logger.info(f"Loading Donut model: {self.model_name_hf}")
                
                # Load processor and model
                self.processor = DonutProcessor.from_pretrained(self.model_name_hf)
                self.model = VisionEncoderDecoderModel.from_pretrained(self.model_name_hf)
                
                # Move to GPU if available
                if self.use_gpu:
                    self.model = self.model.cuda()
                    if self.logger:
                        self.logger.info("Donut model loaded on GPU")
                else:
                    if self.logger:
                        self.logger.info("Donut model loaded on CPU")
                
                # Set to eval mode
                self.model.eval()
                
            except ImportError:
                raise ImportError("Please install transformers: pip install transformers")
            except Exception as e:
                if self.logger:
                    self.logger.error(f"Failed to initialize Donut: {e}")
                raise
    
    def _extract_text(self, file_path: Path, **kwargs) -> tuple[str, Optional[float], Dict[str, Any]]:
        """Extract text using Donut"""
        
        # Initialize model
        self._initialize_model()
        
        metadata = {
            'engine': 'Donut',
            'model': self.model_name_hf,
            'gpu_used': self.use_gpu
        }
        
        try:
            # Load image
            image = Image.open(file_path)
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            metadata['image_size'] = image.size
            
            # Prepare task prompt based on model type
            task_prompt = kwargs.get('task_prompt', '<s>')
            
            if 'docvqa' in self.model_name_hf.lower():
                # For DocVQA model, we can ask questions
                question = kwargs.get('question', 'What is the text content?')
                task_prompt = f"<s_docvqa><s_question>{question}</s_question><s_answer>"
            elif 'cord' in self.model_name_hf.lower():
                # For receipt understanding
                task_prompt = "<s_cord-v2>"
            elif 'rvlcdip' in self.model_name_hf.lower():
                # For document classification
                task_prompt = "<s_rvlcdip>"
            
            # Process image
            if self.logger:
                self.logger.info(f"Processing with Donut, task: {task_prompt[:50]}...")
            
            pixel_values = self.processor(image, return_tensors="pt").pixel_values
            
            if self.use_gpu:
                pixel_values = pixel_values.cuda()
            
            # Generate output
            with torch.no_grad():
                decoder_input_ids = self.processor.tokenizer(
                    task_prompt,
                    add_special_tokens=False,
                    return_tensors="pt"
                ).input_ids
                
                if self.use_gpu:
                    decoder_input_ids = decoder_input_ids.cuda()
                
                # Generate
                outputs = self.model.generate(
                    pixel_values,
                    decoder_input_ids=decoder_input_ids,
                    max_length=self.model.decoder.config.max_position_embeddings,
                    early_stopping=True,
                    pad_token_id=self.processor.tokenizer.pad_token_id,
                    eos_token_id=self.processor.tokenizer.eos_token_id,
                    use_cache=True,
                    num_beams=kwargs.get('num_beams', 1),
                    bad_words_ids=[[self.processor.tokenizer.unk_token_id]],
                    return_dict_in_generate=True
                )
            
            # Decode output
            sequence = self.processor.batch_decode(outputs.sequences)[0]
            sequence = sequence.replace(self.processor.tokenizer.eos_token, "")
            sequence = sequence.replace(self.processor.tokenizer.pad_token, "")
            
            # Parse based on task
            if 'docvqa' in self.model_name_hf.lower():
                # Extract answer
                text = sequence.split("<s_answer>")[-1].replace("</s_answer>", "").strip()
            elif 'cord' in self.model_name_hf.lower():
                # Parse receipt JSON
                import json
                try:
                    receipt_data = json.loads(sequence.replace(task_prompt, ""))
                    text = self._format_receipt(receipt_data)
                    metadata['structured_data'] = receipt_data
                except:
                    text = sequence.replace(task_prompt, "").strip()
            else:
                text = sequence.replace(task_prompt, "").strip()
            
            # For Donut, we don't have explicit confidence scores
            confidence = None
            
            metadata.update({
                'task_prompt': task_prompt,
                'raw_output': sequence[:500],  # First 500 chars
                'output_length': len(sequence)
            })
            
            return text, confidence, metadata
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"Donut processing failed: {e}")
                self.logger.error(f"Traceback: {traceback.format_exc()}")
            
            return "", 0, {'error': str(e), **metadata}
    
    def _format_receipt(self, receipt_data: Dict) -> str:
        """Format receipt data as readable text"""
        lines = []
        
        if 'company' in receipt_data:
            lines.append(f"Company: {receipt_data['company']}")
        if 'date' in receipt_data:
            lines.append(f"Date: {receipt_data['date']}")
        if 'total' in receipt_data:
            lines.append(f"Total: {receipt_data['total']}")
        
        if 'items' in receipt_data:
            lines.append("\nItems:")
            for item in receipt_data['items']:
                item_str = f"  - {item.get('name', 'Unknown')}"
                if 'price' in item:
                    item_str += f": {item['price']}"
                lines.append(item_str)
        
        return '\n'.join(lines)