"""
Main OCR Pipeline with Benchmarking - Extended Version
Coordinates multiple OCR engines, cloud services, and vision models
"""

import sys
from pathlib import Path
import argparse
from typing import Dict, List, Optional, Union
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
from rich.panel import Panel
import time
import json
from datetime import datetime
import concurrent.futures
import traceback
import os

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from base_ocr import BaseOCR, OCRResult, OCRBenchmarker

# Local OCR
from local_ocr.ocr_tesseract import TesseractOCR, TesseractAdvancedOCR
from local_ocr.ocr_easyocr import EasyOCROCR, EasyOCRMultilingual
from local_ocr.ocr_paddleocr import PaddleOCROCR, PaddleOCRAdvanced

# Cloud OCR
from cloud_ocr.ocr_azure import AzureOCR
from cloud_ocr.ocr_donut import DonutOCR

# Vision Models
from vision_models.vision_gpt4 import GPT4VisionOCR
from vision_models.vision_llama import LLaMAVisionOCR
from vision_models.vision_blip2 import BLIP2VisionOCR
from vision_models.vision_clip import CLIPVisionEmbedder

console = Console()

class OCRPipeline:
    def __init__(self, 
                 models: List[str] = None,
                 enable_gpu: bool = True,
                 parallel: bool = False,
                 cache_results: bool = True,
                 save_full_text: bool = True,
                 output_dir: Path = None,
                 api_keys: Dict[str, str] = None):
        
        self.enable_gpu = enable_gpu
        self.parallel = parallel
        self.cache_results = cache_results
        self.save_full_text = save_full_text
        self.output_dir = output_dir or Path("outputs")
        self.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
        self.api_keys = api_keys or {}
        
        self.models = models or ['tesseract']
        
        self.available_models = self._get_available_models()
        
        # Create output directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "local_ocr").mkdir(exist_ok=True)
        (self.output_dir / "cloud_ocr").mkdir(exist_ok=True)
        (self.output_dir / "vision_models").mkdir(exist_ok=True)
        (self.output_dir / "embeddings").mkdir(exist_ok=True)
        (self.output_dir / "comparisons").mkdir(exist_ok=True)
        (self.output_dir / "benchmarks").mkdir(exist_ok=True)
        
        self.benchmarker = OCRBenchmarker()
        
    def _get_available_models(self) -> Dict[str, type]:
        return {
            # Local OCR
            'tesseract': TesseractOCR,
            'tesseract_advanced': TesseractAdvancedOCR,
            'easyocr': EasyOCROCR,
            'easyocr_multilingual': EasyOCRMultilingual,
            'paddleocr': PaddleOCROCR,
            'paddleocr_advanced': PaddleOCRAdvanced,
            
            # Cloud OCR
            'azure': AzureOCR,
            'donut': DonutOCR,
            
            # Vision Models
            'gpt4v': GPT4VisionOCR,
            'llama_vision': LLaMAVisionOCR,
            'blip2': BLIP2VisionOCR,
            'clip': CLIPVisionEmbedder
        }
    
    def _initialize_model(self, model_name: str) -> Optional[BaseOCR]:
        if model_name not in self.available_models:
            console.print(f"[red]Model {model_name} not found[/red]")
            console.print(f"[yellow]Available models: {list(self.available_models.keys())}[/yellow]")
            return None
        
        try:
            model_class = self.available_models[model_name]
            
            kwargs = {
                'enable_logging': False,
                'cache_results': self.cache_results
            }
            
            # Local OCR settings
            if 'tesseract' in model_name:
                kwargs['tesseract_cmd'] = self.tesseract_cmd
                
            if 'easyocr' in model_name or 'paddle' in model_name:
                kwargs['use_gpu'] = self.enable_gpu
            
            # Cloud OCR settings
            if model_name == 'azure':
                kwargs['api_key'] = self.api_keys.get('azure_key') or os.getenv('AZURE_KEY')
                kwargs['endpoint'] = self.api_keys.get('azure_endpoint') or os.getenv('AZURE_ENDPOINT')
            
            # Vision model settings
            if model_name == 'gpt4v':
                kwargs['api_key'] = self.api_keys.get('openai_key') or os.getenv('OPENAI_API_KEY')
            
            if model_name in ['llama_vision', 'blip2', 'donut']:
                kwargs['use_gpu'] = self.enable_gpu
                kwargs['load_in_8bit'] = not self.enable_gpu  # Use 8-bit if no GPU
            
            if model_name == 'clip':
                kwargs['use_gpu'] = self.enable_gpu
            
            model = model_class(**kwargs)
            console.print(f"[green]✓ Initialized {model_name}[/green]")
            return model
            
        except Exception as e:
            console.print(f"[red]Failed to initialize {model_name}: {e}[/red]")
            console.print(f"[yellow]Traceback: {traceback.format_exc()}[/yellow]")
            return None
    
    def process_file(self, file_path: Union[str, Path], **kwargs) -> Dict[str, OCRResult]:
        file_path = Path(file_path)
        
        if not file_path.exists():
            console.print(f"[red]File not found: {file_path}[/red]")
            return {}
        
        console.print(Panel(f"[bold cyan]Processing: {file_path.name}[/bold cyan]"))
        console.print(f"[yellow]Models to run: {self.models}[/yellow]")
        
        results = {}
        
        if self.parallel and len(self.models) > 1:
            # Parallel processing
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
                future_to_model = {}
                for model_name in self.models:
                    model = self._initialize_model(model_name)
                    if model:
                        future = executor.submit(self._process_with_model, model, file_path, **kwargs)
                        future_to_model[future] = model_name
                
                for future in concurrent.futures.as_completed(future_to_model):
                    model_name = future_to_model[future]
                    try:
                        result = future.result()
                        results[model_name] = result
                        self.benchmarker.add_result(result)
                        console.print(f"[green]✓ {model_name}: {result.word_count} words in {result.processing_time:.3f}s[/green]")
                    except Exception as e:
                        console.print(f"[red]Error with {model_name}: {e}[/red]")
        else:
            # Sequential processing
            for model_name in self.models:
                model = self._initialize_model(model_name)
                if model:
                    try:
                        with Progress(
                            SpinnerColumn(),
                            TextColumn("[progress.description]{task.description}"),
                            console=console,
                            transient=True
                        ) as progress:
                            progress.add_task(f"[cyan]Processing with {model_name}...")
                            result = self._process_with_model(model, file_path, **kwargs)
                            results[model_name] = result
                            self.benchmarker.add_result(result)
                            
                            # Special handling for CLIP embeddings
                            if model_name == 'clip' and result.metadata.get('image_embedding'):
                                console.print(f"[green]✓ {model_name}: Generated embeddings (dim: {len(result.metadata['image_embedding'])})[/green]")
                            else:
                                console.print(f"[green]✓ {model_name}: {result.word_count} words in {result.processing_time:.3f}s[/green]")
                    except Exception as e:
                        console.print(f"[red]Error with {model_name}: {e}[/red]")
        
        if results and self.save_full_text:
            self._save_results(file_path, results)
        
        if len(results) > 1:
            console.print(self.benchmarker.get_comparison_table())
        
        return results
    
    def _process_with_model(self, model: BaseOCR, file_path: Path, **kwargs) -> OCRResult:
        """Process file with a single model"""
        return model.process(file_path, **kwargs)
    
    def _save_results(self, file_path: Path, results: Dict[str, OCRResult]):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = file_path.stem
        
        for model_name, result in results.items():
            # Determine output directory based on model type
            if model_name in ['tesseract', 'tesseract_advanced', 'easyocr', 'easyocr_multilingual', 'paddleocr', 'paddleocr_advanced']:
                output_subdir = "local_ocr"
            elif model_name in ['azure', 'donut']:
                output_subdir = "cloud_ocr"
            elif model_name in ['gpt4v', 'llama_vision', 'blip2']:
                output_subdir = "vision_models"
            elif model_name == 'clip':
                output_subdir = "embeddings"
            else:
                output_subdir = "other"
            
            # Save text if available
            if result.text:
                text_file = self.output_dir / output_subdir / f"{base_name}_{model_name}_{timestamp}.txt"
                with open(text_file, 'w', encoding='utf-8') as f:
                    f.write(result.text)
            
            # Save metadata
            metadata_file = self.output_dir / output_subdir / f"{base_name}_{model_name}_{timestamp}_metadata.json"
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(result.to_dict(), f, indent=2)
            
            # Save embeddings if available (for CLIP)
            if model_name == 'clip' and result.metadata.get('image_embedding'):
                import numpy as np
                embedding_file = self.output_dir / output_subdir / f"{base_name}_{model_name}_{timestamp}.npy"
                np.save(embedding_file, np.array(result.metadata['image_embedding']))
                console.print(f"[green]Saved embedding to {embedding_file}[/green]")
    
    def batch_process(self, file_paths: List[Union[str, Path]], **kwargs) -> Dict[str, Dict[str, OCRResult]]:
        """Process multiple files"""
        all_results = {}
        
        for file_path in file_paths:
            console.print(f"\n[bold]Processing file {file_paths.index(file_path) + 1}/{len(file_paths)}[/bold]")
            results = self.process_file(file_path, **kwargs)
            all_results[str(file_path)] = results
        
        return all_results

def main():
    parser = argparse.ArgumentParser(description="Extended OCR Pipeline with Cloud Services and Vision Models")
    parser.add_argument("input", nargs="?", help="Input file or directory to process")
    parser.add_argument("--models", nargs="+", 
                       default=["tesseract"],
                       help="OCR models to use (tesseract, azure, gpt4v, llama_vision, blip2, clip, etc.)")
    parser.add_argument("--no-gpu", action="store_true",
                       help="Disable GPU acceleration")
    parser.add_argument("--parallel", action="store_true",
                       help="Run models in parallel")
    parser.add_argument("--benchmark", action="store_true",
                       help="Run comprehensive benchmark of all models")
    parser.add_argument("--azure-key", help="Azure API key")
    parser.add_argument("--azure-endpoint", help="Azure endpoint")
    parser.add_argument("--openai-key", help="OpenAI API key")
    
    args = parser.parse_args()
    
    if not args.input:
        console.print("""
[yellow]No input file specified.[/yellow]

[bold]Examples:[/bold]
  python main.py image.png --models tesseract azure
  python main.py document.pdf --models gpt4v --openai-key YOUR_KEY
  python main.py folder/ --models tesseract azure gpt4v --benchmark
  python main.py image.jpg --models clip tesseract --parallel
        """)
        return
    
    # Prepare API keys
    api_keys = {}
    if args.azure_key:
        api_keys['azure_key'] = args.azure_key
    if args.azure_endpoint:
        api_keys['azure_endpoint'] = args.azure_endpoint
    if args.openai_key:
        api_keys['openai_key'] = args.openai_key
    
    # If benchmark mode, use all available models
    if args.benchmark:
        args.models = ['tesseract', 'easyocr', 'azure', 'gpt4v', 'blip2', 'clip']
        console.print("[yellow]Benchmark mode: Testing all available models[/yellow]")
    
    pipeline = OCRPipeline(
        models=args.models,
        enable_gpu=not args.no_gpu,
        parallel=args.parallel,
        api_keys=api_keys
    )
    
    input_path = Path(args.input)
    
    if input_path.is_file():
        results = pipeline.process_file(input_path)
        if not results:
            console.print("[red]No results obtained[/red]")
    elif input_path.is_dir():
        # Process all images in directory
        image_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.pdf']
        image_files = [f for f in input_path.iterdir() 
                      if f.suffix.lower() in image_extensions]
        
        if image_files:
            console.print(f"[cyan]Found {len(image_files)} files to process[/cyan]")
            all_results = pipeline.batch_process(image_files)
            
            # Save benchmark report
            if args.benchmark:
                report_file = pipeline.output_dir / "benchmarks" / f"benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
                with open(report_file, 'w') as f:
                    f.write(pipeline.benchmarker.get_comparison_table())
                console.print(f"[green]Benchmark report saved to {report_file}[/green]")
        else:
            console.print(f"[red]No image files found in {input_path}[/red]")
    else:
        console.print(f"[red]Input not found: {input_path}[/red]")

if __name__ == "__main__":
    main()