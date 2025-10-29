"""
Base OCR Class with Built-in Benchmarking and Metrics
This will be the foundation for all OCR models to ensure consistent metrics tracking
"""

import traceback
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Union
from pathlib import Path
import time
import tracemalloc
import logging
from datetime import datetime
import json
import hashlib

class OCRResult:
    """Standardized result format for all OCR models"""
    
    def __init__(self, 
                 text: str = "",
                 processing_time: float = 0.0,
                 word_count: int = 0,
                 char_count: int = 0,
                 confidence: Optional[float] = None,
                 metadata: Optional[Dict[str, Any]] = None,
                 errors: Optional[List[str]] = None,
                 model_name: str = "unknown",
                 timestamp: Optional[datetime] = None,
                 memory_usage_mb: float = 0.0,
                 file_hash: str = ""):
        
        self.text = text
        self.processing_time = processing_time
        self.word_count = word_count or len(text.split())
        self.char_count = char_count or len(text)
        self.confidence = confidence
        self.metadata = metadata or {}
        self.errors = errors or []
        self.model_name = model_name
        self.timestamp = timestamp or datetime.now()
        self.memory_usage_mb = memory_usage_mb
        self.file_hash = file_hash
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary for JSON serialization"""
        return {
            'text': self.text[:500] + '...' if len(self.text) > 500 else self.text,
            'full_text_length': len(self.text),
            'processing_time': self.processing_time,
            'word_count': self.word_count,
            'char_count': self.char_count,
            'confidence': self.confidence,
            'metadata': self.metadata,
            'errors': self.errors,
            'model_name': self.model_name,
            'timestamp': self.timestamp.isoformat(),
            'memory_usage_mb': self.memory_usage_mb,
            'file_hash': self.file_hash,
            'chars_per_second': self.char_count / self.processing_time if self.processing_time > 0 else 0
        }
    
    def __str__(self) -> str:
        """Pretty print for terminal output"""
        return f"""
╔═══════════════════════════════════════════════════════╗
║ OCR Result: {self.model_name:<40} ║
╠═══════════════════════════════════════════════════════╣
║ Processing Time: {self.processing_time:.3f}s
║ Memory Usage: {self.memory_usage_mb:.2f} MB
║ Word Count: {self.word_count:,}
║ Character Count: {self.char_count:,}
║ Chars/Second: {(self.char_count/self.processing_time if self.processing_time > 0 else 0):,.0f}
║ Confidence: {self.confidence if self.confidence else 'N/A'}
║ Errors: {len(self.errors)}
║ Text Preview: {self.text[:100]}...
╚═══════════════════════════════════════════════════════╝
"""

class BaseOCR(ABC):
    """Abstract base class for all OCR implementations"""
    
    def __init__(self, model_name: str = "BaseOCR", 
                 enable_logging: bool = True,
                 cache_results: bool = True):
        """
        Initialize base OCR with common features
        
        Args:
            model_name: Name of the OCR model
            enable_logging: Whether to log processing details
            cache_results: Whether to cache results to avoid reprocessing
        """
        self.model_name = model_name
        self.cache_results = cache_results
        self.results_cache = {}
        
        if enable_logging:
            logging.basicConfig(level=logging.INFO)
            self.logger = logging.getLogger(model_name)
        else:
            self.logger = None
            
    def _get_file_hash(self, file_path: Path) -> str:
        """Generate hash of file for caching"""
        with open(file_path, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()
    
    def process(self, file_path: Union[str, Path], **kwargs) -> OCRResult:
        """
        Main processing method with built-in benchmarking
        
        Args:
            file_path: Path to the file to process
            **kwargs: Additional model-specific parameters
            
        Returns:
            OCRResult object with all metrics
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            return OCRResult(
                errors=[f"File not found: {file_path}"],
                model_name=self.model_name
            )
        
        file_hash = self._get_file_hash(file_path)
        if self.cache_results and file_hash in self.results_cache:
            if self.logger:
                self.logger.info(f"Returning cached result for {file_path.name}")
            return self.results_cache[file_hash]
        
        start_time = time.perf_counter()
        tracemalloc.start()
        errors = []
        
        try:
            text, confidence, metadata = self._extract_text(file_path, **kwargs)
            
            current, peak = tracemalloc.get_traced_memory()
            memory_mb = peak / 1024 / 1024
            tracemalloc.stop()
            
            processing_time = time.perf_counter() - start_time
            
            result = OCRResult(
                text=text,
                processing_time=processing_time,
                confidence=confidence,
                metadata=metadata,
                model_name=self.model_name,
                memory_usage_mb=memory_mb,
                file_hash=file_hash,
                errors=errors
            )
            
            if self.cache_results:
                self.results_cache[file_hash] = result
                
            if self.logger:
                self.logger.info(f"Successfully processed {file_path.name} in {processing_time:.3f}s")
                
            return result
            
        except Exception as e:
            tracemalloc.stop()
            error_msg = f"Error processing {file_path}: {str(e)}\n{traceback.format_exc()}"
            errors.append(error_msg)
            
            if self.logger:
                self.logger.error(error_msg)
                
            return OCRResult(
                errors=errors,
                model_name=self.model_name,
                processing_time=time.perf_counter() - start_time,
                file_hash=file_hash
            )
    
    @abstractmethod
    def _extract_text(self, file_path: Path, **kwargs) -> tuple[str, Optional[float], Dict[str, Any]]:
        """
        Abstract method to be implemented by each OCR model
        
        Args:
            file_path: Path to the file to process
            **kwargs: Model-specific parameters
            
        Returns:
            Tuple of (extracted_text, confidence_score, metadata_dict)
        """
        pass
    
    def batch_process(self, file_paths: List[Union[str, Path]], **kwargs) -> List[OCRResult]:
        """Process multiple files and return results"""
        results = []
        for file_path in file_paths:
            results.append(self.process(file_path, **kwargs))
        return results
    
    def compare_with(self, other_result: OCRResult) -> Dict[str, Any]:
        """Compare this result with another OCR result"""
        pass

class OCRBenchmarker:
    """Utility class to benchmark and compare multiple OCR models"""
    
    def __init__(self):
        self.results = []
        
    def add_result(self, result: OCRResult):
        """Add a result to the benchmark"""
        self.results.append(result)
        
    def get_comparison_table(self) -> str:
        """Generate a comparison table of all results"""
        if not self.results:
            return "No results to compare"
            
        sorted_results = sorted(self.results, key=lambda x: x.processing_time)
        
        table = "\n" + "="*80 + "\n"
        table += "OCR MODEL COMPARISON\n"
        table += "="*80 + "\n"
        table += f"{'Model':<20} {'Time (s)':<12} {'Memory (MB)':<12} {'Words':<10} {'Chars/Sec':<12}\n"
        table += "-"*80 + "\n"
        
        for result in sorted_results:
            chars_per_sec = result.char_count / result.processing_time if result.processing_time > 0 else 0
            table += f"{result.model_name:<20} {result.processing_time:<12.3f} {result.memory_usage_mb:<12.2f} {result.word_count:<10,} {chars_per_sec:<12,.0f}\n"
            
        table += "="*80 + "\n"
        
        if len(self.results) > 1:
            fastest = sorted_results[0]
            slowest = sorted_results[-1]
            table += f"\n🏆 Fastest: {fastest.model_name} ({fastest.processing_time:.3f}s)\n"
            table += f"🐢 Slowest: {slowest.model_name} ({slowest.processing_time:.3f}s)\n"
            table += f"⚡ Speed Difference: {(slowest.processing_time/fastest.processing_time):.1f}x\n"
            
        return table
    
    def save_results(self, output_path: Union[str, Path]):
        """Save all results to JSON file"""
        output_path = Path(output_path)
        data = {
            'timestamp': datetime.now().isoformat(),
            'results': [r.to_dict() for r in self.results]
        }
        
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)