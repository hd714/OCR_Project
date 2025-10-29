"""
PyPDF2 Parser - Fast, simple PDF text extraction
Best for: Quick extraction when you don't need tables or complex layouts
"""

from pathlib import Path
from typing import Optional, Dict, Any
import PyPDF2
from PyPDF2 import PdfReader
import sys
import traceback

sys.path.insert(0, str(Path(__file__).parent.parent))
from base_ocr import BaseOCR

class PyPDFParser(BaseOCR):
    """Fast PDF parser for simple text extraction"""
    
    def __init__(self, **kwargs):
        super().__init__(model_name="PyPDF2", **kwargs)
    
    def _extract_text(self, file_path: Path, **kwargs) -> tuple[str, Optional[float], Dict[str, Any]]:
        """Extract text from PDF using PyPDF2"""
        
        metadata = {
            'engine': 'PyPDF2',
            'file_type': file_path.suffix.lower()
        }
        
        if file_path.suffix.lower() != '.pdf':
            return "", None, {'error': 'PyPDF2 only supports PDF files', **metadata}
        
        try:
            all_text = []
            
            with open(file_path, 'rb') as file:
                pdf_reader = PdfReader(file)
                
                # Get metadata
                if pdf_reader.metadata:
                    metadata['pdf_metadata'] = {
                        'author': pdf_reader.metadata.get('/Author'),
                        'title': pdf_reader.metadata.get('/Title'),
                        'subject': pdf_reader.metadata.get('/Subject'),
                        'creator': pdf_reader.metadata.get('/Creator'),
                        'producer': pdf_reader.metadata.get('/Producer'),
                        'creation_date': str(pdf_reader.metadata.get('/CreationDate')),
                        'modification_date': str(pdf_reader.metadata.get('/ModDate')),
                    }
                
                num_pages = len(pdf_reader.pages)
                metadata['total_pages'] = num_pages
                
                # Extract text from each page
                for page_num, page in enumerate(pdf_reader.pages, 1):
                    try:
                        page_text = page.extract_text()
                        if page_text:
                            all_text.append(f"--- Page {page_num} ---\n{page_text}")
                    except Exception as e:
                        if self.logger:
                            self.logger.warning(f"Failed to extract text from page {page_num}: {e}")
            
            # Combine all text
            text = '\n\n'.join(all_text)
            
            # Basic metrics
            total_words = len(text.split())
            total_chars = len(text)
            
            # Simple confidence score
            confidence = min(1.0, total_chars / 100) if total_chars > 0 else 0
            
            metadata.update({
                'total_words': total_words,
                'total_chars': total_chars,
                'extraction_method': 'pypdf2_direct',
                'requires_ocr': total_chars < 100
            })
            
            return text, confidence, metadata
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"PyPDF2 extraction failed: {e}")
            
            return "", 0, {'error': str(e), **metadata}