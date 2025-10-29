"""
PDFPlumber Parser - Advanced PDF text extraction with table support
Best for: Documents with complex layouts, tables, and structured data
"""

from pathlib import Path
from typing import Optional, Dict, Any, List
import pdfplumber
import pandas as pd
import sys
import traceback

sys.path.insert(0, str(Path(__file__).parent.parent))
from base_ocr import BaseOCR

class PDFPlumberParser(BaseOCR):
    """Advanced PDF parser with table extraction capabilities"""
    
    def __init__(self, 
                 extract_tables: bool = True,
                 extract_metadata: bool = True,
                 **kwargs):
        """
        Initialize PDFPlumber parser
        
        Args:
            extract_tables: Whether to extract tables as structured data
            extract_metadata: Whether to extract PDF metadata
        """
        super().__init__(model_name="PDFPlumber", **kwargs)
        self.extract_tables = extract_tables
        self.extract_metadata = extract_metadata
    
    def _extract_text(self, file_path: Path, **kwargs) -> tuple[str, Optional[float], Dict[str, Any]]:
        """Extract text and tables from PDF"""
        
        metadata = {
            'engine': 'PDFPlumber',
            'file_type': file_path.suffix.lower()
        }
        
        if file_path.suffix.lower() != '.pdf':
            return "", None, {'error': 'PDFPlumber only supports PDF files', **metadata}
        
        try:
            all_text = []
            all_tables = []
            page_data = []
            
            with pdfplumber.open(file_path) as pdf:
                # Extract metadata
                if self.extract_metadata and pdf.metadata:
                    metadata['pdf_metadata'] = {
                        'author': pdf.metadata.get('Author'),
                        'title': pdf.metadata.get('Title'),
                        'subject': pdf.metadata.get('Subject'),
                        'creator': pdf.metadata.get('Creator'),
                        'producer': pdf.metadata.get('Producer'),
                        'creation_date': str(pdf.metadata.get('CreationDate')),
                        'modification_date': str(pdf.metadata.get('ModDate')),
                        'pages': len(pdf.pages)
                    }
                
                # Process each page
                for page_num, page in enumerate(pdf.pages, 1):
                    page_info = {
                        'page_number': page_num,
                        'width': page.width,
                        'height': page.height
                    }
                    
                    # Extract text
                    page_text = page.extract_text()
                    if page_text:
                        all_text.append(f"--- Page {page_num} ---\n{page_text}")
                        page_info['text_length'] = len(page_text)
                        page_info['word_count'] = len(page_text.split())
                    
                    # Extract tables
                    if self.extract_tables:
                        tables = page.extract_tables()
                        if tables:
                            page_info['table_count'] = len(tables)
                            for table_idx, table in enumerate(tables):
                                # Convert to DataFrame for better structure
                                try:
                                    df = pd.DataFrame(table[1:], columns=table[0] if table else None)
                                    all_tables.append({
                                        'page': page_num,
                                        'table_index': table_idx,
                                        'rows': len(df),
                                        'columns': len(df.columns),
                                        'data': df.to_dict('records'),
                                        'text_representation': df.to_string()
                                    })
                                    # Add table text to main text
                                    all_text.append(f"\n[Table {table_idx + 1} on Page {page_num}]\n{df.to_string()}\n")
                                except Exception as e:
                                    if self.logger:
                                        self.logger.warning(f"Failed to parse table on page {page_num}: {e}")
                    
                    # Look for specific biotech/pharma patterns
                    if page_text:
                        # Drug names (capitalized words followed by numbers/symbols)
                        import re
                        drug_pattern = r'\b[A-Z][A-Z0-9\-]+\b'
                        potential_drugs = re.findall(drug_pattern, page_text)
                        if potential_drugs:
                            page_info['potential_drug_mentions'] = list(set(potential_drugs))
                        
                        # Efficacy percentages
                        efficacy_pattern = r'\d+\.?\d*\s*%'
                        percentages = re.findall(efficacy_pattern, page_text)
                        if percentages:
                            page_info['percentages_found'] = percentages
                        
                        # Clinical trial phases
                        phase_pattern = r'Phase\s+[IVX123]+'
                        phases = re.findall(phase_pattern, page_text, re.IGNORECASE)
                        if phases:
                            page_info['clinical_phases'] = phases
                    
                    page_data.append(page_info)
            
            # Combine all text
            text = '\n\n'.join(all_text)
            
            # Calculate extraction quality metrics
            total_words = len(text.split())
            total_chars = len(text)
            
            # Simple confidence based on extraction success
            confidence = min(1.0, total_chars / 100)  # Normalize to 0-1
            
            metadata.update({
                'total_pages': len(page_data),
                'total_tables': len(all_tables),
                'total_words': total_words,
                'total_chars': total_chars,
                'page_details': page_data,
                'tables_extracted': all_tables if self.extract_tables else None,
                'extraction_method': 'direct_text',
                'requires_ocr': False
            })
            
            # Check if this might be a scanned PDF (no text extracted)
            if total_chars < 100 and len(page_data) > 0:
                metadata['requires_ocr'] = True
                metadata['extraction_quality'] = 'poor - likely scanned document'
            else:
                metadata['extraction_quality'] = 'good - native text extracted'
            
            return text, confidence, metadata
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"PDFPlumber extraction failed: {e}")
                self.logger.error(f"Traceback: {traceback.format_exc()}")
            
            return "", 0, {'error': str(e), **metadata}