#!/usr/bin/env python3
"""
Complete OCR to Milvus Pipeline Integration
Connects your existing OCR pipeline to Milvus for production deployment
"""

import sys
import json
import time
from pathlib import Path
from typing import Dict, List, Any, Optional
import numpy as np

# Add paths for your existing modules
sys.path.insert(0, str(Path(__file__).parent / "Biotech_Model_Test"))
sys.path.insert(0, str(Path(__file__).parent / "Biotech_Model_Test" / "src"))

# Import your existing OCR components
from main import OCRPipeline
from base_ocr import OCRResult
from text_parsers.parse_pdfplumber import PDFPlumberParser
from text_parsers.parse_pypdf import PyPDFParser

# Import Milvus integration
from milvus_production import MilvusBiotechPipeline

class OCRToMilvusPipeline:
    """
    Complete pipeline from OCR extraction to Milvus storage and search
    Integrates your existing OCR work with Milvus
    """
    
    def __init__(self, 
                 ocr_models: List[str] = None,
                 milvus_host: str = "localhost",
                 milvus_port: str = "19530"):
        """
        Initialize the complete pipeline
        
        Args:
            ocr_models: OCR models to use (tesseract, easyocr, paddleocr)
            milvus_host: Milvus server host
            milvus_port: Milvus server port
        """
        # Initialize OCR pipeline with your preferred models
        self.ocr_models = ocr_models or ['tesseract']  # Default to fastest
        self.ocr_pipeline = OCRPipeline(
            models=self.ocr_models,
            enable_gpu=False,
            save_full_text=True
        )
        
        # Initialize Milvus connection
        self.milvus_pipeline = MilvusBiotechPipeline(
            host=milvus_host,
            port=milvus_port
        )
        
        # Statistics
        self.stats = {
            'documents_processed': 0,
            'total_processing_time': 0,
            'drugs_found': set(),
            'efficacies_found': []
        }
    
    def process_and_store_document(self, 
                                  file_path: Path,
                                  document_type: str = "clinical_trial") -> Dict[str, Any]:
        """
        Process a document through OCR and store in Milvus
        
        Args:
            file_path: Path to document
            document_type: Type of document (clinical_trial, poster, earnings_call, news)
        
        Returns:
            Processing results including Milvus document ID
        """
        file_path = Path(file_path)
        print(f"\n📄 Processing: {file_path.name}")
        print("="*50)
        
        result = {
            'file': str(file_path),
            'success': False,
            'milvus_id': None,
            'extraction_stats': {},
            'search_verification': {}
        }
        
        try:
            # Step 1: Extract text using OCR/Parser
            print("🔍 Step 1: Extracting text...")
            
            if file_path.suffix.lower() == '.pdf':
                # Try PDF parser first for native text
                parser = PDFPlumberParser()
                ocr_result = parser.process(file_path)
                
                # If no text found, fall back to OCR
                if ocr_result.word_count < 10:
                    print("  PDF has no embedded text, using OCR...")
                    ocr_results = self.ocr_pipeline.process_file(file_path)
                    ocr_result = list(ocr_results.values())[0]
            else:
                # Use OCR for images
                ocr_results = self.ocr_pipeline.process_file(file_path)
                ocr_result = list(ocr_results.values())[0]
            
            if not ocr_result.text:
                print("❌ No text extracted")
                return result
            
            print(f"✅ Extracted {ocr_result.word_count} words in {ocr_result.processing_time:.2f}s")
            
            # Step 2: Extract biotech information
            print("\n💊 Step 2: Extracting biotech information...")
            
            drug_name = self.milvus_pipeline._extract_drug_name(ocr_result.text)
            efficacy = self.milvus_pipeline._extract_efficacy(ocr_result.text)
            p_value = self.milvus_pipeline._extract_p_value(ocr_result.text)
            
            if drug_name:
                print(f"  Drug found: {drug_name}")
                self.stats['drugs_found'].add(drug_name)
            if efficacy:
                print(f"  Efficacy found: {efficacy}")
                self.stats['efficacies_found'].append(efficacy)
            if p_value:
                print(f"  P-value found: {p_value}")
            
            # Step 3: Store in Milvus
            print("\n💾 Step 3: Storing in Milvus...")
            
            doc_id = self.milvus_pipeline.insert_biotech_document(
                text=ocr_result.text,
                document_type=document_type,
                source_file=str(file_path),
                extraction_method=ocr_result.model_name,
                processing_time=ocr_result.processing_time,
                drug_name=drug_name,
                efficacy=efficacy,
                p_value=p_value
            )
            
            print(f"✅ Stored with ID: {doc_id}")
            
            # Step 4: Verify with search
            print("\n🔍 Step 4: Verifying with semantic search...")
            
            test_queries = []
            if drug_name:
                test_queries.append(f"What is {drug_name}'s efficacy?")
            if efficacy:
                test_queries.append(f"Find drugs with {efficacy} efficacy")
            if not test_queries:
                test_queries.append("Show clinical trial results")
            
            for query in test_queries:
                search_results = self.milvus_pipeline.semantic_search(query, top_k=1)
                if search_results:
                    print(f"  Query: '{query}'")
                    print(f"  ✅ Found (Score: {search_results[0]['score']:.3f})")
                    result['search_verification'][query] = search_results[0]['score']
            
            # Update results
            result['success'] = True
            result['milvus_id'] = doc_id
            result['extraction_stats'] = {
                'words': ocr_result.word_count,
                'processing_time': ocr_result.processing_time,
                'method': ocr_result.model_name,
                'drug_name': drug_name,
                'efficacy': efficacy,
                'p_value': p_value
            }
            
            # Update stats
            self.stats['documents_processed'] += 1
            self.stats['total_processing_time'] += ocr_result.processing_time
            
        except Exception as e:
            print(f"❌ Error: {e}")
            result['error'] = str(e)
        
        return result
    
    def batch_process_directory(self, 
                              directory: Path,
                              document_type: str = "clinical_trial") -> List[Dict[str, Any]]:
        """
        Process all documents in a directory
        
        Args:
            directory: Directory containing documents
            document_type: Type of documents
        
        Returns:
            List of processing results
        """
        directory = Path(directory)
        print(f"\n📁 Processing directory: {directory}")
        
        # Find all processable files
        extensions = ['.pdf', '.png', '.jpg', '.jpeg', '.tiff', '.bmp']
        files = []
        for ext in extensions:
            files.extend(directory.glob(f"*{ext}"))
            files.extend(directory.glob(f"*{ext.upper()}"))
        
        if not files:
            print("❌ No processable files found")
            return []
        
        print(f"Found {len(files)} files to process")
        
        results = []
        for i, file_path in enumerate(files, 1):
            print(f"\n[{i}/{len(files)}] Processing {file_path.name}")
            result = self.process_and_store_document(file_path, document_type)
            results.append(result)
        
        return results
    
    def test_biotech_queries(self):
        """Test the system with biotech-specific queries"""
        print("\n" + "="*70)
        print("TESTING BIOTECH SEMANTIC SEARCH")
        print("="*70)
        
        # Boss's specific requirements
        test_queries = [
            ("HUMIRA efficacy percentage", "Should find 75% efficacy"),
            ("adverse events table", "Should find headache, injection site reactions"),
            ("clinical trial p-values", "Should find p<0.001"),
            ("drug dosage recommendations", "Should find 40mg Q2W"),
            ("ACR20 response rate", "Should find ACR20 results"),
            ("find efficacy in tables", "Should find efficacy data from tables")
        ]
        
        print("\n🎯 Testing boss's requirements:")
        print("-"*50)
        
        for query, expected in test_queries:
            print(f"\nQuery: '{query}'")
            print(f"Expected: {expected}")
            
            results = self.milvus_pipeline.semantic_search(query, top_k=3)
            
            if results:
                top_result = results[0]
                print(f"✅ Found (Score: {top_result['score']:.3f})")
                print(f"   Drug: {top_result['drug_name']}")
                print(f"   Efficacy: {top_result['efficacy']}")
                print(f"   Source: {Path(top_result['source_file']).name}")
                print(f"   Text: {top_result['text'][:150]}...")
                
                # Check if we found what we expected
                found_expected = any(
                    exp.lower() in str(top_result).lower() 
                    for exp in expected.split() 
                    if len(exp) > 3
                )
                if found_expected:
                    print(f"   ✅ Contains expected information")
            else:
                print(f"❌ No results found")
    
    def print_summary(self):
        """Print processing summary"""
        print("\n" + "="*70)
        print("PROCESSING SUMMARY")
        print("="*70)
        
        print(f"\n📊 Statistics:")
        print(f"  Documents processed: {self.stats['documents_processed']}")
        print(f"  Total processing time: {self.stats['total_processing_time']:.2f}s")
        
        if self.stats['documents_processed'] > 0:
            avg_time = self.stats['total_processing_time'] / self.stats['documents_processed']
            print(f"  Average time per document: {avg_time:.2f}s")
        
        if self.stats['drugs_found']:
            print(f"\n💊 Drugs found:")
            for drug in self.stats['drugs_found']:
                print(f"  - {drug}")
        
        if self.stats['efficacies_found']:
            print(f"\n📈 Efficacies found:")
            for eff in self.stats['efficacies_found']:
                print(f"  - {eff}")
        
        # Get Milvus stats
        milvus_stats = self.milvus_pipeline.get_collection_stats()
        print(f"\n🗄️  Milvus Collection:")
        print(f"  Total documents: {milvus_stats['total_documents']}")
        print(f"  Embedding dimension: {milvus_stats['embedding_dimension']}")


def create_test_documents():
    """Create test documents for pipeline testing"""
    import cv2
    import numpy as np
    
    print("\n📝 Creating test documents...")
    
    # Create clinical trial document
    img = np.ones((1200, 1600, 3), dtype=np.uint8) * 255
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    cv2.putText(img, "CLINICAL TRIAL REPORT", (100, 100), font, 1.5, (0,0,0), 2)
    cv2.putText(img, "Drug: HUMIRA (adalimumab)", (100, 200), font, 1, (0,0,0), 1)
    cv2.putText(img, "Phase 3 Results:", (100, 300), font, 1, (0,0,0), 1)
    cv2.putText(img, "Primary Endpoint: 75% achieved ACR20 response", (100, 400), font, 1, (0,0,0), 1)
    cv2.putText(img, "Statistical Significance: p<0.001", (100, 500), font, 1, (0,0,0), 1)
    cv2.putText(img, "Adverse Events: Headache 12%, Nausea 8%", (100, 600), font, 1, (0,0,0), 1)
    cv2.putText(img, "Dosage: 40mg subcutaneous Q2W", (100, 700), font, 1, (0,0,0), 1)
    
    cv2.imwrite("test_clinical_trial.png", img)
    print("✅ Created test_clinical_trial.png")
    
    return ["test_clinical_trial.png"]


def main():
    """Main execution function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="OCR to Milvus Pipeline")
    parser.add_argument('--connect', action='store_true', help='Connect to Milvus and setup')
    parser.add_argument('--test', action='store_true', help='Run with test documents')
    parser.add_argument('--process', type=str, help='Process a specific file')
    parser.add_argument('--directory', type=str, help='Process all files in directory')
    parser.add_argument('--search', type=str, help='Search for a query')
    parser.add_argument('--ocr', default='tesseract', help='OCR engine to use')
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("OCR TO MILVUS PIPELINE")
    print("="*70)
    
    # Initialize pipeline
    pipeline = OCRToMilvusPipeline(ocr_models=[args.ocr])
    
    # Connect to Milvus
    print("\n🔌 Connecting to Milvus...")
    if not pipeline.milvus_pipeline.connect_to_milvus():
        print("\n❌ Could not connect to Milvus")
        print("\n📋 Please ensure Docker is running:")
        print("  1. Start Docker Desktop")
        print("  2. Run: docker-compose up -d")
        print("  3. Wait 30 seconds for services to start")
        print("  4. Try again")
        return
    
    # Create collection
    pipeline.milvus_pipeline.create_biotech_collection()
    
    if args.test:
        # Create and process test documents
        test_files = create_test_documents()
        
        for test_file in test_files:
            pipeline.process_and_store_document(Path(test_file))
        
        # Test queries
        pipeline.test_biotech_queries()
    
    elif args.process:
        # Process specific file
        pipeline.process_and_store_document(Path(args.process))
        pipeline.test_biotech_queries()
    
    elif args.directory:
        # Process directory
        results = pipeline.batch_process_directory(Path(args.directory))
        
        # Show results
        successful = sum(1 for r in results if r['success'])
        print(f"\n✅ Successfully processed {successful}/{len(results)} files")
        
        pipeline.test_biotech_queries()
    
    elif args.search:
        # Perform search
        results = pipeline.milvus_pipeline.semantic_search(args.search, top_k=5)
        
        print(f"\n🔍 Search results for: '{args.search}'")
        print("-"*50)
        
        for i, result in enumerate(results, 1):
            print(f"\n{i}. Score: {result['score']:.3f}")
            print(f"   Drug: {result['drug_name']}")
            print(f"   Efficacy: {result['efficacy']}")
            print(f"   Type: {result['document_type']}")
            print(f"   Text: {result['text'][:200]}...")
    
    else:
        print("\nUsage examples:")
        print("  python ocr_milvus_integration.py --test")
        print("  python ocr_milvus_integration.py --process document.pdf")
        print("  python ocr_milvus_integration.py --directory ./documents")
        print("  python ocr_milvus_integration.py --search 'HUMIRA efficacy'")
    
    # Print summary
    pipeline.print_summary()


if __name__ == "__main__":
    main()
