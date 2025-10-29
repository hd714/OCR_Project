#!/usr/bin/env python3
"""
Process Wen's biotech documents through OCR and store in Milvus
These are clinical trial posters and presentations with drug efficacy data
"""

import sys
import os
from pathlib import Path

# Add your OCR pipeline to path
sys.path.insert(0, str(Path.cwd() / "Biotech_Model_Test"))
sys.path.insert(0, str(Path.cwd() / "Biotech_Model_Test" / "src"))

from main import OCRPipeline
from milvus_pipeline import MilvusBiotechPipeline

def process_wen_documents():
    """Process all of Wen's documents"""
    
    print("="*70)
    print("PROCESSING WEN'S BIOTECH DOCUMENTS")
    print("="*70)
    
    # Initialize pipelines
    ocr = OCRPipeline(models=['tesseract'])  # Use fastest OCR
    milvus = MilvusBiotechPipeline()
    
    # Connect to Milvus
    if not milvus.connect_to_milvus():
        print("Error: Milvus not running. Start with: milvus.bat start")
        return
    
    milvus.create_collection()
    
    # Define the documents
    documents = [
        {
            'path': r'.\wen_documents\posters\3102_phase2_dose_optimization.jpg',
            'type': 'poster',
            'description': 'Phase II dose optimization with EZH2/EZH1 inhibitor'
        },
        {
            'path': r'.\wen_documents\posters\7023_axi_cel_outcomes.jpg',
            'type': 'poster',
            'description': 'Axicabtagene Ciloleucel (Axi-Cel) outcomes in lymphoma'
        },
        {
            'path': r'.\wen_documents\clinical_trials\ICML25_P229_mosunetuzumab_FL.pdf',
            'type': 'clinical_trial',
            'description': 'Mosunetuzumab + Lenalidomide in follicular lymphoma'
        },
        {
            'path': r'.\wen_documents\presentations\pharma_day_2025.pdf',
            'type': 'presentation',
            'description': 'Pharma Day 2025 presentation'
        }
    ]
    
    # Process each document
    for doc in documents:
        doc_path = Path(doc['path'])
        if not doc_path.exists():
            print(f"\nSkipping {doc['path']} - file not found")
            print(f"Please save the file to this location first")
            continue
        
        print(f"\n{'='*50}")
        print(f"Processing: {doc_path.name}")
        print(f"Type: {doc['type']}")
        print(f"Description: {doc['description']}")
        print('='*50)
        
        try:
            # OCR extraction
            results = ocr.process_file(doc_path)
            
            if results:
                text = list(results.values())[0].text
                processing_time = list(results.values())[0].processing_time
                
                print(f"✓ Extracted {len(text)} characters in {processing_time:.2f}s")
                
                # Look for key information
                if "efficacy" in text.lower() or "response" in text.lower():
                    print("✓ Found efficacy/response data")
                if "%" in text:
                    import re
                    percentages = re.findall(r'\d+(?:\.\d+)?%', text)
                    if percentages:
                        print(f"✓ Found percentages: {', '.join(percentages[:5])}")
                
                # Store in Milvus
                milvus.insert_document(text)
                print("✓ Stored in Milvus vector database")
                
        except Exception as e:
            print(f"✗ Error processing: {e}")
    
    # Test searches on the new documents
    print(f"\n{'='*70}")
    print("TESTING SEARCHES ON WEN'S DOCUMENTS")
    print('='*70)
    
    test_queries = [
        "tulmimetostat efficacy",
        "ARID1A mutation",
        "follicular lymphoma response rate",
        "mosunetuzumab maintenance",
        "axicabtagene ciloleucel outcomes",
        "EZH2 inhibitor dose"
    ]
    
    for query in test_queries:
        print(f"\nQuery: '{query}'")
        results = milvus.search(query, top_k=2)
        
        for hits in results:
            for i, hit in enumerate(hits[:1], 1):  # Show top result
                print(f"  Result {i}:")
                print(f"    Score: {hit.score:.3f}")
                if hit.entity.get('drug_name'):
                    print(f"    Drug: {hit.entity.get('drug_name')}")
                if hit.entity.get('efficacy'):
                    print(f"    Efficacy: {hit.entity.get('efficacy')}")
                preview = hit.entity.get('text', '')[:100]
                print(f"    Text: {preview}...")

if __name__ == "__main__":
    process_wen_documents()
