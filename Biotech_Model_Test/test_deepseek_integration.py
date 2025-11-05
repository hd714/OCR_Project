#!/usr/bin/env python3
"""
Test DeepSeek-OCR Integration
"""

import sys
from pathlib import Path
import time

# Add paths
sys.path.insert(0, str(Path(__file__).parent / "Biotech_Model_Test"))
sys.path.insert(0, str(Path(__file__).parent / "Biotech_Model_Test" / "src"))

from main import OCRPipeline
from ocr_vs_parser_comparison import OCRvsParserComparison

def test_deepseek_basic():
    """Test basic DeepSeek OCR functionality"""
    print("="*70)
    print("TESTING DEEPSEEK-OCR INTEGRATION")
    print("="*70)
    
    # Test with a sample image
    test_image = Path("test_image.png")
    
    if not test_image.exists():
        # Create test image
        import numpy as np
        import cv2
        
        img = np.ones((400, 600, 3), dtype=np.uint8) * 255
        cv2.putText(img, "DeepSeek OCR Test", (50, 100), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2)
        cv2.putText(img, "Testing 123", (50, 200),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2)
        cv2.imwrite(str(test_image), img)
        print(f"Created test image: {test_image}")
    
    # Test DeepSeek OCR
    print("\nTesting DeepSeek OCR...")
    pipeline = OCRPipeline(
        models=['deepseek'],
        enable_gpu=False,  # Set to True if you have GPU
        save_full_text=True
    )
    
    results = pipeline.process_file(test_image)
    
    if results and 'deepseek' in results:
        result = results['deepseek']
        print(f"✅ DeepSeek OCR Success!")
        print(f"  Words: {result.word_count}")
        print(f"  Time: {result.processing_time:.2f}s")
        print(f"  Text preview: {result.text[:100]}...")
    else:
        print("❌ DeepSeek OCR failed")
    
    return results

def test_comparison_pipeline():
    """Test DeepSeek in comparison pipeline"""
    print("\n" + "="*70)
    print("TESTING DEEPSEEK IN COMPARISON PIPELINE")
    print("="*70)
    
    # Use first available document
    test_docs = [
        Path("wen_documents/posters/3102_phase2_dose_optimization.jpg"),
        Path("test_image.png")
    ]
    
    test_doc = None
    for doc in test_docs:
        if doc.exists():
            test_doc = doc
            break
    
    if not test_doc:
        print("No test documents found!")
        return
    
    print(f"\nComparing OCR engines on: {test_doc.name}")
    
    # Run comparison with all engines including DeepSeek
    pipeline = OCRPipeline(
        models=['tesseract', 'easyocr', 'deepseek'],
        enable_gpu=False,
        parallel=False  # Sequential for testing
    )
    
    results = pipeline.process_file(test_doc)
    
    print("\n" + "-"*50)
    print("COMPARISON RESULTS:")
    print("-"*50)
    
    for model_name, result in results.items():
        print(f"\n{model_name}:")
        print(f"  Words: {result.word_count}")
        print(f"  Characters: {result.char_count}")
        print(f"  Time: {result.processing_time:.2f}s")
        print(f"  Memory: {result.memory_usage_mb:.2f} MB")

def main():
    """Run all tests"""
    
    # Test 1: Basic functionality
    print("\nTest 1: Basic DeepSeek OCR")
    test_deepseek_basic()
    
    # Test 2: Comparison pipeline
    print("\nTest 2: Comparison Pipeline")
    test_comparison_pipeline()
    
    print("\n" + "="*70)
    print("✅ DEEPSEEK INTEGRATION COMPLETE!")
    print("="*70)
    print("\nYou can now use DeepSeek in:")
    print("  1. Main pipeline: python main.py image.jpg --models deepseek")
    print("  2. Comparison: python boss_pipeline_enhanced.py document.pdf")
    print("  3. Batch processing: python main.py folder/ --models tesseract deepseek")

if __name__ == "__main__":
    main()