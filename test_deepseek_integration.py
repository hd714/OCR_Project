#!/usr/bin/env python3
"""Test DeepSeek OCR Integration"""

import sys
from pathlib import Path
import numpy as np
import cv2

# Add paths
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "Biotech_Model_Test"))
sys.path.insert(0, str(Path(__file__).parent / "Biotech_Model_Test" / "src"))

def test_deepseek():
    print("="*70)
    print("TESTING DEEPSEEK OCR INTEGRATION")
    print("="*70)
    
    # Import DeepSeek
    from local_ocr.ocr_deepseek import DeepSeekOCR, DeepSeekOCRAdvanced
    print("✅ DeepSeek imported successfully")
    
    # Create test image
    print("\nCreating test image...")
    img = np.ones((400, 600, 3), dtype=np.uint8) * 255
    cv2.putText(img, "DeepSeek OCR Test", (50, 100), 
               cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2)
    cv2.putText(img, "HUMIRA 75% efficacy", (50, 200),
               cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2)
    cv2.putText(img, "p-value < 0.001", (50, 300),
               cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2)
    cv2.imwrite("test_deepseek.png", img)
    print("✅ Created test_deepseek.png")
    
    # Test basic DeepSeek
    print("\nTesting DeepSeekOCR...")
    deepseek = DeepSeekOCR(use_gpu=False)
    result = deepseek.process("test_deepseek.png")
    
    print(f"✅ Processing complete!")
    print(f"  Words: {result.word_count}")
    print(f"  Time: {result.processing_time:.2f}s")
    print(f"  Text preview: {result.text[:200]}...")
    
    # Test advanced version
    print("\nTesting DeepSeekOCRAdvanced...")
    deepseek_adv = DeepSeekOCRAdvanced(use_gpu=False)
    result_adv = deepseek_adv.process("test_deepseek.png")
    
    print(f"✅ Advanced processing complete!")
    print(f"  Words: {result_adv.word_count}")
    
    print("\n" + "="*70)
    print("✅ DEEPSEEK INTEGRATION SUCCESSFUL!")
    print("="*70)
    
    return True

if __name__ == "__main__":
    test_deepseek()
