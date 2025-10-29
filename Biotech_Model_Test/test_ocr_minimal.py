"""
Simple OCR test - minimal configuration
"""
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont

def create_ultra_simple_image():
    """Create the simplest possible test image"""
    # Create white image
    img = Image.new('RGB', (400, 100), color='white')
    draw = ImageDraw.Draw(img)
    
    # Add simple text (using default font)
    draw.text((10, 30), "HELLO WORLD 123", fill='black')
    
    # Save in multiple formats
    img.save("ultra_simple.png")
    img.save("ultra_simple.jpg")
    
    # Also create with OpenCV
    cv_img = np.ones((100, 400, 3), dtype=np.uint8) * 255
    cv2.putText(cv_img, "TEST", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,0,0), 2)
    cv2.imwrite("cv_simple.png", cv_img)
    
    print("Created test images: ultra_simple.png, ultra_simple.jpg, cv_simple.png")
    return "ultra_simple.png"

def test_easyocr_minimal():
    """Test EasyOCR with absolute minimal setup"""
    print("\n" + "="*50)
    print("Testing EasyOCR (Minimal)")
    print("="*50)
    
    try:
        import easyocr
        
        # Create reader with minimal config
        print("Creating reader (this may download ~64MB model on first run)...")
        reader = easyocr.Reader(['en'], gpu=False)
        
        # Test on simple image
        img_path = create_ultra_simple_image()
        
        # Method 1: Simple text list
        print("\nMethod 1: Simple text extraction")
        result = reader.readtext(img_path, detail=0)  # Just text, no boxes
        print(f"Result: {result}")
        
        # Method 2: With details
        print("\nMethod 2: Detailed extraction")
        result = reader.readtext(img_path, detail=1)  # With boxes and confidence
        for item in result:
            if len(item) >= 3:
                print(f"  Text: '{item[1]}', Confidence: {item[2]:.2f}")
        
        if result:
            print("\n✓ EasyOCR is working!")
            return True
        else:
            print("\n✗ EasyOCR returned no text")
            return False
            
    except Exception as e:
        print(f"\n✗ EasyOCR failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_paddleocr_minimal():
    """Test PaddleOCR with absolute minimal setup"""
    print("\n" + "="*50)
    print("Testing PaddleOCR (Minimal)")
    print("="*50)
    
    try:
        from paddleocr import PaddleOCR
        
        # Create OCR with minimal config
        print("Creating PaddleOCR (may download models on first run)...")
        ocr = PaddleOCR(lang='en', use_gpu=False)
        
        # Test on simple image
        img_path = "ultra_simple.png"
        
        print(f"\nProcessing {img_path}...")
        result = ocr.ocr(img_path)
        
        # Parse result
        if result and result[0]:
            print(f"Found {len(result[0])} text regions:")
            for line in result[0]:
                if line and len(line) >= 2:
                    text = line[1][0]
                    conf = line[1][1]
                    print(f"  Text: '{text}', Confidence: {conf:.2f}")
            print("\n✓ PaddleOCR is working!")
            return True
        else:
            print("\n✗ PaddleOCR returned no text")
            
            # Try with lower thresholds
            print("\nTrying with lower thresholds...")
            ocr2 = PaddleOCR(
                lang='en', 
                use_gpu=False,
                det_db_thresh=0.1,
                drop_score=0.1
            )
            result2 = ocr2.ocr(img_path)
            if result2 and result2[0]:
                print(f"With lower thresholds: Found {len(result2[0])} regions")
                return True
            return False
            
    except Exception as e:
        print(f"\n✗ PaddleOCR failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_tesseract_baseline():
    """Test Tesseract as baseline"""
    print("\n" + "="*50)
    print("Testing Tesseract (Baseline)")
    print("="*50)
    
    try:
        import pytesseract
        from PIL import Image
        
        # Set tesseract path for Windows
        import os
        if os.path.exists(r"C:\Program Files\Tesseract-OCR\tesseract.exe"):
            pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
        
        img = Image.open("ultra_simple.png")
        text = pytesseract.image_to_string(img)
        print(f"Tesseract result: '{text.strip()}'")
        
        if text.strip():
            print("✓ Tesseract is working!")
            return True
        return False
        
    except Exception as e:
        print(f"✗ Tesseract failed: {e}")
        return False

if __name__ == "__main__":
    print("="*50)
    print("MINIMAL OCR TEST")
    print("="*50)
    
    # Test all three
    tesseract_ok = test_tesseract_baseline()
    easyocr_ok = test_easyocr_minimal()
    paddleocr_ok = test_paddleocr_minimal()
    
    print("\n" + "="*50)
    print("RESULTS SUMMARY")
    print("="*50)
    print(f"Tesseract: {'✓ Working' if tesseract_ok else '✗ Not working'}")
    print(f"EasyOCR:   {'✓ Working' if easyocr_ok else '✗ Not working'}")
    print(f"PaddleOCR: {'✓ Working' if paddleocr_ok else '✗ Not working'}")
    
    if not easyocr_ok or not paddleocr_ok:
        print("\nTo fix non-working OCR engines:")
        print("1. Run: python fix_ocr_install.py")
        print("2. Make sure you have internet connection for model downloads")
        print("3. Check if you're behind a firewall/proxy")