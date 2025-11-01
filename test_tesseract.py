import pytesseract
from PIL import Image
import cv2
import numpy as np

# Set Tesseract path
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Test with the actual medical image
image_path = "wen_documents/posters/3102_phase2_dose_optimization.jpg"
print(f"Testing Tesseract with: {image_path}")
print("="*60)

# Method 1: Direct PIL
print("\nMethod 1: Direct PIL Image")
try:
    img_pil = Image.open(image_path)
    text = pytesseract.image_to_string(img_pil)
    print(f"Extracted {len(text)} characters")
    print(f"First 200 chars: {text[:200]}")
except Exception as e:
    print(f"Failed: {e}")

# Method 2: With preprocessing
print("\nMethod 2: With Preprocessing")
try:
    img_cv = cv2.imread(image_path)
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    # Increase contrast
    alpha = 1.5  # Contrast control
    beta = 10    # Brightness control
    adjusted = cv2.convertScaleAbs(gray, alpha=alpha, beta=beta)
    text = pytesseract.image_to_string(adjusted)
    print(f"Extracted {len(text)} characters")
    print(f"First 200 chars: {text[:200]}")
except Exception as e:
    print(f"Failed: {e}")

# Method 3: Different PSM modes
print("\nMethod 3: Testing PSM modes")
for psm in [3, 6, 11]:
    try:
        config = f'--psm {psm}'
        text = pytesseract.image_to_string(Image.open(image_path), config=config)
        print(f"PSM {psm}: Extracted {len(text)} characters")
    except Exception as e:
        print(f"PSM {psm} failed: {e}")
