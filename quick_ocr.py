import pytesseract
from PIL import Image
import os
from datetime import datetime

# Set Tesseract path
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

def process_image(image_path):
    print(f"Processing: {image_path}")
    
    # Extract text
    img = Image.open(image_path)
    text = pytesseract.image_to_string(img)
    
    # Save output
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"tesseract_output_{timestamp}.txt"
    
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(text)
    
    print(f"SUCCESS! Extracted {len(text)} characters")
    print(f"Saved to: {output_file}")
    print(f"\nFirst 200 characters:")
    print(text[:200])
    
    return text

# Process the medical poster
if __name__ == "__main__":
    text = process_image("wen_documents/posters/3102_phase2_dose_optimization.jpg")
