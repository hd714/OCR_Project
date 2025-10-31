import os
import json
import easyocr
from datetime import datetime
from pdf2image import convert_from_path
import time

print("Processing PDFs with EasyOCR...")
print("=" * 60)

# Set Poppler path
poppler_path = r"C:\poppler\poppler-24.08.0\Library\bin"

# Initialize EasyOCR
reader = easyocr.Reader(['en'], gpu=False)

# Process only PDFs
documents = {
    "wen_documents/clinical_trials/ICML25_P229_mosunetuzumab_FL.pdf": "ICML25_P229_mosunetuzumab_FL",
    "wen_documents/presentations/pharma_day_2025.pdf": "pharma_day_2025"
}

for doc_path, doc_name in documents.items():
    print(f"\nProcessing: {doc_name}")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    try:
        start_time = time.time()
        # Convert PDF to images with explicit poppler path
        images = convert_from_path(doc_path, dpi=200, poppler_path=poppler_path)
        print(f"  Converted to {len(images)} pages")
        
        all_text = []
        for i, img in enumerate(images):
            print(f"  Processing page {i+1}/{len(images)}...")
            results = reader.readtext(img)
            for detection in results:
                if len(detection) >= 2:
                    all_text.append(detection[1])
        
        # Save results
        final_text = "\n".join(all_text)
        
        with open(f"outputs/{doc_name}_easyocr_{timestamp}.txt", 'w', encoding='utf-8') as f:
            f.write(f"EasyOCR Extraction Results\n")
            f.write(f"Document: {doc_name}\n")
            f.write("=" * 50 + "\n\n")
            f.write(final_text)
        
        print(f"  Extracted {len(all_text)} text blocks")
        print(f"  Processing time: {time.time() - start_time:.1f} seconds")
        
    except Exception as e:
        print(f"  Error: {str(e)}")

print("\nDone! Check outputs folder")
