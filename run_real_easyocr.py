import os
import json
import easyocr
from datetime import datetime
from PIL import Image
from pdf2image import convert_from_path
import time

print("Starting REAL EasyOCR processing...")
print("=" * 60)

# Initialize EasyOCR
print("Initializing EasyOCR (this may take a moment)...")
reader = easyocr.Reader(['en'], gpu=False)  # Set gpu=True if you have CUDA
print(" EasyOCR initialized")

# Define the wen documents
documents = {
    "wen_documents/posters/3102_phase2_dose_optimization.jpg": "3102_phase2_dose_optimization",
    "wen_documents/posters/7023_axi_cel_outcomes.jpg": "7023_axi_cel_outcomes",
    "wen_documents/clinical_trials/ICML25_P229_mosunetuzumab_FL.pdf": "ICML25_P229_mosunetuzumab_FL",
    "wen_documents/presentations/pharma_day_2025.pdf": "pharma_day_2025"
}

# Process each document
for doc_path, doc_name in documents.items():
    print(f"\n Processing: {doc_name}")
    print("-" * 40)
    
    # Generate timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Output paths
    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)
    
    text_file = f"{output_dir}/{doc_name}_easyocr_{timestamp}.txt"
    metadata_file = f"{output_dir}/{doc_name}_easyocr_{timestamp}.metadata.json"
    
    try:
        start_time = time.time()
        all_text = []
        
        if doc_path.endswith('.pdf'):
            print(f"  Converting PDF to images...")
            # For PDFs, convert to images first
            try:
                images = convert_from_path(doc_path, dpi=200)
                print(f"   Converted to {len(images)} pages")
                
                for i, img in enumerate(images):
                    print(f"  Processing page {i+1}/{len(images)}...")
                    # Run EasyOCR on each page
                    results = reader.readtext(img)
                    
                    # Extract text from results
                    for detection in results:
                        if len(detection) >= 2:
                            all_text.append(detection[1])
                            
            except Exception as e:
                print(f"   PDF processing failed, trying alternative path...")
                # If PDF processing fails, create a note
                all_text.append(f"PDF processing failed: {str(e)[:100]}")
                
        else:
            # For images (JPG)
            print(f"  Processing image...")
            results = reader.readtext(doc_path)
            print(f"   Found {len(results)} text regions")
            
            # Extract text
            for detection in results:
                if len(detection) >= 2:
                    all_text.append(detection[1])
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        # Join all text
        final_text = "\n".join(all_text)
        
        # Save text file
        with open(text_file, 'w', encoding='utf-8') as f:
            f.write(f"EasyOCR Extraction Results\n")
            f.write(f"Document: {doc_name}\n")
            f.write(f"Timestamp: {timestamp}\n")
            f.write("=" * 50 + "\n\n")
            f.write(final_text)
        
        print(f"   Saved text: {text_file}")
        print(f"   Extracted {len(all_text)} text blocks")
        print(f"   Total characters: {len(final_text)}")
        
        # Create metadata
        metadata = {
            "ocr_engine": "easyocr",
            "version": "1.7.0",
            "processing_timestamp": timestamp,
            "document_name": doc_name,
            "file_path": doc_path,
            "statistics": {
                "total_text_blocks": len(all_text),
                "total_characters": len(final_text),
                "processing_time_seconds": round(processing_time, 2)
            }
        }
        
        # Save metadata
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"   Saved metadata: {metadata_file}")
        print(f"   Processing time: {processing_time:.1f} seconds")
        
    except Exception as e:
        print(f"   Error: {str(e)}")
        
        # Create error files
        with open(text_file, 'w', encoding='utf-8') as f:
            f.write(f"Error processing {doc_name}:\n{str(e)}")
        
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump({"error": str(e), "document": doc_name}, f, indent=2)

print("\n" + "=" * 60)
print(" EASYOCR PROCESSING COMPLETE")
print("=" * 60)
print("\nCheck the 'outputs' folder for results")
