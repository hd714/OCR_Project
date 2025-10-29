# --- process_wen_docs_fixed.py ---
# Handles large OCR documents by chunking them before sending to Milvus

import os
import sys
from pathlib import Path

# ----- make sure we can import your OCR code -----
ROOT = Path(__file__).resolve().parent
# Add Biotech_Model_Test and its src to sys.path
sys.path.insert(0, str(ROOT / "Biotech_Model_Test"))
sys.path.insert(0, str(ROOT / "Biotech_Model_Test" / "src"))

from main import OCRPipeline
from milvus_pipeline import MilvusBiotechPipeline
from milvus_chunking_fix import DocumentChunker


def process_wen_documents():
    print("="*70)
    print("PROCESSING WEN'S DOCUMENTS (CHUNKING ENABLED)")
    print("="*70)

    ocr = OCRPipeline(models=['tesseract'])
    milvus = MilvusBiotechPipeline()

    if not milvus.connect_to_milvus():
        print("Error: Milvus not running. Start it first.")
        return

    milvus.create_collection()

    docs = [
        {'path': r'.\wen_documents\posters\3102_phase2_dose_optimization.jpg', 'type': 'poster'},
        {'path': r'.\wen_documents\posters\7023_axi_cel_outcomes.jpg', 'type': 'poster'},
        {'path': r'.\wen_documents\clinical_trials\ICML25_P229_mosunetuzumab_FL.pdf', 'type': 'clinical_trial'},
        {'path': r'.\wen_documents\presentations\pharma_day_2025.pdf', 'type': 'presentation'}
    ]

    for doc in docs:
        path = ROOT / doc['path']
        if not path.exists():
            print(f"✗ Missing: {path}")
            continue

        print(f"\n--- Processing {path.name} ---")
        results = ocr.process_file(path)
        if not results:
            print(f"✗ OCR failed for {path.name}")
            continue

        text = list(results.values())[0].text
        print(f"✓ Extracted {len(text)} characters")

        # Chunk large texts before inserting
        if len(text) > 60000:
            print(f"Document too large ({len(text)} chars), chunking...")
            chunker = DocumentChunker(chunk_size=60000)
            chunks = chunker.chunk_text(text, {'source': path.name, 'type': doc['type']})
            print(f"Created {len(chunks)} chunks")
            for chunk in chunks:
                milvus.insert_document(chunk['text'][:60000])
                print(f"✓ Stored chunk {chunk['chunk_index']+1}/{chunk['total_chunks']}")
        else:
            milvus.insert_document(text)
            print("✓ Stored full document in Milvus")


if __name__ == "__main__":
    process_wen_documents()
