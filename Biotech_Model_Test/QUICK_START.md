# Quick Start Guide - OCR Pipeline

## 1. Initial Setup (One Time)

```bash
# Navigate to project directory
cd Biotech_Model_Test

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Run setup script
python setup.py

# Or manually install dependencies
pip install -r requirements.txt
```

## 2. Test Your Setup

```bash
# Create and test a simple image
python test_ocr.py
```

## 3. Basic Usage Examples

### Process a Single Image
```bash
cd src
python main.py ../data/sample.png
```

### Use Specific Models Only
```bash
# Just Tesseract (fast baseline)
python main.py image.png --models tesseract

# Advanced models only
python main.py image.png --models tesseract_advanced paddleocr_advanced
```

### Process Directory
```bash
# Process all images in a folder
python main.py /path/to/images/

# Benchmark mode with full comparison
python main.py /path/to/images/ --benchmark
```

### Parallel Processing (Faster)
```bash
python main.py image.png --parallel --models tesseract easyocr paddleocr
```

## 4. Python Script Usage

```python
# Import the pipeline
import sys
sys.path.append('src')
from main import OCRPipeline

# Initialize pipeline
pipeline = OCRPipeline(
    models=['tesseract', 'easyocr'],  # Choose models
    output_dir='outputs',
    enable_gpu=True,
    parallel_processing=True
)

# Process single file
results = pipeline.process_file('document.png')

# Access results
for model_name, result in results.items():
    print(f"{model_name}: {result.word_count} words")
    print(f"Text preview: {result.text[:100]}")
    print(f"Processing time: {result.processing_time:.3f}s")

# Batch processing
files = ['doc1.png', 'doc2.png', 'doc3.png']
all_results = pipeline.batch_process(files)
```

## 5. Understanding the Output

### Directory Structure
```
outputs/
├── local_ocr/              # Extracted text files
│   ├── doc_tesseract_timestamp.txt
│   └── doc_tesseract_timestamp_metadata.json
├── comparisons/            # Model comparison JSONs
└── benchmarks/             # Performance metrics
```

### Metrics Explained
- **Processing Time**: How long OCR took (seconds)
- **Memory Usage**: RAM used (MB)
- **Word/Char Count**: Amount of text extracted
- **Chars/Second**: Processing speed metric
- **Confidence**: OCR confidence score (0-1)

## 6. Troubleshooting

### Issue: Tesseract not found
```bash
# Linux/Ubuntu
sudo apt-get install tesseract-ocr

# macOS
brew install tesseract

# Windows
# Download installer from GitHub
```

### Issue: Import errors
```bash
# Make sure you're in virtual environment
source .venv/bin/activate

# Reinstall requirements
pip install -r requirements.txt
```

### Issue: GPU not detected
```bash
# Install CUDA-enabled PyTorch
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

## 7. Advanced Features

### Custom Preprocessing
```python
from local_ocr.ocr_tesseract import TesseractOCR

ocr = TesseractOCR()
result = ocr.process(
    "image.png",
    preprocess_methods=['grayscale', 'denoise', 'threshold', 'deskew']
)
```

### Table Extraction (PaddleOCR)
```python
from local_ocr.ocr_paddleocr import PaddleOCROCR

ocr = PaddleOCROCR(enable_table=True)
result = ocr.process("table_document.png")
tables = ocr.extract_tables("table_document.png")
```

### Multilingual OCR
```python
from local_ocr.ocr_easyocr import EasyOCRMultilingual

ocr = EasyOCRMultilingual(auto_detect_languages=True)
result = ocr.process("multilingual_doc.png")
```

## 8. For Your Boss's Requirements

### Benchmarking for Evaluation
```bash
# Run comprehensive benchmark on test documents
python main.py ./test_documents --benchmark --models all

# Results will show:
# - Processing speed comparison
# - Memory usage
# - Accuracy metrics (confidence scores)
# - Best model for your document types
```

### Integration with Vector Database (Future)
```python
# Extract text for embedding
results = pipeline.process_file("pharmavision_poster.pdf")

# Best text for embedding
best_result = max(results.values(), key=lambda x: x.confidence)
text_for_embedding = best_result.text

# Future: Send to Milvus
# embeddings = generate_embeddings(text_for_embedding)
# milvus_client.insert(embeddings, metadata=best_result.to_dict())
```

## Need Help?
- Check README.md for detailed documentation
- Run `python test_ocr.py` to validate setup
- Review outputs/ directory for processing results
