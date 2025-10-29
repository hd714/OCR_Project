# OCR Pipeline - Multimodal Document Processing System

A production-ready, benchmarking-focused OCR pipeline for evaluating and comparing multiple OCR engines. Built specifically for biotech/pharma document analysis with future support for multimodal embeddings and vector databases.

## 🎯 Key Features

### Built-in from Day 1:
- **Comprehensive Benchmarking**: Processing time, memory usage, character/word counts, confidence scores
- **Parallel Processing**: Run multiple OCR engines simultaneously
- **Result Caching**: Avoid reprocessing the same files
- **Rich Terminal UI**: Beautiful progress bars and comparison tables
- **Standardized Output Format**: Consistent results across all engines
- **Error Handling & Recovery**: Graceful handling of failures
- **Multiple Processing Strategies**: Different preprocessing methods for optimal results

### OCR Engines Supported:
1. **Tesseract** (+ Advanced version with multi-strategy voting)
2. **EasyOCR** (+ Multilingual version with auto-language detection)  
3. **PaddleOCR** (+ Advanced version with rotation detection)

## 📊 Architecture

```
┌─────────────────────────────────────────────────────────┐
│                     Main Pipeline                        │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐    │
│  │  Tesseract  │  │   EasyOCR   │  │  PaddleOCR  │    │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘    │
│         │                 │                 │           │
│         └─────────────────┼─────────────────┘           │
│                           │                             │
│                    ┌──────▼──────┐                     │
│                    │   BaseOCR   │                     │
│                    │  (Metrics)  │                     │
│                    └──────┬──────┘                     │
│                           │                             │
│                    ┌──────▼──────┐                     │
│                    │ OCRResult   │                     │
│                    │(Standardized)│                     │
│                    └──────┬──────┘                     │
│                           │                             │
│                    ┌──────▼──────┐                     │
│                    │ Benchmarker │                     │
│                    │ (Comparison)│                     │
│                    └─────────────┘                     │
└─────────────────────────────────────────────────────────┘
```

## 🚀 Quick Start

### Installation

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install Tesseract (system dependency)
# Ubuntu/Debian:
sudo apt-get install tesseract-ocr
# macOS:
brew install tesseract
# Windows: Download from https://github.com/UB-Mannheim/tesseract/wiki
```

### Basic Usage

```bash
# Test your setup
python test_ocr.py

# Process single file
python main.py image.png

# Process with specific models
python main.py image.png --models tesseract easyocr

# Benchmark mode (process entire directory)
python main.py /path/to/images --benchmark

# Parallel processing
python main.py image.png --parallel

# Disable GPU
python main.py image.png --no-gpu
```

### Interactive Mode

```bash
# Run without arguments for interactive mode
python main.py
```

## 📈 Metrics Tracked

Every OCR run automatically tracks:

| Metric | Description |
|--------|-------------|
| **Processing Time** | Total time to process document (seconds) |
| **Memory Usage** | Peak memory consumption (MB) |
| **Word Count** | Number of words extracted |
| **Character Count** | Total characters extracted |
| **Chars/Second** | Processing speed metric |
| **Confidence Score** | Average OCR confidence (0-1) |
| **Min/Max Confidence** | Confidence range |
| **Error Count** | Number of errors encountered |

## 🔧 Extending the Pipeline

### Adding New OCR Engines

1. Create new file: `ocr_newengine.py`
2. Inherit from `BaseOCR`:

```python
from base_ocr import BaseOCR
from pathlib import Path
from typing import Optional, Dict, Any

class NewEngineOCR(BaseOCR):
    def __init__(self, **kwargs):
        super().__init__(model_name="NewEngine", **kwargs)
        # Initialize your engine
    
    def _extract_text(self, file_path: Path, **kwargs) -> tuple[str, Optional[float], Dict[str, Any]]:
        # Your implementation here
        text = "extracted text"
        confidence = 0.95  # Optional
        metadata = {"custom": "data"}
        return text, confidence, metadata
```

3. Register in `main.py`:

```python
def _get_available_models(self) -> Dict[str, type]:
    return {
        # ... existing models ...
        'newengine': NewEngineOCR
    }
```

### Custom Preprocessing

Each OCR engine supports custom preprocessing:

```python
from ocr_tesseract import TesseractOCR

engine = TesseractOCR()
result = engine.process(
    "image.png",
    preprocess_methods=['grayscale', 'denoise', 'threshold', 'deskew']
)
```

## 📁 Output Structure

```
outputs/
├── local_ocr/          # OCR results
│   ├── document_tesseract_20250127_143022.txt
│   ├── document_tesseract_20250127_143022_metadata.json
│   └── ...
├── comparisons/        # Model comparisons
│   └── document_comparison_20250127_143022.json
└── benchmarks/         # Performance metrics
    └── benchmark_20250127_143022.txt
```

## 🔄 Integration with Vector Databases

The pipeline is designed for future integration with vector databases:

```python
# Future usage example
from main import OCRPipeline
from vector_db import MilvusClient  # Future implementation

pipeline = OCRPipeline()
results = pipeline.process_file("document.pdf")

# Extract text for embedding
for model_name, result in results.items():
    embeddings = embed_text(result.text)  # Your embedding function
    milvus_client.insert(
        collection="documents",
        data={
            "text": result.text,
            "embedding": embeddings,
            "metadata": result.to_dict()
        }
    )
```

## 🧪 Testing

Run the comprehensive test suite:

```bash
# Basic test
python test_ocr.py

# Test specific models
python -c "from ocr_tesseract import TesseractOCR; t = TesseractOCR(); print(t.process('test_image.png'))"

# Benchmark test
python main.py ./test_images --benchmark --models tesseract easyocr paddleocr
```

## 📊 Example Output

```
╔════════════════════════════════════════════════════════════════╗
║ OCR Result: Tesseract                                         ║
╠════════════════════════════════════════════════════════════════╣
║ Processing Time: 0.234s
║ Memory Usage: 45.23 MB
║ Word Count: 1,245
║ Character Count: 7,823
║ Chars/Second: 33,449
║ Confidence: 0.92
║ Errors: 0
║ Text Preview: OCR Pipeline Test Document. This is a test...
╚════════════════════════════════════════════════════════════════╝

OCR MODEL COMPARISON
================================================================================
Model                Time (s)     Memory (MB)  Words      Chars/Sec   
--------------------------------------------------------------------------------
tesseract            0.234        45.23        1,245      33,449      
easyocr              1.456        234.12       1,251      5,371       
paddleocr            0.567        89.45        1,243      13,799      
================================================================================

🏆 Fastest: tesseract (0.234s)
🐢 Slowest: easyocr (1.456s)
⚡ Speed Difference: 6.2x
```

## 🎯 Future Roadmap

### Phase 1: Local OCR (Current) ✅
- [x] Tesseract integration
- [x] EasyOCR integration  
- [x] PaddleOCR integration
- [x] Benchmarking framework
- [x] Result standardization

### Phase 2: Text Parsers (Next)
- [ ] PDFPlumber for layout analysis
- [ ] PyPDF2 for direct text extraction
- [ ] PDFMiner for detailed parsing
- [ ] Textract for multiple formats
- [ ] python-docx for Word documents

### Phase 3: Cloud OCR
- [ ] Azure Computer Vision
- [ ] Google Cloud Vision
- [ ] AWS Textract
- [ ] Donut (Document Understanding Transformer)

### Phase 4: Vision Models
- [ ] LLaMA vision integration
- [ ] GPT-4V integration
- [ ] CLIP embeddings
- [ ] Layout understanding models

### Phase 5: Multimodal Pipeline
- [ ] Combined text + image embeddings
- [ ] Milvus vector database integration
- [ ] Semantic search optimization
- [ ] Performance evaluation dashboard

## 🤝 Contributing

1. Follow the existing architecture pattern
2. Inherit from `BaseOCR` for new engines
3. Ensure all metrics are tracked
4. Add comprehensive error handling
5. Update this README with new features

## 📄 License

MIT License - Feel free to use in your biotech/pharma projects!

## 🆘 Troubleshooting

### Common Issues:

**Issue**: Tesseract not found
```bash
# Solution: Install Tesseract
sudo apt-get install tesseract-ocr
```

**Issue**: GPU not detected for EasyOCR/PaddleOCR
```bash
# Solution: Install CUDA-enabled PyTorch
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

**Issue**: PaddleOCR download fails
```bash
# Solution: Manual model download
wget https://paddleocr.bj.bcebos.com/PP-OCRv4/english/en_PP-OCRv4_det_infer.tar
```

**Issue**: Memory errors with large images
```python
# Solution: Resize before processing
from PIL import Image
img = Image.open("large_image.jpg")
img.thumbnail((2000, 2000))  # Max 2000px dimension
img.save("resized.jpg")
```

## 📚 References

- [Tesseract Documentation](https://github.com/tesseract-ocr/tesseract)
- [EasyOCR GitHub](https://github.com/JaidedAI/EasyOCR)
- [PaddleOCR GitHub](https://github.com/PaddlePaddle/PaddleOCR)
- [Multimodal Embeddings Research](https://arxiv.org/abs/2103.00020)
