# OCR Project - Complete Models and Libraries Summary

## Project Overview
This is a comprehensive OCR pipeline project designed for medical/pharmaceutical document processing. 
It integrates multiple local OCR engines, cloud-based services, and vision-language models.

---

## LOCAL OCR MODELS

### 1. **Tesseract OCR**
- Library: pytesseract (>= 0.3.10)
- Classes: TesseractOCR, TesseractAdvancedOCR
- Speed: Fast baseline
- Features: Preprocessing, multiple PSM support, confidence scoring
- Path: C:\Program Files\Tesseract-OCR	esseract.exe (Windows)

### 2. **EasyOCR**
- Library: easyocr (>= 1.7.1)
- Dependencies: PyTorch (torch, torchvision)
- Classes: EasyOCROCR, EasyOCRMultilingual
- Models: Auto-downloads ~64MB on first run
- Storage: ~/.EasyOCR/model/
- Features: Neural network-based, GPU support, CRAFT+CRNN
- Languages: English (configurable)

### 3. **PaddleOCR**
- Libraries: paddleocr (>= 3.3.0), paddlepaddle (>= 2.6.1)
- Classes: PaddleOCROCR, PaddleOCRAdvanced
- Algorithm: DBNet detection + SVTR_LCNet recognition
- Models: Auto-downloads on first run
- Storage: ~/.paddleocr/
- Features: Angle classification, multi-language support, efficient inference
- Optimized for: Chinese and Asian languages

### 4. **DeepSeek-OCR**
- Model: deepseek-ai/DeepSeek-OCR (HuggingFace)
- Libraries: transformers, torch
- Model Size: ~10GB
- Architecture: Vision-Language Transformer
- Classes: DeepSeekOCR, DeepSeekOCRAdvanced
- Storage: ~/.cache/huggingface/hub/
- GPU Required: Yes (12GB+ VRAM minimum for production)
- Features: State-of-the-art, custom prompting, bfloat16/float16 support
- Fallback: Mock mode on CPU-only systems
- Performance: 5-15 seconds/page on GPU, 95-99% accuracy

---

## CLOUD OCR SERVICES

### 1. **Azure Document Intelligence**
- Library: azure-ai-formrecognizer (>= 3.3.0)
- Class: AzureOCR
- API Endpoint: https://biotech-ocr.cognitiveservices.azure.com/
- Models Available: prebuilt-read, prebuilt-document, prebuilt-invoice, prebuilt-receipt, prebuilt-idcard
- Features: Table extraction, key-value pairs, per-line confidence, document classification
- Authentication: API Key via environment variable

### 2. **Donut (Document Understanding Transformer)**
- Model: naver-clova-ix/donut-base-finetuned-docvqa (HuggingFace)
- Library: transformers, torch
- Class: DonutOCR
- Architecture: Vision encoder-decoder
- Features: End-to-end document understanding, receipt parsing, document Q&A
- Variants: DocVQA, CORD-v2, RVLCDIP

---

## VISION-LANGUAGE MODELS

### 1. **GPT-4 Vision**
- Model: gpt-4-vision-preview (OpenAI API)
- Library: openai (>= 1.0.0)
- Class: GPT4VisionOCR
- Features: Advanced document understanding, table structure preservation
- Authentication: OPENAI_API_KEY environment variable

### 2. **BLIP-2 (Vision-Language Pre-training)**
- Model: Salesforce/blip2-opt-2.7b (HuggingFace)
- Libraries: transformers, torch, bitsandbytes
- Class: BLIP2VisionOCR
- Features: 8-bit quantization, float16 GPU precision, vision-to-text generation
- Size: 2.7B parameters

### 3. **LLaMA Vision**
- Class: LLaMAVisionOCR
- Status: Vision-language model implementation

### 4. **CLIP**
- Library: transformers
- Class: CLIPVisionEmbedder
- Features: Image embedding generation, vision-text alignment

---

## KEY DEPENDENCIES

OCR Engines:
- pytesseract >= 0.3.10
- easyocr >= 1.7.1
- paddlepaddle >= 2.6.1
- paddleocr >= 3.3.0

Deep Learning:
- torch >= 2.0.1
- torchvision >= 0.15.2
- transformers >= 4.36.0
- bitsandbytes >= 0.41.0
- accelerate >= 0.25.0

Cloud Services:
- azure-ai-formrecognizer >= 3.3.0
- azure-core >= 1.29.0
- openai >= 1.0.0

PDF Processing:
- pdf2image >= 1.17.0
- PyPDF2 >= 3.0.1
- pdfplumber >= 0.10.0

Core:
- numpy >= 1.23.0
- opencv-python >= 4.8.0
- Pillow >= 10.0.0
- rich >= 13.6.0

Utilities:
- pandas >= 2.1.0
- matplotlib >= 3.8.0
- tqdm >= 4.66.1
- pymilvus >= 2.3.0

---

## PROJECT STRUCTURE

Biotech_Model_Test/
  src/
    local_ocr/           # Local OCR implementations
      ocr_tesseract.py
      ocr_easyocr.py
      ocr_paddleocr.py
      ocr_deepseek.py
    cloud_ocr/           # Cloud OCR services
      ocr_azure.py
      ocr_donut.py
    vision_models/       # Vision-language models
      vision_gpt4.py
      vision_blip2.py
      vision_llama.py
      vision_clip.py
    text_parsers/
  base_ocr.py            # Base class and benchmarking
  main.py                # Main pipeline coordinator
  test_ocr_minimal.py    # Minimal test suite

---

## MODEL INITIALIZATION

BaseOCR Class:
- Abstract base class for all OCR implementations
- Built-in benchmarking and metrics tracking
- Result caching with MD5 file hashing
- Logging integration
- OCRResult: Standardized output format
- OCRBenchmarker: Model comparison framework

---

## CONFIGURATION

Environment Variables:
- AZURE_KEY: Azure Document Intelligence API key
- AZURE_ENDPOINT: Azure service endpoint
- OPENAI_API_KEY: OpenAI GPT-4V API key
- HF_HOME: HuggingFace cache directory (optional)

.env file contains:
- AZURE_KEY configuration for cloud services

---

## DEPLOYMENT NOTES

1. GPU Systems: DeepSeek-OCR automatically uses GPU if CUDA available
2. CPU Systems: All models work on CPU; DeepSeek uses mock mode
3. First Run: May take 10-30 minutes for model downloads
4. Model Storage: ~10GB for DeepSeek-OCR, ~64MB for EasyOCR, auto-managed
5. Tesseract: Requires separate system installation (Windows: Program Files)
6. Memory Requirements: 12GB+ VRAM for GPU models, 10GB+ storage for weights

---

## QUICK COMPARISON

| Model | Type | Speed | Accuracy | GPU | Size |
|-------|------|-------|----------|-----|------|
| Tesseract | Local | Very Fast | Good | No | Binary |
| EasyOCR | Local | Medium | Very Good | Yes | 64MB |
| PaddleOCR | Local | Medium | Very Good | Yes | Auto-DL |
| DeepSeek | Local | Slower | Excellent | Yes* | 10GB |
| Azure | Cloud | Medium | Excellent | N/A | N/A |
| GPT-4V | Cloud | Slow | Excellent | N/A | N/A |
| Donut | Local | Medium | Good | Yes | Auto-DL |
| BLIP-2 | Local | Medium | Good | Yes | 2.7B |

*DeepSeek: GPU required for production, mock mode on CPU

---

## TOTAL STATISTICS

- Local OCR Models: 4 (Tesseract, EasyOCR, PaddleOCR, DeepSeek-OCR)
- Cloud Services: 2 (Azure, OpenAI)
- Vision Models: 4 (GPT-4V, BLIP-2, LLaMA, CLIP)
- Total Implementations: 14+ classes
- Dependencies: 30+ Python packages
- GPU-Capable Models: 6
- Auto-Downloading Models: 5 (EasyOCR, PaddleOCR, DeepSeek, Donut, BLIP-2)
