# OCR Engine Comparison Project

## Overview
This project compares Tesseract and EasyOCR performance on medical/pharmaceutical documents, including both JPG images and PDF files.

## Quick Start

### Prerequisites
```bash
pip install easyocr torch torchvision pdf2image pillow pytesseract
```

### Windows Users - Install Poppler for PDF Support
1. Download Poppler: https://github.com/oschwartz10612/poppler-windows/releases/
2. Extract to C:\poppler
3. Add to PATH: `$env:PATH += ";C:\poppler\poppler-24.08.0\Library\bin"`

## Running the OCR Engines

### 1. Process Documents with EasyOCR
```python
python run_real_easyocr.py
```
This will process all documents in the wen_documents folder and save outputs to the outputs/ directory.

### 2. Process PDFs Specifically
```python
python process_pdfs_fixed.py
```

## Viewing Results

### Option 1: Basic Comparison Dashboard
Open `ocr_comparison_dashboard.html` in any browser to see:
- Performance metrics
- Processing times
- Character count comparisons
- Sample text outputs

### Option 2: Full Analysis Dashboard
Open `ocr_full_comparison.html` in any browser for:
- 7 different analysis sections
- Full text comparisons
- Visual charts
- Recommendations

### Direct Browser Opening (Windows)
```powershell
Start-Process "ocr_comparison_dashboard.html"
# or
Start-Process "ocr_full_comparison.html"
```

## Project Structure
```
OCR_Project/
 wen_documents/          # Source documents
    posters/           # JPG files
    clinical_trials/   # PDF files
    presentations/     # PDF files
 outputs/               # OCR results
    *_tesseract_*.txt
    *_easyocr_*.txt
 local_ocr/            # Additional outputs
 Biotech_Model_Test/   # Test files
 *.html                # Comparison dashboards
```

## Results Summary

| Document | Format | Tesseract | EasyOCR | Winner |
|----------|--------|-----------|---------|--------|
| 3102 Phase 2 | JPG | 17,236 chars in 26s  | 14,898 chars in 151s | Tesseract (speed) |
| 7023 Axi-Cel | JPG | 12,494 chars in 17s | 13,540 chars in 99s  | EasyOCR (accuracy) |
| ICML25 P229 | PDF | Failed  | 16,068 chars in 161s  | EasyOCR |
| Pharma Day | PDF (168pg) | Failed  | Processing... | TBD |

## Key Findings
- **Tesseract**: 6x faster on images, but cannot process PDFs directly
- **EasyOCR**: Slower but more versatile, handles all formats with Poppler
- **Recommendation**: Use Tesseract for quick image processing, EasyOCR for PDFs and accuracy

## Running Individual Scripts

### Check outputs
```powershell
Get-ChildItem outputs/*.txt | Select-Object Name, Length
```

### View first 20 lines of any output
```powershell
Get-Content outputs/[filename].txt -Head 20
```

## Troubleshooting

### If PDFs fail to process
- Ensure Poppler is installed and in PATH
- Check with: `pdftoppm -h`

### If EasyOCR is slow
- Processing uses CPU by default
- GPU acceleration available with CUDA
- Expect ~2-3 minutes per image on CPU

## Contact
Created by hd714 for OCR comparison analysis
