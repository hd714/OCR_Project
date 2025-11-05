# DeepSeek-OCR Integration Guide

## Overview

DeepSeek-OCR is a state-of-the-art vision-language model for OCR tasks, particularly effective for:
- Medical/scientific documents with complex layouts
- Tables and structured data extraction
- Multi-language content
- Mathematical formulas
- High-accuracy text extraction from posters and clinical trial documents

## GPU Requirements

**IMPORTANT: DeepSeek-OCR requires a CUDA-capable GPU for production use.**

### Minimum Requirements:
- **GPU**: NVIDIA GPU with CUDA support
- **VRAM**: At least 12GB (16GB+ recommended)
- **CUDA**: Version 11.8 or higher
- **Compute Capability**: 7.0 or higher (e.g., RTX 2060 or newer, Tesla V100, A100)

### System Requirements:
- **OS**: Linux (recommended) or Windows 10/11 with WSL2
- **RAM**: 32GB+ system RAM recommended
- **Storage**: 15GB+ free space for model weights

## Installation Steps for GPU Systems

### 1. Install CUDA and cuDNN
Follow NVIDIA's official installation guide for your OS:
https://developer.nvidia.com/cuda-downloads

### 2. Install PyTorch with CUDA support
```bash
# For CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### 3. Install Required Dependencies
```bash
pip install transformers==4.46.3
pip install tokenizers==0.20.3
pip install einops
pip install addict
pip install easydict
pip install Pillow
```

### 4. (Optional) Install Flash Attention for Better Performance
```bash
# This significantly speeds up inference but requires compatible GPU
pip install flash-attn==2.7.3
```

**Note**: Flash Attention installation may fail on Windows. It's optional but recommended for optimal performance on Linux.

## Usage

### Basic Usage

```python
from Biotech_Model_Test.src.local_ocr import DeepSeekOCR

# Initialize with GPU
deepseek = DeepSeekOCR(use_gpu=True)

# Process an image
result = deepseek.process("document.jpg")

print(f"Extracted text: {result.text}")
print(f"Word count: {result.word_count}")
print(f"Processing time: {result.processing_time:.2f}s")
```

### Advanced Usage with Custom Prompts

```python
from Biotech_Model_Test.src.local_ocr import DeepSeekOCRAdvanced

# The advanced version automatically selects prompts based on filename
deepseek = DeepSeekOCRAdvanced(use_gpu=True)

# For clinical trial documents
result = deepseek.process("clinical_trial_poster.jpg")

# For medical posters (automatically detected from filename)
result = deepseek.process("3102_phase2_dose_optimization.jpg")
```

### Using in Pipeline

```bash
# Single model
python Biotech_Model_Test/main.py document.jpg --models deepseek

# Compare with other OCR engines
python Biotech_Model_Test/main.py document.jpg --models tesseract easyocr paddleocr deepseek

# Boss pipeline with HTML report
python boss_pipeline_enhanced.py document.jpg
```

## Performance Expectations

### GPU Mode (Production):
- **Processing Time**: 5-15 seconds per page (depending on complexity)
- **Accuracy**: 95-99% on medical/scientific documents
- **Memory Usage**: ~10-12GB VRAM
- **Output Quality**: Full text extraction with table structure preserved

### CPU/Mock Mode (Development):
- **Processing Time**: Instant (returns mock data)
- **Purpose**: Testing pipeline integration without GPU
- **Output**: Sample text for demonstration
- **Note**: Not suitable for production use

## Model Details

- **Model Name**: `deepseek-ai/DeepSeek-OCR`
- **Model Size**: ~10GB
- **Architecture**: Vision-Language Transformer
- **First Run**: Model downloads automatically from HuggingFace (may take 10-30 minutes)
- **Subsequent Runs**: Model loads from cache (~30-60 seconds)

## Prompting Strategies

DeepSeek-OCR supports custom prompts for domain-specific extraction:

### Medical Posters:
```python
prompt = """Extract all text from this medical poster.
Include:
- Title and authors
- Study objectives and methods
- Results and efficacy data
- Tables and figures
- Statistical significance (p-values)
- Conclusions
Maintain the hierarchical structure."""
```

### Clinical Trial Documents:
```python
prompt = """Extract all text from this clinical trial document.
Pay special attention to:
- Drug names and dosages (e.g., HUMIRA, KEYTRUDA)
- Efficacy percentages and p-values
- Tables with adverse events
- Patient demographics and statistics
- ACR20/ACR50 response rates
Preserve all numbers and maintain table structure."""
```

### General OCR (Default):
```python
prompt = "Free OCR. "
# This uses the model's default OCR mode
```

## Troubleshooting

### Issue: "No GPU detected - falling back to mock mode"
**Solution**:
- Verify CUDA is installed: `nvidia-smi`
- Check PyTorch can see GPU: `python -c "import torch; print(torch.cuda.is_available())"`
- Reinstall PyTorch with CUDA support

### Issue: "CUDA out of memory"
**Solutions**:
1. Close other GPU-intensive applications
2. Process one image at a time (avoid parallel processing)
3. Upgrade to GPU with more VRAM
4. Use 8-bit quantization (experimental):
   ```python
   deepseek = DeepSeekOCR(use_gpu=True, load_in_8bit=True)
   ```

### Issue: Model download is too slow
**Solution**:
- Use HuggingFace mirror if in restricted regions
- Pre-download model: `huggingface-cli download deepseek-ai/DeepSeek-OCR`

### Issue: Flash Attention installation fails
**Solution**:
- Flash Attention is optional - the model works without it
- On Windows, Flash Attention is difficult to install - skip it
- On Linux, ensure you have CUDA toolkit and nvcc installed

## Comparison with Other OCR Engines

| Feature | Tesseract | EasyOCR | PaddleOCR | DeepSeek-OCR |
|---------|-----------|---------|-----------|--------------|
| Speed | Fast | Medium | Medium | Slower |
| Accuracy | Good | Very Good | Very Good | Excellent |
| Tables | Poor | Fair | Good | Excellent |
| Medical Terms | Fair | Good | Good | Excellent |
| GPU Required | No | No | No | Yes |
| Setup | Easy | Easy | Easy | Complex |
| Best For | Quick scans | General use | Asian languages | Critical documents |

## When to Use DeepSeek-OCR

**Use DeepSeek-OCR when**:
- Maximum accuracy is critical
- Document contains complex tables
- Medical/scientific terminology must be preserved
- Document layout is complex (multi-column, posters)
- You have access to a GPU

**Use Tesseract/EasyOCR/PaddleOCR when**:
- No GPU available
- Speed is more important than accuracy
- Simple document layouts
- Batch processing on CPU

## Files Modified for Integration

1. `Biotech_Model_Test/src/local_ocr/ocr_deepseek.py` - Main implementation
2. `Biotech_Model_Test/src/local_ocr/__init__.py` - Exports DeepSeek classes
3. `boss_pipeline_enhanced.py` - Integrated into HTML reporting
4. `Biotech_Model_Test/main.py` - Added to model selection

## Environment Variables (Optional)

```bash
# Set HuggingFace cache directory
export HF_HOME=/path/to/large/storage

# Disable HuggingFace telemetry
export HF_HUB_DISABLE_TELEMETRY=1

# Use offline mode (after model downloaded)
export TRANSFORMERS_OFFLINE=1
```

## License & Attribution

DeepSeek-OCR is developed by DeepSeek AI and released under their license.
Please review the model card for usage restrictions and citations:
https://huggingface.co/deepseek-ai/DeepSeek-OCR

## Support

For issues specific to:
- **DeepSeek model**: https://github.com/deepseek-ai/DeepSeek-OCR
- **This integration**: Contact the development team
- **GPU setup**: Consult NVIDIA CUDA documentation
