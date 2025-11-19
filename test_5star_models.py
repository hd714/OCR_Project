"""
Test Script for All 5-Star OCR Models
Tests TrOCR, LayoutLMv3, Nougat, GOT-OCR2.0, and Surya
Can run in mock mode without GPU for testing integration
"""

import sys
from pathlib import Path
import time
import traceback
from typing import List, Dict, Any

# Add Biotech_Model_Test to path for imports
sys.path.insert(0, str(Path(__file__).parent / "Biotech_Model_Test"))

# Import base classes
from base_ocr import BaseOCR, OCRResult, OCRBenchmarker

# Import all new OCR models
from src.local_ocr.ocr_trocr import TrOCR, TrOCRLarge, TrOCRHandwritten
from src.local_ocr.ocr_layoutlm import LayoutLMv3OCR, LayoutLMv3Large, LayoutLMv3Forms, LayoutLMv3Tables
from src.local_ocr.ocr_nougat import NougatOCR, NougatSmall, NougatLarge
from src.local_ocr.ocr_got import GOTOCR, GOTOCRFormat, GOTOCRMultiPage
from src.local_ocr.ocr_surya import SuryaOCR, SuryaMultilingual, SuryaLineLevel

# Import existing models for comparison
try:
    from src.local_ocr.ocr_tesseract import TesseractOCR
except:
    TesseractOCR = None
try:
    from src.local_ocr.ocr_easyocr import EasyOCROCR
except:
    EasyOCROCR = None
try:
    from src.local_ocr.ocr_deepseek import DeepSeekOCR
except:
    DeepSeekOCR = None


def print_banner(text: str):
    """Print a formatted banner"""
    print("\n" + "="*80)
    print(f"  {text}")
    print("="*80)


def test_single_model(model_class, model_name: str, test_image: Path, **kwargs) -> OCRResult:
    """Test a single OCR model"""
    print(f"\n[{model_name}] Initializing...")

    try:
        # Initialize model
        model = model_class(**kwargs)

        # Process image
        print(f"[{model_name}] Processing {test_image.name}...")
        result = model.process(test_image)

        # Print results
        if result.errors:
            print(f"[{model_name}] ❌ Errors: {result.errors}")
        else:
            print(f"[{model_name}] ✅ Success!")
            print(f"  - Time: {result.processing_time:.3f}s")
            print(f"  - Characters: {result.char_count:,}")
            print(f"  - Words: {result.word_count:,}")
            print(f"  - Confidence: {result.confidence:.2f}" if result.confidence else "  - Confidence: N/A")
            print(f"  - Memory: {result.memory_usage_mb:.2f} MB")

            # Show text preview
            preview = result.text[:200].replace('\n', ' ')
            if len(preview) > 200:
                preview = preview[:197] + "..."
            print(f"  - Preview: {preview}")

        return result

    except Exception as e:
        print(f"[{model_name}] ❌ Failed to initialize: {str(e)[:100]}")
        traceback.print_exc()
        return OCRResult(
            model_name=model_name,
            errors=[str(e)],
            processing_time=0.0
        )


def run_all_tests(test_image: Path, use_gpu: bool = False, mock_mode: bool = True):
    """Run tests on all 5-star models"""

    print_banner("🌟 5-STAR OCR MODEL TEST SUITE 🌟")
    print(f"Test Image: {test_image}")
    print(f"GPU Mode: {'Enabled' if use_gpu else 'Disabled'}")
    print(f"Mock Mode: {'Yes (Testing Integration)' if mock_mode else 'No (Real Inference)'}")

    # Benchmarker to compare results
    benchmarker = OCRBenchmarker()

    # Define models to test
    models_to_test = [
        # TrOCR Family
        ("TrOCR-Base", TrOCR, {"model_variant": "base", "use_gpu": use_gpu}),
        ("TrOCR-Large", TrOCRLarge, {"use_gpu": use_gpu}),
        ("TrOCR-Handwritten", TrOCRHandwritten, {"use_gpu": use_gpu}),

        # LayoutLMv3 Family
        ("LayoutLMv3-Base", LayoutLMv3OCR, {"model_size": "base", "use_gpu": use_gpu}),
        ("LayoutLMv3-Forms", LayoutLMv3Forms, {"use_gpu": use_gpu}),
        ("LayoutLMv3-Tables", LayoutLMv3Tables, {"use_gpu": use_gpu}),

        # Nougat Family
        ("Nougat-Base", NougatOCR, {"model_size": "base", "use_gpu": use_gpu}),
        ("Nougat-Small", NougatSmall, {"use_gpu": use_gpu}),

        # GOT-OCR2.0 Family
        ("GOT-OCR2.0", GOTOCR, {"use_gpu": use_gpu}),
        ("GOT-OCR2.0-Format", GOTOCRFormat, {"use_gpu": use_gpu}),

        # Surya Family
        ("Surya-OCR", SuryaOCR, {"use_gpu": use_gpu}),
        ("Surya-Multilingual", SuryaMultilingual, {"use_gpu": use_gpu}),
    ]

    # Add existing models for comparison if available
    if TesseractOCR:
        models_to_test.insert(0, ("Tesseract", TesseractOCR, {}))
    if EasyOCROCR:
        models_to_test.insert(0, ("EasyOCR", EasyOCROCR, {"use_gpu": use_gpu}))
    if DeepSeekOCR:
        models_to_test.insert(0, ("DeepSeek-OCR", DeepSeekOCR, {"use_gpu": use_gpu}))

    # Test each model
    results = {}
    for model_name, model_class, kwargs in models_to_test:
        print_banner(f"Testing {model_name}")
        result = test_single_model(model_class, model_name, test_image, **kwargs)
        results[model_name] = result
        benchmarker.add_result(result)

        # Small delay between models
        time.sleep(0.5)

    # Print comparison
    print_banner("📊 BENCHMARK RESULTS")
    print(benchmarker.get_comparison_table())

    # Save results
    output_file = Path(f"5star_models_results_{time.strftime('%Y%m%d_%H%M%S')}.json")
    benchmarker.save_results(output_file)
    print(f"\n✅ Results saved to: {output_file}")

    # Print summary
    print_banner("🏆 TOP PERFORMERS")

    # Sort by processing time
    sorted_by_speed = sorted(
        [(name, r) for name, r in results.items() if not r.errors],
        key=lambda x: x[1].processing_time
    )

    if sorted_by_speed:
        print("\n🚀 FASTEST:")
        for i, (name, result) in enumerate(sorted_by_speed[:3], 1):
            print(f"  {i}. {name}: {result.processing_time:.3f}s")

    # Sort by confidence (if available)
    sorted_by_confidence = sorted(
        [(name, r) for name, r in results.items() if not r.errors and r.confidence],
        key=lambda x: x[1].confidence,
        reverse=True
    )

    if sorted_by_confidence:
        print("\n🎯 MOST CONFIDENT:")
        for i, (name, result) in enumerate(sorted_by_confidence[:3], 1):
            print(f"  {i}. {name}: {result.confidence:.2%}")

    # Models with errors
    failed_models = [name for name, r in results.items() if r.errors]
    if failed_models:
        print(f"\n⚠️ MODELS WITH ERRORS: {', '.join(failed_models)}")

    return results


def test_specific_model(model_name: str, test_image: Path, use_gpu: bool = False):
    """Test a specific model in detail"""

    print_banner(f"Detailed Test: {model_name}")

    model_map = {
        "trocr": TrOCR,
        "layoutlm": LayoutLMv3OCR,
        "nougat": NougatOCR,
        "got": GOTOCR,
        "surya": SuryaOCR,
    }

    model_lower = model_name.lower()
    model_class = None

    for key, cls in model_map.items():
        if key in model_lower:
            model_class = cls
            break

    if not model_class:
        print(f"❌ Model '{model_name}' not found!")
        print(f"Available: {', '.join(model_map.keys())}")
        return

    # Test the model
    result = test_single_model(model_class, model_name, test_image, use_gpu=use_gpu)

    # Print detailed results
    if not result.errors:
        print(f"\n📝 FULL TEXT OUTPUT ({len(result.text)} chars):")
        print("-" * 40)
        print(result.text[:1000])  # First 1000 chars
        if len(result.text) > 1000:
            print(f"\n... ({len(result.text) - 1000} more characters)")
        print("-" * 40)

        # Print metadata
        if result.metadata:
            print("\n📊 METADATA:")
            for key, value in result.metadata.items():
                print(f"  - {key}: {value}")


def find_test_images():
    """Find available test images"""
    test_dirs = [
        Path("mock_data"),
        Path("data"),
        Path("test_images"),
        Path("Biotech_Model_Test/mock_data"),
    ]

    image_extensions = ['.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.pdf']
    test_images = []

    for test_dir in test_dirs:
        if test_dir.exists():
            for ext in image_extensions:
                test_images.extend(test_dir.glob(f"*{ext}"))

    return test_images


def main():
    """Main test function"""
    import argparse

    parser = argparse.ArgumentParser(description="Test 5-Star OCR Models")
    parser.add_argument("--image", type=str, help="Path to test image")
    parser.add_argument("--gpu", action="store_true", help="Use GPU if available")
    parser.add_argument("--real", action="store_true", help="Use real inference (not mock)")
    parser.add_argument("--model", type=str, help="Test specific model")
    parser.add_argument("--list", action="store_true", help="List available test images")

    args = parser.parse_args()

    # List available images
    if args.list:
        print_banner("Available Test Images")
        test_images = find_test_images()
        if test_images:
            for img in test_images:
                print(f"  - {img}")
        else:
            print("No test images found!")
        return

    # Find test image
    if args.image:
        test_image = Path(args.image)
        if not test_image.exists():
            print(f"❌ Image not found: {test_image}")
            return
    else:
        # Try to find a test image
        test_images = find_test_images()
        if test_images:
            test_image = test_images[0]
            print(f"📸 Using test image: {test_image}")
        else:
            print("❌ No test images found!")
            print("Please specify an image with --image <path>")
            print("Or create a 'mock_data' folder with test images")
            return

    # Run tests
    if args.model:
        # Test specific model
        test_specific_model(args.model, test_image, use_gpu=args.gpu)
    else:
        # Test all models
        mock_mode = not args.real
        run_all_tests(test_image, use_gpu=args.gpu, mock_mode=mock_mode)

    print("\n✅ Testing complete!")


if __name__ == "__main__":
    main()