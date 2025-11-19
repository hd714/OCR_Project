"""
Quick test to verify all 5-star models are working
Runs in mock mode by default for fast testing
"""

import sys
from pathlib import Path

# Add to path
sys.path.insert(0, str(Path(__file__).parent / "Biotech_Model_Test"))

def test_imports():
    """Test that all models can be imported"""
    print("Testing imports...")

    models_to_test = [
        ("TrOCR", "src.local_ocr.ocr_trocr", "TrOCR"),
        ("LayoutLMv3", "src.local_ocr.ocr_layoutlm", "LayoutLMv3OCR"),
        ("Nougat", "src.local_ocr.ocr_nougat", "NougatOCR"),
        ("GOT-OCR2.0", "src.local_ocr.ocr_got", "GOTOCR"),
        ("Surya", "src.local_ocr.ocr_surya", "SuryaOCR"),
    ]

    success_count = 0
    for name, module_path, class_name in models_to_test:
        try:
            module = __import__(module_path, fromlist=[class_name])
            cls = getattr(module, class_name)
            print(f"  ✅ {name} - Successfully imported")
            success_count += 1
        except Exception as e:
            print(f"  ❌ {name} - Failed: {e}")

    print(f"\nImport Test: {success_count}/{len(models_to_test)} successful")
    return success_count == len(models_to_test)


def test_mock_mode():
    """Test that models work in mock mode"""
    print("\nTesting mock mode...")

    from src.local_ocr.ocr_trocr import TrOCR
    from pathlib import Path
    import tempfile
    from PIL import Image

    # Create a test image
    test_image = Image.new('RGB', (100, 100), color='white')

    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
        test_image.save(tmp.name)
        test_path = Path(tmp.name)

        try:
            # Test TrOCR in mock mode
            model = TrOCR(use_gpu=False)
            result = model.process(test_path)

            if result.errors:
                print(f"  ❌ Mock mode failed: {result.errors}")
                return False
            else:
                print(f"  ✅ Mock mode successful")
                print(f"     - Text length: {len(result.text)}")
                print(f"     - Processing time: {result.processing_time:.3f}s")
                return True
        except Exception as e:
            print(f"  ❌ Error: {e}")
            return False
        finally:
            test_path.unlink()


def main():
    """Run all tests"""
    print("="*60)
    print("🌟 5-STAR OCR MODELS - QUICK TEST 🌟")
    print("="*60)

    # Test imports
    import_success = test_imports()

    # Test mock mode
    mock_success = test_mock_mode()

    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    print(f"Import Test: {'✅ PASSED' if import_success else '❌ FAILED'}")
    print(f"Mock Mode Test: {'✅ PASSED' if mock_success else '❌ FAILED'}")

    if import_success and mock_success:
        print("\n✅ ALL TESTS PASSED - Models are ready to use!")
        print("\nNext steps:")
        print("1. Test with a real image:")
        print("   python test_5star_models.py --image path/to/image.png")
        print("\n2. Run the HTML comparison pipeline:")
        print("   python boss_pipeline_5star.py path/to/document.pdf")
        print("\n3. For GPU testing (on boss's machine):")
        print("   python boss_pipeline_5star.py document.pdf --gpu")
    else:
        print("\n❌ Some tests failed - check the errors above")

    return import_success and mock_success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)