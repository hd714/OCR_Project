import pytesseract
from PIL import Image
from pathlib import Path
import json
from datetime import datetime

# Set Tesseract path
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

def test_all_images():
    """Test OCR on all image files in the project"""
    
    # Find all image files
    image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff']
    all_images = []
    
    for ext in image_extensions:
        all_images.extend(Path('.').rglob(f'*{ext}'))
    
    results = {
        'tested': datetime.now().isoformat(),
        'total_files': len(all_images),
        'successful': [],
        'failed': []
    }
    
    print(f"\n{'='*60}")
    print(f"Testing {len(all_images)} image files with Tesseract")
    print(f"{'='*60}\n")
    
    for img_path in all_images:
        try:
            print(f"Testing: {img_path}")
            img = Image.open(img_path)
            text = pytesseract.image_to_string(img)
            
            results['successful'].append({
                'file': str(img_path),
                'characters': len(text),
                'words': len(text.split())
            })
            
            print(f"  SUCCESS - {len(text):,} characters extracted")
            
        except Exception as e:
            results['failed'].append({
                'file': str(img_path),
                'error': str(e)
            })
            print(f"  FAILED - {e}")
    
    # Save results
    with open('tesseract_test_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"SUMMARY:")
    print(f"  Total files tested: {results['total_files']}")
    print(f"  Successful: {len(results['successful'])}")
    print(f"  Failed: {len(results['failed'])}")
    print(f"  Results saved to: tesseract_test_results.json")
    print(f"{'='*60}\n")
    
    return results

if __name__ == "__main__":
    test_all_images()
