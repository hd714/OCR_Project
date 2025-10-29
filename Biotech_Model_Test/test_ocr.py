"""
Test script for OCR Pipeline
Run this to validate your setup
"""
import sys
from pathlib import Path

here = Path(__file__).resolve().parent
sys.path.insert(0, str(here))
sys.path.insert(0, str(here / "src"))

import traceback
from base_ocr import BaseOCR
from local_ocr.ocr_tesseract import TesseractOCR
from local_ocr.ocr_easyocr import EasyOCROCR
from local_ocr.ocr_paddleocr import PaddleOCROCR
from main import OCRPipeline

from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cv2
from rich.console import Console
from rich.panel import Panel

console = Console()

def create_test_image():
    width, height = 1200, 800
    image = np.ones((height, width, 3), dtype=np.uint8) * 255
    
    font_scale = 2.5
    thickness = 3
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    texts = [
        "OCR PIPELINE TEST IMAGE",
        "This is a test document",
        "Testing numbers: 1234567890",
        "UPPERCASE AND lowercase text",
        "Special chars: @ # $ % & *",
        "Email: test@example.com",
        "Date: January 27, 2025"
    ]
    
    y_position = 100
    for text in texts:
        cv2.putText(image, text, (50, y_position), font, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)
        y_position += 100
    
    test_path = Path("test_image.png")
    cv2.imwrite(str(test_path), image)
    console.print(f"[green]✓ Created test image: {test_path}[/green]")
    
    return test_path

def test_basic_import():
    console.print("\n[yellow]Testing module imports...[/yellow]")
    
    modules_to_test = [
        ("base_ocr", "BaseOCR"),
        ("local_ocr.ocr_tesseract", "TesseractOCR"),
        ("local_ocr.ocr_easyocr", "EasyOCROCR"),
        ("local_ocr.ocr_paddleocr", "PaddleOCROCR"),
        ("main", "OCRPipeline")
    ]
    
    all_good = True
    for module_name, class_name in modules_to_test:
        try:
            module = __import__(module_name, fromlist=[class_name])
            cls = getattr(module, class_name)
            console.print(f"  [green]✓[/green] {module_name}.{class_name}")
        except Exception as e:
            console.print(f"  [red]✗[/red] {module_name}.{class_name}: {e}")
            all_good = False
    
    return all_good

def test_ocr_engines():
    console.print("\n[yellow]Testing OCR engines...[/yellow]")
    
    test_image = create_test_image()
    tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    
    try:
        console.print("\n[cyan]Testing Tesseract...[/cyan]")
        tesseract = TesseractOCR(tesseract_cmd=tesseract_cmd, enable_logging=False, preprocess=False)
        result = tesseract.process(test_image)
        if result.word_count > 0:
            console.print(f"  [green]✓ Tesseract processed {result.word_count} words in {result.processing_time:.3f}s[/green]")
            console.print(f"    Text preview: {result.text[:80]}...")
        else:
            console.print(f"  [red]✗ Tesseract returned no text[/red]")
    except Exception as e:
        console.print(f"  [red]✗ Tesseract failed: {e}[/red]")
    
    try:
        console.print("\n[cyan]Testing EasyOCR...[/cyan]")
        easyocr = EasyOCROCR(languages=['en'], use_gpu=False, enable_logging=False)
        result = easyocr.process(test_image)
        if result.word_count > 0:
            console.print(f"  [green]✓ EasyOCR processed {result.word_count} words in {result.processing_time:.3f}s[/green]")
        else:
            console.print(f"  [yellow]⚠ EasyOCR returned no text[/yellow]")
    except Exception as e:
        console.print(f"  [yellow]⚠ EasyOCR skipped: {e}[/yellow]")
    
    try:
        console.print("\n[cyan]Testing PaddleOCR...[/cyan]")
        paddleocr = PaddleOCROCR(lang='en', use_gpu=False, enable_logging=False)
        result = paddleocr.process(test_image)
        if result.word_count > 0:
            console.print(f"  [green]✓ PaddleOCR processed {result.word_count} words in {result.processing_time:.3f}s[/green]")
        else:
            console.print(f"  [yellow]⚠ PaddleOCR returned no text[/yellow]")
    except Exception as e:
        console.print(f"  [yellow]⚠ PaddleOCR skipped: {e}[/yellow]")

def test_pipeline():
    console.print("\n[yellow]Testing main pipeline...[/yellow]")
    
    try:
        pipeline = OCRPipeline(
            models=['tesseract'],
            enable_gpu=False,
            save_full_text=True
        )
        
        test_image = Path("test_image.png")
        if test_image.exists():
            results = pipeline.process_file(test_image)
            if results:
                first_result = list(results.values())[0]
                if first_result.word_count > 0:
                    console.print(f"[green]✓ Pipeline SUCCESS - extracted {first_result.word_count} words[/green]")
                else:
                    console.print(f"[red]✗ Pipeline ran but extracted 0 words[/red]")
            else:
                console.print(f"[red]✗ Pipeline returned no results[/red]")
        else:
            console.print("[red]Test image not found[/red]")
            
    except Exception as e:
        console.print(f"[red]✗ Pipeline failed: {e}[/red]")

def main():
    console.print(Panel("[bold cyan]OCR Pipeline Test Suite[/bold cyan]"))
    
    if not test_basic_import():
        console.print("\n[red]Some imports failed. Please check your installation.[/red]")
        return
    
    test_ocr_engines()
    test_pipeline()
    
    console.print("\n[bold green]Testing complete![/bold green]")
    console.print("\nNext steps:")
    console.print("  [cyan]python main.py test_image.png[/cyan]")

if __name__ == "__main__":
    main()