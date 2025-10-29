"""
Debug script to identify OCR issues
"""
import sys
from pathlib import Path
import traceback
import numpy as np
import cv2

# Add paths
here = Path(__file__).resolve().parent
sys.path.insert(0, str(here))
sys.path.insert(0, str(here / "src"))

from rich.console import Console
console = Console()

def create_simple_test_image():
    """Create a very simple test image"""
    width, height = 800, 200
    image = np.ones((height, width, 3), dtype=np.uint8) * 255
    
    # Add simple black text
    cv2.putText(image, "Hello World Test 123", (50, 100), 
                cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 3, cv2.LINE_AA)
    
    # Save as multiple formats
    cv2.imwrite("simple.png", image)
    cv2.imwrite("simple.jpg", image)
    
    return image

def debug_easyocr():
    console.print("\n[yellow]Debugging EasyOCR...[/yellow]")
    
    try:
        import easyocr
        console.print("[green]✓ EasyOCR imported successfully[/green]")

        # Create reader with verbose output
        console.print("Creating EasyOCR reader (this may download models on first run)...")
        reader = easyocr.Reader(['en'], gpu=False, verbose=True)
        console.print("[green]✓ Reader created[/green]")
        
        # Test with simple image
        image = create_simple_test_image()
        
        # Try different methods
        console.print("\nTrying different input methods:")
        
        # Method 1: Direct numpy array
        try:
            result = reader.readtext(image, detail=0)
            console.print(f"  Numpy array: {result}")
        except Exception as e:
            console.print(f"  [red]Numpy array failed: {e}[/red]")
        
        # Method 2: File path
        try:
            result = reader.readtext('simple.png', detail=0)
            console.print(f"  PNG file: {result}")
        except Exception as e:
            console.print(f"  [red]PNG file failed: {e}[/red]")
        
        # Method 3: With full details
        try:
            result = reader.readtext('simple.png', detail=1)
            if result:
                console.print(f"  Detailed result: Found {len(result)} text regions")
                for bbox, text, conf in result:
                    console.print(f"    Text: '{text}', Confidence: {conf:.2f}")
            else:
                console.print("  [yellow]No text detected[/yellow]")
        except Exception as e:
            console.print(f"  [red]Detailed extraction failed: {e}[/red]")
            console.print(f"  [red]Traceback: {traceback.format_exc()}[/red]")
            
    except ImportError as e:
        console.print(f"[red]Failed to import EasyOCR: {e}[/red]")
    except Exception as e:
        console.print(f"[red]Unexpected error: {e}[/red]")
        console.print(f"[red]Traceback: {traceback.format_exc()}[/red]")

def debug_paddleocr():
    console.print("\n[yellow]Debugging PaddleOCR...[/yellow]")
    
    try:
        from paddleocr import PaddleOCR
        console.print("[green]✓ PaddleOCR imported successfully[/green]")
        
        # Create OCR instance with debug info
        console.print("Creating PaddleOCR instance (this may download models on first run)...")
        ocr = PaddleOCR(
            use_angle_cls=True,
            lang='en',
            use_gpu=False,
            show_log=True  # Enable logging
        )
        console.print("[green]✓ PaddleOCR instance created[/green]")
        
        # Test with simple image
        image_path = 'simple.png'
        
        console.print(f"\nProcessing {image_path}...")
        result = ocr.ocr(image_path, cls=True)
        
        if result and result[0]:
            console.print(f"[green]✓ Found {len(result[0])} text regions[/green]")
            for line in result[0]:
                if line:
                    bbox, (text, confidence) = line[0], line[1]
                    console.print(f"  Text: '{text}', Confidence: {confidence:.2f}")
        else:
            console.print("[yellow]No text detected[/yellow]")
            
            # Try with different settings
            console.print("\nTrying with different settings...")
            ocr2 = PaddleOCR(
                use_angle_cls=False,
                lang='en',
                use_gpu=False,
                det_db_thresh=0.1,  # Lower detection threshold
                drop_score=0.3  # Lower drop score
            )
            result2 = ocr2.ocr(image_path, cls=False)
            if result2 and result2[0]:
                console.print(f"[green]✓ With adjusted settings: Found {len(result2[0])} regions[/green]")
            else:
                console.print("[red]Still no text detected[/red]")
                
    except ImportError as e:
        console.print(f"[red]Failed to import PaddleOCR: {e}[/red]")
        console.print("\nTry reinstalling:")
        console.print("  pip uninstall paddlepaddle paddleocr")
        console.print("  pip install paddlepaddle paddleocr")
    except Exception as e:
        console.print(f"[red]Unexpected error: {e}[/red]")
        console.print(f"[red]Traceback: {traceback.format_exc()}[/red]")

def check_dependencies():
    console.print("[yellow]Checking dependencies...[/yellow]")
    
    packages = {
        'torch': 'PyTorch',
        'torchvision': 'TorchVision',
        'paddlepaddle': 'PaddlePaddle',
        'easyocr': 'EasyOCR',
        'paddleocr': 'PaddleOCR'
    }
    
    for package, name in packages.items():
        try:
            module = __import__(package)
            version = getattr(module, '__version__', 'unknown')
            console.print(f"  [green]✓ {name}: {version}[/green]")
        except ImportError:
            console.print(f"  [red]✗ {name}: Not installed[/red]")

if __name__ == "__main__":
    console.print("[bold cyan]OCR Debug Tool[/bold cyan]\n")
    
    check_dependencies()
    debug_easyocr()
    debug_paddleocr()
    
    console.print("\n[bold]Diagnosis Summary:[/bold]")
    console.print("If models aren't downloading, check your internet connection")
    console.print("If text isn't detected, try adjusting detection thresholds")
