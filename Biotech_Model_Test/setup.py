#!/usr/bin/env python3
"""
Setup script for Biotech OCR Pipeline
Run this to verify your environment and install dependencies
"""

import subprocess
import sys
import platform
from pathlib import Path

def run_command(cmd, check=True):
    """Run a shell command and return success status"""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, check=check)
        return True, result.stdout
    except subprocess.CalledProcessError as e:
        return False, e.stderr

def check_python_version():
    """Check if Python version is 3.8+"""
    version = sys.version_info
    if version.major == 3 and version.minor >= 8:
        print(f"✓ Python {version.major}.{version.minor}.{version.micro}")
        return True
    else:
        print(f"✗ Python {version.major}.{version.minor} (need 3.8+)")
        return False

def check_tesseract():
    """Check if Tesseract is installed"""
    success, output = run_command("tesseract --version", check=False)
    if success:
        print("✓ Tesseract installed")
        return True
    else:
        print("✗ Tesseract not found")
        print("  Installation instructions:")
        if platform.system() == "Linux":
            print("    sudo apt-get install tesseract-ocr")
        elif platform.system() == "Darwin":  # macOS
            print("    brew install tesseract")
        elif platform.system() == "Windows":
            print("    Download from: https://github.com/UB-Mannheim/tesseract/wiki")
        return False

def install_requirements():
    """Install Python requirements"""
    print("\nInstalling Python packages...")
    
    # Essential packages first
    essential = [
        "numpy",
        "opencv-python",
        "Pillow",
        "rich",
        "pytesseract"
    ]
    
    for package in essential:
        print(f"  Installing {package}...")
        success, _ = run_command(f"{sys.executable} -m pip install {package}", check=False)
        if success:
            print(f"    ✓ {package}")
        else:
            print(f"    ✗ {package} (will try with requirements.txt)")
    
    # Try full requirements
    print("\nInstalling from requirements.txt...")
    success, output = run_command(f"{sys.executable} -m pip install -r requirements.txt", check=False)
    
    if not success:
        print("  Some packages failed to install. You can install them manually:")
        print("  pip install -r requirements.txt")
    else:
        print("  ✓ All requirements installed")

def create_directories():
    """Create necessary directories"""
    dirs = [
        "outputs/local_ocr",
        "outputs/benchmarks",
        "outputs/comparisons",
        "data"
    ]
    
    print("\nCreating directories...")
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        print(f"  ✓ {dir_path}")

def test_imports():
    """Test if core modules can be imported"""
    print("\nTesting imports...")
    
    # Add src to path
    sys.path.insert(0, str(Path(__file__).parent / "src"))
    
    modules = [
        ("base_ocr", "BaseOCR"),
        ("local_ocr.ocr_tesseract", "TesseractOCR"),
    ]
    
    all_good = True
    for module_name, class_name in modules:
        try:
            module = __import__(module_name, fromlist=[class_name])
            getattr(module, class_name)
            print(f"  ✓ {module_name}")
        except Exception as e:
            print(f"  ✗ {module_name}: {e}")
            all_good = False
    
    return all_good

def main():
    print("="*60)
    print("Biotech OCR Pipeline Setup")
    print("="*60)
    
    print("\nChecking environment...")
    
    # Check Python version
    if not check_python_version():
        print("\nPlease upgrade Python to 3.8+")
        sys.exit(1)
    
    # Check Tesseract
    tesseract_ok = check_tesseract()
    
    # Create directories
    create_directories()
    
    # Install requirements
    install_requirements()
    
    # Test imports
    imports_ok = test_imports()
    
    print("\n" + "="*60)
    print("Setup Summary")
    print("="*60)
    
    if tesseract_ok and imports_ok:
        print("✓ Setup complete! You're ready to go.")
        print("\nNext steps:")
        print("  1. Test the pipeline: python test_ocr.py")
        print("  2. Process a file: python src/main.py your_image.png")
        print("  3. Run benchmark: python src/main.py ./data --benchmark")
    else:
        print("⚠ Setup incomplete. Please resolve the issues above.")
        if not tesseract_ok:
            print("\n  - Install Tesseract OCR")
        if not imports_ok:
            print("\n  - Fix Python imports (check requirements.txt)")
    
    print("\nFor GPU support with EasyOCR/PaddleOCR:")
    print("  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118")

if __name__ == "__main__":
    main()
