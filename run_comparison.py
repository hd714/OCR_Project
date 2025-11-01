import sys
from pathlib import Path
import subprocess

print("OCR COMPARISON LAUNCHER")
print("1. Compare test file")
print("2. Compare all posters")
print("3. Enter custom files")

choice = input("Choice (1-3): ")

if choice == "1":
    subprocess.run('python ultimate_ocr_comparison.py "wen_documents/posters/3102_phase2_dose_optimization.jpg"', shell=True)
elif choice == "2":
    files = list(Path("wen_documents/posters").glob("*.jpg"))[:5]
    if files:
        files_str = ' '.join([f'"{f}"' for f in files])
        subprocess.run(f'python ultimate_ocr_comparison.py {files_str}', shell=True)
elif choice == "3":
    file = input("Enter image path: ")
    subprocess.run(f'python ultimate_ocr_comparison.py "{file}"', shell=True)
