from pathlib import Path

print("Testing Boss Pipeline Setup...")
print("="*50)

files_to_check = [
    "boss_pipeline.py",
    "Biotech_Model_Test/base_ocr.py",
]

for file in files_to_check:
    if Path(file).exists():
        print(f"✅ Found: {file}")
    else:
        print(f"❌ Missing: {file}")
