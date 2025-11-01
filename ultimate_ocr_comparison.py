import pytesseract
from PIL import Image
import easyocr
import time
import difflib
from pathlib import Path
from datetime import datetime
import json
import numpy as np
import webbrowser
import sys
import warnings
warnings.filterwarnings('ignore')

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

class UltimateOCRComparison:
    def __init__(self):
        print("Initializing OCR Comparison System...")
        self.results_dir = Path("comparison_results")
        self.results_dir.mkdir(exist_ok=True)
        print("Loading EasyOCR models...")
        self.easy_reader = easyocr.Reader(['en'], gpu=False)
        print("Ready to compare OCR engines!\n")
    
    def calculate_similarity(self, text1, text2):
        matcher = difflib.SequenceMatcher(None, text1, text2)
        return matcher.ratio() * 100
    
    def get_text_differences(self, text1, text2, max_diffs=5):
        """Get word-level differences instead of line-level"""
        words1 = text1.split()
        words2 = text2.split()
        
        # Use difflib to find differences at word level
        matcher = difflib.SequenceMatcher(None, words1, words2)
        differences = []
        
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == 'replace':
                tesseract_text = ' '.join(words1[i1:min(i2, i1+10)])  # Get up to 10 words
                easyocr_text = ' '.join(words2[j1:min(j2, j1+10)])
                differences.append({
                    'type': 'Different words',
                    'tesseract': tesseract_text[:100],
                    'easyocr': easyocr_text[:100]
                })
            elif tag == 'delete':
                tesseract_text = ' '.join(words1[i1:min(i2, i1+10)])
                differences.append({
                    'type': 'Only in Tesseract',
                    'tesseract': tesseract_text[:100],
                    'easyocr': '(not detected)'
                })
            elif tag == 'insert':
                easyocr_text = ' '.join(words2[j1:min(j2, j1+10)])
                differences.append({
                    'type': 'Only in EasyOCR',
                    'tesseract': '(not detected)',
                    'easyocr': easyocr_text[:100]
                })
            
            if len(differences) >= max_diffs:
                break
        
        # If no differences found, compare first 100 chars
        if not differences and text1 != text2:
            for i in range(0, min(len(text1), len(text2)), 100):
                chunk1 = text1[i:i+100]
                chunk2 = text2[i:i+100]
                if chunk1 != chunk2:
                    differences.append({
                        'type': f'Difference at char {i}',
                        'tesseract': chunk1[:50] + '...',
                        'easyocr': chunk2[:50] + '...'
                    })
                    if len(differences) >= max_diffs:
                        break
        
        return differences[:max_diffs]
    
    def process_with_tesseract(self, image_path):
        print(f"  Processing with Tesseract...")
        start_time = time.time()
        img = Image.open(image_path)
        text = pytesseract.image_to_string(img)
        data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
        confidences = [float(conf) for conf in data['conf'] if int(conf) > 0]
        avg_confidence = np.mean(confidences) if confidences else 0
        return {
            'text': text,
            'time': time.time() - start_time,
            'confidence': avg_confidence,
            'word_count': len(text.split()),
            'char_count': len(text)
        }
    
    def process_with_easyocr(self, image_path):
        print(f"  Processing with EasyOCR...")
        start_time = time.time()
        results = self.easy_reader.readtext(str(image_path))
        text_parts = []
        confidences = []
        for (bbox, text, confidence) in results:
            text_parts.append(text)
            confidences.append(confidence)
        full_text = ' '.join(text_parts)
        avg_confidence = np.mean(confidences) * 100 if confidences else 0
        return {
            'text': full_text,
            'time': time.time() - start_time,
            'confidence': avg_confidence,
            'word_count': len(full_text.split()),
            'char_count': len(full_text)
        }
    
    def process_single_image(self, image_path):
        print(f"\nProcessing: {Path(image_path).name}")
        print("=" * 50)
        tesseract_result = self.process_with_tesseract(image_path)
        easyocr_result = self.process_with_easyocr(image_path)
        similarity = self.calculate_similarity(tesseract_result['text'], easyocr_result['text'])
        differences = self.get_text_differences(tesseract_result['text'], easyocr_result['text'])
        print(f"  Tesseract: {tesseract_result['time']:.2f}s, {tesseract_result['confidence']:.1f}% confidence")
        print(f"  EasyOCR: {easyocr_result['time']:.2f}s, {easyocr_result['confidence']:.1f}% confidence")
        print(f"  Similarity: {similarity:.1f}%")
        print(f"  Found {len(differences)} key differences")
        return {
            'file_name': Path(image_path).name,
            'tesseract': tesseract_result,
            'easyocr': easyocr_result,
            'similarity': similarity,
            'differences': differences
        }
    
    def generate_html_report(self, all_results):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        html_file = self.results_dir / f"comparison_report_{timestamp}.html"
        
        total_files = len(all_results)
        avg_tesseract_time = np.mean([r['tesseract']['time'] for r in all_results])
        avg_easyocr_time = np.mean([r['easyocr']['time'] for r in all_results])
        avg_similarity = np.mean([r['similarity'] for r in all_results])
        
        html = f"""<!DOCTYPE html>
<html>
<head>
    <title>OCR Comparison Report</title>
    <style>
        body {{font-family: Arial; background: linear-gradient(135deg, #667eea, #764ba2); padding: 20px;}}
        .container {{max-width: 1200px; margin: 0 auto;}}
        .header {{background: white; border-radius: 10px; padding: 30px; margin-bottom: 20px; text-align: center;}}
        h1 {{color: #667eea; margin: 0;}}
        .stats {{display: grid; grid-template-columns: repeat(4, 1fr); gap: 15px; margin-bottom: 20px;}}
        .stat {{background: white; padding: 20px; border-radius: 10px; text-align: center;}}
        .stat-value {{font-size: 2em; color: #667eea; font-weight: bold;}}
        .stat-label {{color: #666; margin-top: 5px;}}
        .file-section {{background: white; border-radius: 10px; padding: 20px; margin-bottom: 20px;}}
        .file-header {{background: #667eea; color: white; padding: 15px; margin: -20px -20px 20px -20px; border-radius: 10px 10px 0 0;}}
        .comparison {{display: grid; grid-template-columns: 1fr 1fr; gap: 20px;}}
        .engine {{background: #f5f5f5; padding: 15px; border-radius: 8px;}}
        .metric {{display: flex; justify-content: space-between; margin: 10px 0;}}
        .differences {{background: #fff9e6; padding: 15px; border-radius: 8px; margin-top: 20px;}}
        .diff-item {{background: white; padding: 10px; margin: 10px 0; border-radius: 5px; border-left: 4px solid #f39c12;}}
        .diff-type {{color: #e67e22; font-weight: bold; margin-bottom: 5px;}}
        .diff-text {{font-family: monospace; font-size: 0.9em; margin: 5px 0; padding: 5px; background: #f9f9f9; border-radius: 3px;}}
        .tesseract-text {{border-left: 3px solid #3498db;}}
        .easyocr-text {{border-left: 3px solid #2ecc71;}}
        .similarity-bar {{background: #e0e0e0; height: 30px; border-radius: 15px; overflow: hidden; margin: 20px 0;}}
        .similarity-fill {{background: linear-gradient(90deg, #667eea, #764ba2); height: 100%; display: flex; align-items: center; justify-content: center; color: white;}}
        .no-diff {{color: #27ae60; text-align: center; font-style: italic;}}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>OCR Comparison Report</h1>
            <p>Tesseract vs EasyOCR Analysis</p>
        </div>
        <div class="stats">
            <div class="stat">
                <div class="stat-value">{total_files}</div>
                <div class="stat-label">Files Processed</div>
            </div>
            <div class="stat">
                <div class="stat-value">{avg_tesseract_time:.2f}s</div>
                <div class="stat-label">Avg Tesseract Time</div>
            </div>
            <div class="stat">
                <div class="stat-value">{avg_easyocr_time:.2f}s</div>
                <div class="stat-label">Avg EasyOCR Time</div>
            </div>
            <div class="stat">
                <div class="stat-value">{avg_similarity:.1f}%</div>
                <div class="stat-label">Avg Similarity</div>
            </div>
        </div>"""
        
        for idx, result in enumerate(all_results, 1):
            diffs_html = ""
            if result['differences']:
                for i, diff in enumerate(result['differences'], 1):
                    diffs_html += f"""
                    <div class="diff-item">
                        <div class="diff-type">Difference {i}: {diff.get('type', 'Word difference')}</div>
                        <div class="diff-text tesseract-text"><strong>Tesseract:</strong> {diff['tesseract']}</div>
                        <div class="diff-text easyocr-text"><strong>EasyOCR:</strong> {diff['easyocr']}</div>
                    </div>"""
            else:
                diffs_html = '<p class="no-diff">✓ No significant differences found - both OCR engines produced very similar results!</p>'
            
            html += f"""
        <div class="file-section">
            <div class="file-header">
                <h2>File {idx}: {result['file_name']}</h2>
            </div>
            <div class="comparison">
                <div class="engine">
                    <h3>🔵 Tesseract</h3>
                    <div class="metric"><span>Time:</span><strong>{result['tesseract']['time']:.3f}s</strong></div>
                    <div class="metric"><span>Confidence:</span><strong>{result['tesseract']['confidence']:.1f}%</strong></div>
                    <div class="metric"><span>Characters:</span><strong>{result['tesseract']['char_count']:,}</strong></div>
                    <div class="metric"><span>Words:</span><strong>{result['tesseract']['word_count']:,}</strong></div>
                </div>
                <div class="engine">
                    <h3>🟢 EasyOCR</h3>
                    <div class="metric"><span>Time:</span><strong>{result['easyocr']['time']:.3f}s</strong></div>
                    <div class="metric"><span>Confidence:</span><strong>{result['easyocr']['confidence']:.1f}%</strong></div>
                    <div class="metric"><span>Characters:</span><strong>{result['easyocr']['char_count']:,}</strong></div>
                    <div class="metric"><span>Words:</span><strong>{result['easyocr']['word_count']:,}</strong></div>
                </div>
            </div>
            <div class="similarity-bar">
                <div class="similarity-fill" style="width: {result['similarity']}%">
                    {result['similarity']:.1f}% Similar
                </div>
            </div>
            <div class="differences">
                <h3>🔍 Key Differences (Top 5)</h3>
                {diffs_html}
            </div>
        </div>"""
        
        html += """
    </div>
</body>
</html>"""
        
        with open(html_file, 'w', encoding='utf-8') as f:
            f.write(html)
        
        return html_file
    
    def process_multiple_images(self, image_paths):
        all_results = []
        print("\nSTARTING OCR COMPARISON")
        print("="*60)
        
        for image_path in image_paths:
            try:
                result = self.process_single_image(image_path)
                all_results.append(result)
            except Exception as e:
                print(f"Error processing {image_path}: {str(e)}")
        
        if all_results:
            html_file = self.generate_html_report(all_results)
            print("\n" + "="*60)
            print("COMPARISON COMPLETE!")
            print(f"Files processed: {len(all_results)}")
            print(f"Report saved: {html_file}")
            print("="*60)
            webbrowser.open(str(html_file))
            return all_results
        else:
            print("No images were successfully processed!")
            return []

if __name__ == "__main__":
    image_paths = sys.argv[1:] if len(sys.argv) > 1 else []
    
    if not image_paths:
        test_images = ["wen_documents/posters/3102_phase2_dose_optimization.jpg"]
        image_paths = [p for p in test_images if Path(p).exists()]
        
        if not image_paths:
            for ext in ['.jpg', '.jpeg', '.png']:
                image_paths.extend(list(Path('.').glob(f'*{ext}'))[:3])
    
    if not image_paths:
        print("No image files found!")
    else:
        comparator = UltimateOCRComparison()
        comparator.process_multiple_images(image_paths[:5])
