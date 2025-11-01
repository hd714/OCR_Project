import sys
from pathlib import Path
import pytesseract
from PIL import Image
from datetime import datetime
import webbrowser

# Set Tesseract path
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

class BossOCR:
    def __init__(self):
        self.results_dir = Path("boss_results")
        self.results_dir.mkdir(exist_ok=True)
    
    def process(self, image_path):
        print("\n" + "="*60)
        print(f"PROCESSING: {image_path}")
        print("="*60)
        
        # Extract text
        img = Image.open(image_path)
        text = pytesseract.image_to_string(img)
        
        # Save files
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        text_file = self.results_dir / f"text_{timestamp}.txt"
        html_file = self.results_dir / f"report_{timestamp}.html"
        
        # Save text
        with open(text_file, "w", encoding="utf-8") as f:
            f.write(text)
        
        # Create HTML report
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>OCR Results - {Path(image_path).name}</title>
            <style>
                body {{ 
                    font-family: Arial, sans-serif; 
                    margin: 0; 
                    padding: 20px; 
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                }}
                .container {{
                    max-width: 1200px;
                    margin: 0 auto;
                    background: white;
                    border-radius: 10px;
                    overflow: hidden;
                    box-shadow: 0 10px 40px rgba(0,0,0,0.2);
                }}
                .header {{
                    background: #2d3748;
                    color: white;
                    padding: 30px;
                }}
                .stats {{
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                    gap: 20px;
                    padding: 30px;
                    background: #f7fafc;
                }}
                .stat-card {{
                    background: white;
                    padding: 20px;
                    border-radius: 8px;
                    text-align: center;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                }}
                .stat-value {{
                    font-size: 36px;
                    font-weight: bold;
                    color: #667eea;
                }}
                .stat-label {{
                    color: #718096;
                    margin-top: 5px;
                }}
                .content {{
                    padding: 30px;
                }}
                .text-preview {{
                    background: #f7fafc;
                    padding: 20px;
                    border-radius: 8px;
                    font-family: monospace;
                    font-size: 14px;
                    max-height: 500px;
                    overflow-y: auto;
                    white-space: pre-wrap;
                    word-wrap: break-word;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>OCR Processing Complete!</h1>
                    <p>File: {Path(image_path).name}</p>
                    <p>Processed: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
                </div>
                
                <div class="stats">
                    <div class="stat-card">
                        <div class="stat-value">{len(text):,}</div>
                        <div class="stat-label">Characters</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value">{len(text.split()):,}</div>
                        <div class="stat-label">Words</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value">{len(text.splitlines()):,}</div>
                        <div class="stat-label">Lines</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value">Tesseract</div>
                        <div class="stat-label">OCR Engine</div>
                    </div>
                </div>
                
                <div class="content">
                    <h2>Extracted Text Preview</h2>
                    <div class="text-preview">{text[:3000]}...</div>
                </div>
            </div>
        </body>
        </html>
        """
        
        # Save HTML
        with open(html_file, "w", encoding="utf-8") as f:
            f.write(html)
        
        print(f"\nSUCCESS!")
        print(f"Characters extracted: {len(text):,}")
        print(f"Words: {len(text.split()):,}")
        print(f"\nFiles saved:")
        print(f"  Text: {text_file}")
        print(f"  HTML: {html_file}")
        
        # Open HTML in browser
        webbrowser.open(str(html_file))
        
        return text

# Run if called directly
if __name__ == "__main__":
    import sys
    
    boss = BossOCR()
    
    if len(sys.argv) > 1:
        boss.process(sys.argv[1])
    else:
        # Process the medical poster by default
        boss.process("wen_documents/posters/3102_phase2_dose_optimization.jpg")
