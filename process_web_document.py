#!/usr/bin/env python3
"""
Web Document OCR Processor
Downloads documents from URLs and runs the OCR pipeline on them.
"""

import argparse
import sys
import os
import tempfile
import subprocess
from pathlib import Path
import requests
from urllib.parse import urlparse, unquote
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def download_document(url: str, output_dir: Path = None) -> Path:
    """
    Download a document from a URL.

    Args:
        url: URL of the document to download
        output_dir: Directory to save the file (uses temp dir if None)

    Returns:
        Path to the downloaded file
    """
    try:
        # Create output directory
        if output_dir is None:
            output_dir = Path(tempfile.gettempdir()) / "ocr_web_downloads"
        output_dir.mkdir(parents=True, exist_ok=True)

        # Get filename from URL
        parsed_url = urlparse(url)
        filename = unquote(os.path.basename(parsed_url.path))

        # If no filename in URL, create one based on the domain
        if not filename or '.' not in filename:
            domain = parsed_url.netloc.replace('.', '_')
            filename = f"{domain}_document.pdf"

        # Ensure valid extension
        valid_extensions = ['.pdf', '.png', '.jpg', '.jpeg', '.tiff', '.bmp']
        if not any(filename.lower().endswith(ext) for ext in valid_extensions):
            filename += '.pdf'  # Default to PDF

        output_path = output_dir / filename

        # Download the file
        logger.info(f"Downloading from: {url}")
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }

        response = requests.get(url, headers=headers, stream=True, timeout=30)
        response.raise_for_status()

        # Save the file
        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0

        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        progress = (downloaded / total_size) * 100
                        print(f"\rDownloading: {progress:.1f}%", end='', flush=True)

        print()  # New line after progress
        logger.info(f"Downloaded successfully to: {output_path}")
        return output_path

    except requests.RequestException as e:
        logger.error(f"Failed to download document: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise

def process_with_pipeline(file_path: Path, pipeline_script: str = "boss_pipeline_enhanced.py"):
    """
    Run the OCR pipeline on a document.

    Args:
        file_path: Path to the document
        pipeline_script: Name of the pipeline script to run
    """
    try:
        logger.info(f"Running OCR pipeline on: {file_path}")

        # Check if pipeline script exists
        if not Path(pipeline_script).exists():
            logger.error(f"Pipeline script not found: {pipeline_script}")
            return False

        # Run the pipeline
        cmd = [sys.executable, pipeline_script, str(file_path)]
        logger.info(f"Executing: {' '.join(cmd)}")

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True
        )

        if result.returncode == 0:
            logger.info("Pipeline completed successfully!")
            print("\n" + "="*60)
            print("PIPELINE OUTPUT:")
            print("="*60)
            print(result.stdout)
            return True
        else:
            logger.error(f"Pipeline failed with error:\n{result.stderr}")
            return False

    except Exception as e:
        logger.error(f"Failed to run pipeline: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(
        description="Download and process web documents with OCR pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process a PDF from a URL
  python process_web_document.py https://example.com/document.pdf

  # Process multiple URLs
  python process_web_document.py url1 url2 url3

  # Keep downloaded files
  python process_web_document.py --keep https://example.com/doc.pdf

  # Use a different pipeline script
  python process_web_document.py --pipeline main.py https://example.com/doc.pdf

Common document URLs for testing:
  - ArXiv papers: https://arxiv.org/pdf/[paper_id].pdf
  - PubMed Central: https://www.ncbi.nlm.nih.gov/pmc/articles/[PMC_ID]/pdf/
  - Clinical trial protocols from clinicaltrials.gov
        """
    )

    parser.add_argument(
        'urls',
        nargs='+',
        help='URL(s) of documents to process'
    )

    parser.add_argument(
        '--keep',
        action='store_true',
        help='Keep downloaded files after processing (default: delete)'
    )

    parser.add_argument(
        '--output-dir',
        type=Path,
        default=None,
        help='Directory to save downloaded files (default: temp directory)'
    )

    parser.add_argument(
        '--pipeline',
        default='boss_pipeline_enhanced.py',
        help='Pipeline script to use (default: boss_pipeline_enhanced.py)'
    )

    parser.add_argument(
        '--skip-download',
        action='store_true',
        help='Skip download if file already exists in output directory'
    )

    args = parser.parse_args()

    # Process each URL
    for url in args.urls:
        print("\n" + "="*60)
        print(f"Processing: {url}")
        print("="*60)

        try:
            # Check if it's a local file
            if os.path.exists(url):
                file_path = Path(url)
                logger.info(f"Using local file: {file_path}")
            else:
                # Download the document
                file_path = download_document(url, args.output_dir)

            # Run the pipeline
            success = process_with_pipeline(file_path, args.pipeline)

            # Clean up if requested
            if not args.keep and file_path.exists() and not os.path.exists(url):
                logger.info(f"Removing downloaded file: {file_path}")
                file_path.unlink()

            if success:
                logger.info(f"Successfully processed: {url}")
            else:
                logger.error(f"Failed to process: {url}")

        except Exception as e:
            logger.error(f"Error processing {url}: {e}")
            continue

    print("\n" + "="*60)
    print("All documents processed!")
    print("="*60)

if __name__ == "__main__":
    main()