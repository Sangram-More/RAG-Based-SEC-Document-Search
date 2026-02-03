from pathlib import Path
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from src.config.logging_config import logger
from playwright.sync_api import sync_playwright


def convert_html_to_pdf(html_path: Path, output_dir: Path = None) -> Path:
    """
    Convert HTML filing to PDF using Playwright (Chromium).
    Properly renders JavaScript charts and modern web content.

    Args:
        html_path: Path to HTML file
        output_dir: Directory to save PDF (default: data/pdf)

    Returns:
        Path to created PDF file
    """
    html_path = Path(html_path).resolve()

    if output_dir is None:
        output_dir = html_path.parent.parent / "pdf"

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pdf_path = output_dir / f"{html_path.stem}.pdf"

    logger.info(f"Converting {html_path.name} to PDF...")

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()

        # Load local HTML file
        page.goto(f"file:///{html_path}")

        # Wait for page to fully render (including JS charts)
        page.wait_for_load_state("networkidle")

        # Generate PDF
        page.pdf(
            path=str(pdf_path),
            format="Letter",
            margin={
                "top": "0.5in",
                "right": "0.5in",
                "bottom": "0.5in",
                "left": "0.5in"
            },
            print_background=True
        )

        browser.close()

    logger.info(f"Created PDF: {pdf_path}")
    return pdf_path


def load_pdf_document(pdf_path: Path) -> list[Document]:
    """
    Load PDF filing for RAG processing.

    Args:
        pdf_path: Path to PDF file

    Returns:
        List of Document objects (one per page)
    """
    pdf_path = Path(pdf_path)
    logger.info(f"Loading PDF: {pdf_path.name}")

    loader = PyPDFLoader(str(pdf_path))
    pages = loader.load()

    # Extract metadata from filename
    filename = pdf_path.stem
    parts = filename.split("_")
    ticker = parts[0] if len(parts) >= 1 else "UNKNOWN"
    filing_date = parts[1] if len(parts) >= 2 else "UNKNOWN"

    # Add metadata to each page
    for page in pages:
        page.metadata.update({
            "ticker": ticker,
            "filing_date": filing_date,
            "doc_type": "pdf"
        })

    logger.info(f"Loaded {len(pages)} pages from {pdf_path.name}")
    return pages


def convert_all_html_to_pdf(input_dir: Path, output_dir: Path = None) -> list[Path]:
    """
    Convert all HTML files in directory to PDF.

    Args:
        input_dir: Directory containing HTML files
        output_dir: Directory to save PDFs (default: data/pdf)

    Returns:
        List of paths to created PDF files
    """
    input_dir = Path(input_dir)
    html_files = list(input_dir.glob("*.html"))

    logger.info(f"Found {len(html_files)} HTML files to convert")

    pdf_paths = []
    for html_file in html_files:
        try:
            pdf_path = convert_html_to_pdf(html_file, output_dir)
            pdf_paths.append(pdf_path)
        except Exception as e:
            logger.error(f"Failed to convert {html_file.name}: {e}")

    logger.info(f"Converted {len(pdf_paths)}/{len(html_files)} files to PDF")
    return pdf_paths


# ==============================================================================
# TEST CODE
# ==============================================================================

if __name__ == "__main__":
    test_file = Path("./data/raw/GOOGL_2024-01-31.html")

    if test_file.exists():
        print("=" * 60)
        print("Testing HTML to PDF Conversion (Playwright)")
        print("=" * 60)

        # 1. Convert HTML to PDF
        print("\n1. Converting HTML to PDF...")
        pdf_path = convert_html_to_pdf(test_file)
        print(f"   Created: {pdf_path}")
        print(f"   Size: {pdf_path.stat().st_size / 1024:.1f} KB")

        # 2. Load PDF for RAG
        print("\n2. Loading PDF for RAG...")
        pages = load_pdf_document(pdf_path)
        print(f"   Loaded {len(pages)} pages")

        # 3. Show sample content
        print("\n3. Sample content from first page:")
        print("-" * 40)
        sample = pages[0].page_content[:500].encode('ascii', 'replace').decode('ascii')
        print(sample)
        print("-" * 40)

        # 4. Show metadata
        print("\n4. Document metadata:")
        print(f"   Ticker: {pages[0].metadata.get('ticker')}")
        print(f"   Filing Date: {pages[0].metadata.get('filing_date')}")
        print(f"   Doc Type: {pages[0].metadata.get('doc_type')}")
        print(f"   Source: {pages[0].metadata.get('source')}")

        print("\n" + "=" * 60)
        print("PDF Conversion Test Complete!")
        print("=" * 60)
    else:
        print(f"Test file not found: {test_file}")