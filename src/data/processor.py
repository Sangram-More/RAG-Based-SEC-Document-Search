from pathlib import Path
from langchain_community.document_loaders import BSHTMLLoader
from langchain_core.documents import Document 
from src.config.logging_config import logger
import re

def clean_text(text: str) -> str:
    """
    Clean and normalize text from SEC Filings.

    Args:
        text: Raw text extracted from HTML.
    
    Returns:
        returns clean text in string fromat.
    """

    # Removing excessive whitespaces (replacing multiple spaces with single spcae)
    text = re.sub(r' +', ' ', text)

    # Normalising new lines (multiple newlines to double new lines)
    text = re.sub(r'\n\s*\n', '\n\n', text)

    # Remove leading/trailing whitespace from each line
    lines = [line.strip() for line in text.split('\n')]
    text = '\n'.join(lines)

    # Remove any remaining excessive newlines
    text = re.sub(r'\n{3,}', '\n\n', text)

    # Strip leading/trailing whitespace from entire document
    text = text.strip()
    
    return text

# -----------------------------------------------------------------------

def load_html_document(file_path: Path) -> Document:
    """
    Load and clean HTML SEC filing.
    
    Args:
        file_path: Path to HTML file (e.g., "data/raw/AAPL_2024-11-01.html")
    
    Returns:
        Document object with cleaned content and metadata
    """
    file_path = Path(file_path)
    logger.info(f"Loading Document: {file_path.name}")

    # Using BSHTMLLoader to load and parse the html file
    loader = BSHTMLLoader(str(file_path), open_encoding="utf-8")
    docs = loader.load()

    # Getting the first document from the list returned by BSHTMLLoader
    raw_content = docs[0].page_content

    # Text cleaning
    cleaned_content = clean_text(raw_content)

    # Extract metadata from filename
    # Filename format: {ticker}_{filing_date}.html (e.g., AAPL_2024-11-01.html)
    filename = file_path.stem  # "AAPL_2024-11-01"
    parts = filename.split("_")
    ticker = parts[0] if len(parts) >= 1 else "UNKNOWN"
    filing_date = parts[1] if len(parts) >= 2 else "UNKNOWN"

    # Create documents with metadata
    document = Document(
        page_content=cleaned_content,
        metadata = {
            "source":file_path.name,
            "ticker":ticker,
            "filing_date":filing_date,
            "doc_type":"text"
        }
    )

    logger.info(f"Loaded {ticker} filing from {filing_date} ({len(cleaned_content)} chars)")

    return document

# ----------------------------------------------------------------

# ==================================================================
# TEST CODE 
# ==================================================================

if __name__ == "__main__":
    from pathlib import Path
    
    # Test with a real file
    test_file = Path("./data/raw/AAPL_2024-11-01.html")
    
    if test_file.exists():
        doc = load_html_document(test_file)
        
        print(f"Ticker: {doc.metadata['ticker']}")
        print(f"Filing Date: {doc.metadata['filing_date']}")
        print(f"Content Length: {len(doc.page_content)} chars")
        print(f"\nFirst 500 chars:\n{doc.page_content[:500]}...")
    else:
        print(f"Test file not found: {test_file}")