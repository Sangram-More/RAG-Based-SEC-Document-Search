"""
Document Processing Script

This script processes SEC filings through the complete pipeline:
1. Convert HTML files to PDF (with charts)
2. Load PDFs as Document objects
3. Chunk documents using selected strategy
4. Save processed chunks to JSON for vector store indexing

Usage:
    python scripts/process_documents.py
    python scripts/process_documents.py --strategy semantic
    python scripts/process_documents.py --chunk-size 800
"""

import argparse
import json
import sys
from pathlib import Path

# Add project root to path so we can import src modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config.logging_config import logger
from src.data.processor import convert_all_html_to_pdf, load_pdf_document
from src.rag.chunker import chunk_documents


def process_all_documents(
    raw_dir: Path = Path("./data/raw"),
    pdf_dir: Path = Path("./data/pdf"),
    output_dir: Path = Path("./data/processed"),
    strategy: str = "fixed_size",
    chunk_size: int = 500,
    chunk_overlap: int = 50
) -> dict:
    """
    Process all SEC filings through the complete pipeline.

    Args:
        raw_dir: Directory containing HTML files
        pdf_dir: Directory to store converted PDFs
        output_dir: Directory to save processed JSON files
        strategy: Chunking strategy ("fixed_size", "page_based", "semantic")
        chunk_size: Chunk size for fixed_size strategy
        chunk_overlap: Overlap for fixed_size strategy

    Returns:
        Summary statistics dictionary
    """
    logger.info("=" * 60)
    logger.info("DOCUMENT PROCESSING PIPELINE")
    logger.info("=" * 60)

    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Stats tracking
    stats = {
        "html_files": 0,
        "pdfs_created": 0,
        "total_pages": 0,
        "total_chunks": 0,
        "files_processed": []
    }

    # =========================================
    # STEP 1: Convert HTML to PDF
    # =========================================
    logger.info("\n[STEP 1] Converting HTML files to PDF...")

    html_files = list(Path(raw_dir).glob("*.html"))
    stats["html_files"] = len(html_files)

    if html_files:
        pdf_paths = convert_all_html_to_pdf(raw_dir, pdf_dir)
        stats["pdfs_created"] = len(pdf_paths)
        logger.info(f"Converted {len(pdf_paths)} HTML files to PDF")
    else:
        logger.warning(f"No HTML files found in {raw_dir}")

    # =========================================
    # STEP 2 & 3: Load PDFs and Chunk
    # =========================================
    logger.info(f"\n[STEP 2] Loading PDFs and chunking (strategy: {strategy})...")

    pdf_files = list(Path(pdf_dir).glob("*.pdf"))

    all_chunks = []

    for pdf_file in pdf_files:
        logger.info(f"Processing: {pdf_file.name}")

        # Load PDF pages
        pages = load_pdf_document(pdf_file)
        stats["total_pages"] += len(pages)

        # Chunk the pages
        chunks = chunk_documents(
            pages,
            strategy=strategy,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        stats["total_chunks"] += len(chunks)

        # Track file info
        stats["files_processed"].append({
            "filename": pdf_file.name,
            "pages": len(pages),
            "chunks": len(chunks)
        })

        all_chunks.extend(chunks)

    # =========================================
    # STEP 4: Save to JSON
    # =========================================
    logger.info("\n[STEP 3] Saving processed documents to JSON...")

    # Convert chunks to serializable format
    chunks_data = []
    for chunk in all_chunks:
        chunks_data.append({
            "text": chunk.page_content,
            "metadata": chunk.metadata
        })

    # Save all chunks to single JSON file
    output_file = output_dir / f"chunks_{strategy}.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump({
            "strategy": strategy,
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "total_chunks": len(chunks_data),
            "chunks": chunks_data
        }, f, indent=2, default=str)

    logger.info(f"Saved {len(chunks_data)} chunks to {output_file}")

    # Also save per-company files for easier inspection
    chunks_by_ticker = {}
    for chunk in all_chunks:
        ticker = chunk.metadata.get("ticker", "UNKNOWN")
        if ticker not in chunks_by_ticker:
            chunks_by_ticker[ticker] = []
        chunks_by_ticker[ticker].append({
            "text": chunk.page_content,
            "metadata": chunk.metadata
        })

    for ticker, ticker_chunks in chunks_by_ticker.items():
        ticker_file = output_dir / f"{ticker}_chunks.json"
        with open(ticker_file, "w", encoding="utf-8") as f:
            json.dump({
                "ticker": ticker,
                "total_chunks": len(ticker_chunks),
                "chunks": ticker_chunks
            }, f, indent=2, default=str)
        logger.info(f"Saved {len(ticker_chunks)} chunks for {ticker}")

    # =========================================
    # SUMMARY
    # =========================================
    logger.info("\n" + "=" * 60)
    logger.info("PROCESSING COMPLETE")
    logger.info("=" * 60)
    logger.info(f"HTML files found:    {stats['html_files']}")
    logger.info(f"PDFs created:        {stats['pdfs_created']}")
    logger.info(f"Total pages:         {stats['total_pages']}")
    logger.info(f"Total chunks:        {stats['total_chunks']}")
    logger.info(f"Chunking strategy:   {strategy}")
    logger.info(f"Output directory:    {output_dir}")
    logger.info("=" * 60)

    return stats


# =========================================
# MAIN
# =========================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process SEC filings for RAG")
    parser.add_argument(
        "--strategy",
        type=str,
        default="fixed_size",
        choices=["fixed_size", "page_based", "semantic"],
        help="Chunking strategy to use"
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=500,
        help="Chunk size for fixed_size strategy"
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=50,
        help="Overlap between chunks"
    )

    args = parser.parse_args()

    stats = process_all_documents(
        strategy=args.strategy,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap
    )

    # Print summary table
    print("\n" + "=" * 50)
    print("FILES PROCESSED:")
    print("=" * 50)
    for file_info in stats["files_processed"]:
        print(f"  {file_info['filename']}: {file_info['pages']} pages → {file_info['chunks']} chunks")
