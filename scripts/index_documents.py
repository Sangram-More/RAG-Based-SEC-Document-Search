"""
Document Indexing Script

This script indexes SEC filings into a ChromaDB vector store:
1. Load PDF documents from data/pdf/
2. Chunk documents using selected strategy
3. Create embeddings and store in ChromaDB
4. Save checkpoints per company for resilience

Usage:
    python scripts/index_documents.py
    python scripts/index_documents.py --strategy semantic
    python scripts/index_documents.py --batch-size 100
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from langchain_core.documents import Document
from src.config.logging_config import logger
from src.config.settings import get_settings
from src.data.processor import load_pdf_document
from src.rag.chunker import chunk_documents
from src.rag.vectorstore import get_embeddings
from langchain_chroma import Chroma


def index_documents(
    pdf_dir: Path = Path("./data/pdf"),
    persist_dir: Path = None,
    collection_name: str = "sec_filings",
    strategy: str = "fixed_size",
    chunk_size: int = 500,
    chunk_overlap: int = 50,
    batch_size: int = 100
) -> dict:
    """
    Index all PDF documents into ChromaDB.

    Args:
        pdf_dir: Directory containing PDF files
        persist_dir: Directory to persist ChromaDB (default: from settings)
        collection_name: Name of the Chroma collection
        strategy: Chunking strategy ("fixed_size", "page_based", "semantic")
        chunk_size: Size of chunks for fixed_size strategy
        chunk_overlap: Overlap between chunks
        batch_size: Number of documents to embed at once

    Returns:
        Statistics dictionary
    """
    settings = get_settings()
    if persist_dir is None:
        persist_dir = settings.chromadb_dir

    persist_dir = Path(persist_dir)
    persist_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("DOCUMENT INDEXING PIPELINE")
    logger.info("=" * 60)
    logger.info(f"PDF Directory: {pdf_dir}")
    logger.info(f"ChromaDB Directory: {persist_dir}")
    logger.info(f"Collection: {collection_name}")
    logger.info(f"Strategy: {strategy}")
    logger.info(f"Batch Size: {batch_size}")

    # Stats tracking
    stats = {
        "start_time": datetime.now().isoformat(),
        "pdfs_processed": 0,
        "pdfs_failed": 0,
        "total_pages": 0,
        "total_chunks": 0,
        "companies": {}
    }

    # Get all PDF files
    pdf_files = sorted(Path(pdf_dir).glob("*.pdf"))
    logger.info(f"Found {len(pdf_files)} PDF files")

    if not pdf_files:
        logger.warning(f"No PDF files found in {pdf_dir}")
        return stats

    # Initialize embedding model once
    logger.info("\n[STEP 1] Loading embedding model...")
    embeddings = get_embeddings()

    # Collect all chunks
    all_chunks = []
    checkpoint_dir = persist_dir / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)

    # Process each PDF
    logger.info("\n[STEP 2] Processing PDF files...")
    for i, pdf_file in enumerate(pdf_files, 1):
        ticker = pdf_file.stem.split("_")[0]
        logger.info(f"[{i}/{len(pdf_files)}] Processing: {pdf_file.name}")

        try:
            # Load PDF
            pages = load_pdf_document(pdf_file)
            stats["total_pages"] += len(pages)

            # Chunk documents
            chunks = chunk_documents(
                pages,
                strategy=strategy,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )
            stats["total_chunks"] += len(chunks)
            stats["pdfs_processed"] += 1

            # Track per-company stats
            if ticker not in stats["companies"]:
                stats["companies"][ticker] = {"pages": 0, "chunks": 0, "files": []}
            stats["companies"][ticker]["pages"] += len(pages)
            stats["companies"][ticker]["chunks"] += len(chunks)
            stats["companies"][ticker]["files"].append(pdf_file.name)

            all_chunks.extend(chunks)

            logger.info(f"    Pages: {len(pages)}, Chunks: {len(chunks)}")

            # Save checkpoint per company (every unique ticker processed)
            checkpoint_file = checkpoint_dir / f"{ticker}_checkpoint.json"
            with open(checkpoint_file, "w") as f:
                json.dump({
                    "ticker": ticker,
                    "files_processed": stats["companies"][ticker]["files"],
                    "total_chunks": stats["companies"][ticker]["chunks"],
                    "timestamp": datetime.now().isoformat()
                }, f, indent=2)

        except Exception as e:
            logger.error(f"    Failed to process {pdf_file.name}: {e}")
            stats["pdfs_failed"] += 1
            continue

    # Create vector store with all chunks
    logger.info(f"\n[STEP 3] Creating vector store with {len(all_chunks)} chunks...")

    if all_chunks:
        # Process in batches for memory efficiency
        total_batches = (len(all_chunks) + batch_size - 1) // batch_size

        for batch_num in range(total_batches):
            start_idx = batch_num * batch_size
            end_idx = min(start_idx + batch_size, len(all_chunks))
            batch = all_chunks[start_idx:end_idx]

            logger.info(f"    Batch {batch_num + 1}/{total_batches}: indexing {len(batch)} chunks...")

            if batch_num == 0:
                # First batch: create new collection
                vectorstore = Chroma.from_documents(
                    documents=batch,
                    embedding=embeddings,
                    collection_name=collection_name,
                    persist_directory=str(persist_dir),
                    collection_metadata={"hnsw:space": "cosine"}
                )
            else:
                # Subsequent batches: add to existing collection
                vectorstore.add_documents(batch)

        logger.info(f"Vector store created at: {persist_dir}")
    else:
        logger.warning("No chunks to index!")

    # Save final stats
    stats["end_time"] = datetime.now().isoformat()
    stats_file = persist_dir / "indexing_stats.json"
    with open(stats_file, "w") as f:
        json.dump(stats, f, indent=2)

    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("INDEXING COMPLETE")
    logger.info("=" * 60)
    logger.info(f"PDFs processed:  {stats['pdfs_processed']}")
    logger.info(f"PDFs failed:     {stats['pdfs_failed']}")
    logger.info(f"Total pages:     {stats['total_pages']}")
    logger.info(f"Total chunks:    {stats['total_chunks']}")
    logger.info(f"Companies:       {len(stats['companies'])}")
    logger.info(f"Stats saved to:  {stats_file}")
    logger.info("=" * 60)

    return stats


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Index SEC filings into ChromaDB")
    parser.add_argument(
        "--pdf-dir",
        type=str,
        default="./data/pdf",
        help="Directory containing PDF files"
    )
    parser.add_argument(
        "--collection",
        type=str,
        default="sec_filings",
        help="Name of the ChromaDB collection"
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default="fixed_size",
        choices=["fixed_size", "page_based", "semantic"],
        help="Chunking strategy"
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
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Batch size for embedding (100-500 recommended)"
    )

    args = parser.parse_args()

    stats = index_documents(
        pdf_dir=Path(args.pdf_dir),
        collection_name=args.collection,
        strategy=args.strategy,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        batch_size=args.batch_size
    )

    # Print per-company summary
    print("\n" + "=" * 50)
    print("PER-COMPANY SUMMARY:")
    print("=" * 50)
    for ticker, info in stats["companies"].items():
        print(f"  {ticker}: {info['pages']} pages -> {info['chunks']} chunks")
