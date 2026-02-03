from pathlib import Path
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.config.logging_config import logger


# ==============================================
# CHUNKING STRATEGY 1: Fixed-Size Chunking
# ==============================================

def chunk_by_size(
    documents: list[Document],
    chunk_size: int = 500,
    chunk_overlap: int = 50
) -> list[Document]:
    """
    Split documents into fixed-size chunks with overlap.

    Pros: Simple, predictable chunk sizes
    Cons: May split sentences awkwardly

    Args:
        documents: List of Document objects (from load_pdf_document)
        chunk_size: Maximum characters per chunk
        chunk_overlap: Overlap between chunks

    Returns:
        List of chunked Document objects with metadata
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len
    )

    chunks = []

    for doc in documents:
        doc_chunks = splitter.split_text(doc.page_content)

        for i, chunk_text in enumerate(doc_chunks):
            chunk = Document(
                page_content=chunk_text,
                metadata={
                    **doc.metadata,
                    "chunk_index": i,
                    "total_chunks": len(doc_chunks),
                    "chunking_strategy": "fixed_size"
                }
            )
            chunks.append(chunk)

    logger.info(f"[Fixed-Size] Created {len(chunks)} chunks from {len(documents)} pages")
    return chunks


# ==============================================
# CHUNKING STRATEGY 2: Page-Based Chunking
# ==============================================

def chunk_by_page(documents: list[Document]) -> list[Document]:
    """
    Use each PDF page as a single chunk.

    Pros: Respects document structure, preserves context
    Cons: Chunks may be too large for embedding models

    Args:
        documents: List of Document objects (one per page from load_pdf_document)

    Returns:
        List of Document objects (same as input, with updated metadata)
    """
    chunks = []

    for doc in documents:
        chunk = Document(
            page_content=doc.page_content,
            metadata={
                **doc.metadata,
                "chunk_index": 0,
                "total_chunks": 1,
                "chunking_strategy": "page_based"
            }
        )
        chunks.append(chunk)

    logger.info(f"[Page-Based] Created {len(chunks)} chunks (1 per page)")
    return chunks


# ==============================================
# CHUNKING STRATEGY 3: Semantic Chunking
# ==============================================

def chunk_by_semantic(
    documents: list[Document],
    max_chunk_size: int = 1500
) -> list[Document]:
    """
    Split by paragraphs and section headers, keeping semantic units together.

    Pros: Keeps related content together, respects document structure
    Cons: Variable chunk sizes, may create very large chunks

    Args:
        documents: List of Document objects
        max_chunk_size: Maximum size before forcing a split

    Returns:
        List of chunked Document objects
    """
    # Use paragraph-aware separators (prioritize double newlines)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=max_chunk_size,
        chunk_overlap=100,
        separators=[
            "\n\n\n",      # Section breaks
            "\n\n",        # Paragraphs
            "\nItem ",     # SEC section headers
            "\nPART ",     # SEC part headers
            "\n",          # Lines
            ". ",          # Sentences
            " ",           # Words
            ""
        ],
        length_function=len
    )

    chunks = []

    for doc in documents:
        doc_chunks = splitter.split_text(doc.page_content)

        for i, chunk_text in enumerate(doc_chunks):
            chunk = Document(
                page_content=chunk_text,
                metadata={
                    **doc.metadata,
                    "chunk_index": i,
                    "total_chunks": len(doc_chunks),
                    "chunking_strategy": "semantic"
                }
            )
            chunks.append(chunk)

    logger.info(f"[Semantic] Created {len(chunks)} chunks from {len(documents)} pages")
    return chunks


# ==============================================
# UNIFIED CHUNKING FUNCTION
# ==============================================

def chunk_documents(
    documents: list[Document],
    strategy: str = "fixed_size",
    chunk_size: int = 500,
    chunk_overlap: int = 50
) -> list[Document]:
    """
    Chunk documents using the specified strategy.

    Args:
        documents: List of Document objects from load_pdf_document()
        strategy: One of "fixed_size", "page_based", "semantic"
        chunk_size: Size for fixed_size strategy
        chunk_overlap: Overlap for fixed_size strategy

    Returns:
        List of chunked Document objects
    """
    if strategy == "fixed_size":
        return chunk_by_size(documents, chunk_size, chunk_overlap)
    elif strategy == "page_based":
        return chunk_by_page(documents)
    elif strategy == "semantic":
        return chunk_by_semantic(documents, max_chunk_size=chunk_size * 3)
    else:
        raise ValueError(f"Unknown strategy: {strategy}. Use 'fixed_size', 'page_based', or 'semantic'")


# ==============================================
# TEST CODE
# ==============================================

if __name__ == "__main__":
    from src.data.processor import load_pdf_document

    # Load PDF
    pdf_path = Path("./data/pdf/GOOGL_2024-01-31.pdf")
    pages = load_pdf_document(pdf_path)
    print(f"Loaded {len(pages)} pages\n")

    # Test all 3 strategies
    print("=" * 50)
    print("CHUNKING STRATEGY COMPARISON")
    print("=" * 50)

    # Strategy 1: Fixed-size
    chunks_fixed = chunk_documents(pages, strategy="fixed_size", chunk_size=500)
    print(f"\n1. Fixed-Size (500 chars):")
    print(f"   Chunks: {len(chunks_fixed)}")
    print(f"   Avg size: {sum(len(c.page_content) for c in chunks_fixed) // len(chunks_fixed)} chars")

    # Strategy 2: Page-based
    chunks_page = chunk_documents(pages, strategy="page_based")
    print(f"\n2. Page-Based:")
    print(f"   Chunks: {len(chunks_page)}")
    print(f"   Avg size: {sum(len(c.page_content) for c in chunks_page) // len(chunks_page)} chars")

    # Strategy 3: Semantic
    chunks_semantic = chunk_documents(pages, strategy="semantic")
    print(f"\n3. Semantic:")
    print(f"   Chunks: {len(chunks_semantic)}")
    print(f"   Avg size: {sum(len(c.page_content) for c in chunks_semantic) // len(chunks_semantic)} chars")

    print("\n" + "=" * 50)