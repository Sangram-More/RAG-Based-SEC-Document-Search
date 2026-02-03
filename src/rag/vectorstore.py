"""
Vector Store Module

This module provides functions to create, load, and manage
ChromaDB vector stores for the FinRAG system.

Usage:
    from src.rag.vectorstore import create_vectorstore, load_vectorstore

    # Create new vector store from documents
    vectorstore = create_vectorstore(documents, "./data/chroma_db", "sec_filings")

    # Load existing vector store
    vectorstore = load_vectorstore("./data/chroma_db", "sec_filings")
"""

import sys
from pathlib import Path

# Add project root to path for imports when running directly
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from src.config.logging_config import logger
from src.config.settings import get_settings


def get_embeddings(model_name: str = None) -> HuggingFaceEmbeddings:
    """
    Create embedding model instance.

    Args:
        model_name: HuggingFace model name (default: from settings)

    Returns:
        HuggingFaceEmbeddings instance
    """
    if model_name is None:
        model_name = get_settings().embedding_model

    logger.info(f"Loading embedding model: {model_name}")

    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cpu"},  # Use "cuda" if GPU available
        encode_kwargs={"normalize_embeddings": True}
    )

    logger.info(f"Embedding model loaded successfully")
    return embeddings


def create_vectorstore(
    documents: list[Document],
    persist_dir: str = None,
    collection_name: str = "sec_filings"
) -> Chroma:
    """
    Create and populate a new vector store from documents.

    Args:
        documents: List of LangChain Document objects to embed
        persist_dir: Directory to persist the vector store (default: from settings)
        collection_name: Name of the Chroma collection

    Returns:
        Chroma vector store instance
    """
    if persist_dir is None:
        persist_dir = str(get_settings().chromadb_dir)

    # Ensure directory exists
    Path(persist_dir).mkdir(parents=True, exist_ok=True)

    logger.info(f"Creating vector store with {len(documents)} documents")
    logger.info(f"Collection: {collection_name}, Directory: {persist_dir}")

    # Get embedding model
    embeddings = get_embeddings()

    # Create Chroma vector store
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        collection_name=collection_name,
        persist_directory=persist_dir,
        collection_metadata={"hnsw:space": "cosine"}
    )

    logger.info(f"Vector store created with {len(documents)} documents")
    return vectorstore


def load_vectorstore(
    persist_dir: str = None,
    collection_name: str = "sec_filings"
) -> Chroma:
    """
    Load an existing vector store from disk.

    Args:
        persist_dir: Directory where vector store is persisted
        collection_name: Name of the Chroma collection to load

    Returns:
        Chroma vector store instance
    """
    if persist_dir is None:
        persist_dir = str(get_settings().chromadb_dir)

    logger.info(f"Loading vector store from {persist_dir}")
    logger.info(f"Collection: {collection_name}")

    # Get embedding model (must match what was used to create)
    embeddings = get_embeddings()

    # Load existing Chroma vector store
    vectorstore = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=persist_dir
    )

    # Get collection count
    collection = vectorstore._collection
    count = collection.count()
    logger.info(f"Loaded vector store with {count} documents")

    return vectorstore


# =============================================================================
# TEST CODE
# =============================================================================

if __name__ == "__main__":
    import json

    # Load processed chunks
    chunks_file = Path("./data/processed/chunks_fixed_size.json")

    if not chunks_file.exists():
        print(f"Chunks file not found: {chunks_file}")
        print("Run 'python scripts/process_documents.py' first")
        exit(1)

    with open(chunks_file) as f:
        data = json.load(f)

    # Convert to Document objects
    documents = []
    for chunk in data["chunks"][:50]:  # Use first 50 for testing
        doc = Document(
            page_content=chunk["text"],
            metadata=chunk["metadata"]
        )
        documents.append(doc)

    print(f"Loaded {len(documents)} chunks for testing")

    # Test 1: Create vector store
    print("\n" + "=" * 50)
    print("TEST 1: Creating vector store...")
    print("=" * 50)

    vectorstore = create_vectorstore(
        documents=documents,
        persist_dir="./data/chroma_db",
        collection_name="test_collection"
    )
    print(f"Created vector store successfully")

    # Test 2: Similarity search
    print("\n" + "=" * 50)
    print("TEST 2: Similarity search...")
    print("=" * 50)

    query = "What are the risk factors for the company?"
    results = vectorstore.similarity_search(query, k=3)

    print(f"Query: {query}")
    print(f"Found {len(results)} results:\n")

    for i, doc in enumerate(results):
        print(f"[{i+1}] {doc.page_content[:200]}...")
        print(f"    Metadata: {doc.metadata}")
        print()

    # Test 3: Load existing vector store
    print("\n" + "=" * 50)
    print("TEST 3: Loading existing vector store...")
    print("=" * 50)

    loaded_vs = load_vectorstore(
        persist_dir="./data/chroma_db",
        collection_name="test_collection"
    )
    print(f"Loaded vector store successfully")

    # Verify search still works
    results2 = loaded_vs.similarity_search("revenue growth", k=2)
    print(f"Search after reload found {len(results2)} results")
