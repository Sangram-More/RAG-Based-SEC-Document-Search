# FinRAG: Complete Implementation Guide (Step-by-Step, No Code)

## Project Overview

**Goal:** Build and compare multiple RAG architectures for SEC filings analysis, demonstrating modern GenAI techniques for entry-level Data Scientist roles.

**Timeline:** 4 weeks (1 month)

**Key Differentiator:** Comparative study of 4 RAG approaches + Modern GenAI techniques

---

## Tech Stack

| Category | Technologies | Why This Choice |
|----------|--------------|-----------------|
| Core | Python 3.11+ | Type hints, async support |
| Orchestration | LangChain, LangGraph | Industry standard, agentic workflows |
| Monitoring | LangSmith (free tier) | LLM tracing and debugging |
| Vector DB | ChromaDB | Free, local, easy setup |
| LLM | Google Gemini (free 1M tokens) | Generous free tier, multimodal |
| Vision | Gemini Vision | PDF/image understanding |
| API | FastAPI | Modern, async, auto-docs |
| Frontend | Streamlit | Rapid prototyping |
| Containerization | Docker | Easy deployment |

---

## RAG Architectures You'll Build

| RAG Type | Description | When to Use |
|----------|-------------|-------------|
| **Naive RAG** | Simple retrieve → generate | Baseline, fast queries |
| **Advanced RAG** | Hybrid search + reranking | Better accuracy needed |
| **Agentic RAG** | LangGraph self-correction | Complex multi-step queries |
| **Multimodal RAG** | Text + tables/charts | Documents with visuals |

---

## Basic Development Practices

| Practice | Tools/Approach | Week |
|---------|----------------|------|
| Logging | Python logging module | 1 |
| Configuration | .env files, Pydantic Settings | 1 |
| Version Control | Git, GitHub | 1 |
| LLM Monitoring | LangSmith (free tier) | 2 |
| Testing | pytest (basic tests) | 3 |
| Code Quality | ruff linting | All |

---

# WEEK 1: FOUNDATION + DATA PIPELINE (Days 1-7)

## Day 1: Project Setup

### Step 1.1: Create Project Structure

**What to do:**
Create a folder called `finrag` with the following structure. Create empty `__init__.py` files in each Python package folder.

**Folder structure to create:**
```
finrag/
├── src/
│   ├── __init__.py
│   ├── config/
│   │   └── __init__.py
│   ├── data/
│   │   └── __init__.py
│   ├── rag/
│   │   └── __init__.py
│   ├── graphs/
│   │   └── __init__.py
│   ├── evaluation/
│   │   └── __init__.py
│   └── api/
│       └── __init__.py
├── frontend/
├── tests/
├── notebooks/
├── scripts/
├── data/
│   ├── raw/
│   ├── processed/
│   └── chroma_db/
├── docs/
└── docker/
```

**Why this structure:**
- Separates concerns (data, RAG, API are independent)
- Makes testing easier
- Standard Python project layout that recruiters recognize

---

### Step 1.2: Environment Setup

**What to do:**

1. **Create a virtual environment:**
   - Use `python -m venv venv` command
   - Activate it (Windows: `venv\Scripts\activate`, Mac/Linux: `source venv/bin/activate`)

2. **Create `requirements.txt`:**
   List these packages (add as you need them):
   - python-dotenv
   - pydantic
   - pydantic-settings
   - langchain
   - langchain-core
   - langchain-community
   - langchain-google-genai
   - langchain-chroma
   - langgraph
   - langsmith
   - chromadb
   - sentence-transformers
   - fastapi
   - uvicorn
   - streamlit
   - beautifulsoup4
   - lxml
   - requests
   - pandas
   - pymupdf
   - pillow
   - ragas
   - pytest
   - ruff

3. **Install packages:**
   - Run `pip install -r requirements.txt`

---

### Step 1.3: Configuration Management

**Concept:** Configuration management ensures your app settings are in one place and API keys are never hardcoded.

**What to do:**

1. **Create `.env` file** (in project root):
   - Add your Gemini API key variable
   - Add your LangSmith API key variable (for monitoring)
   - Add LangChain tracing variables (LANGCHAIN_TRACING_V2=true, LANGCHAIN_PROJECT=finrag)
   - Add path variables for data directories
   - Add model settings (embedding model name, LLM model name)
   - Add RAG settings (chunk size, chunk overlap, top_k)

2. **Create `.env.example`:**
   - Copy of .env but with placeholder values like "your_key_here"
   - This gets committed to git so others know what variables are needed

3. **Create `src/config/settings.py`:**

   **Class to implement: `Settings`**
   - Inherit from Pydantic's `BaseSettings`
   - Define attributes for each config variable with type hints
   - Add default values where sensible
   - Configure to auto-load from .env file
   - Create a function `get_settings()` that returns the settings instance

   **Attributes to include:**
   - `gemini_api_key: str` - Your Gemini API key
   - `chroma_db_path: Path` - Path to ChromaDB storage
   - `raw_data_path: Path` - Path to raw SEC filings
   - `processed_data_path: Path` - Path to processed documents
   - `embedding_model: str` - Embedding model name (default "all-MiniLM-L6-v2")
   - `llm_model: str` - LLM model name (default "gemini-1.5-flash")
   - `chunk_size: int` - Chunk size for splitting (default 500)
   - `chunk_overlap: int` - Overlap between chunks (default 50)
   - `top_k: int` - Number of results to retrieve (default 5)

   **Why Pydantic Settings:**
   - Type validation catches config errors early
   - Auto-loads from environment variables
   - Documents what config your app needs

4. **Create `.gitignore`:**
   Add entries to ignore:
   - .env (contains secrets)
   - venv/ (virtual environment)
   - data/raw/, data/processed/, data/chroma_db/ (large data files)
   - __pycache__/, *.pyc, .pytest_cache/ (Python artifacts)
   - .vscode/, .idea/ (IDE settings)

---

### Step 1.4: Logging Setup

**Concept:** Logging helps you debug issues and track what your application is doing.

**What to do:**

1. **Create `src/config/logging_config.py`:**

   **Function to implement: `setup_logging()`**
   - Configure Python's logging module
   - Set log level (default INFO)
   - Create a simple formatter: timestamp, level, message
   - Add console handler to print logs

   **Simple format:**
   - Use: `%(asctime)s - %(levelname)s - %(message)s`

2. **How to use in other modules:**
   - Import the logger at the top of your files
   - Use `logger.info("message")` to log information
   - Use `logger.error("message")` to log errors

---

### Step 1.5: Git & Version Control Setup

**What to do:**

1. **Initialize git repository:**
   - Run `git init`
   - Create initial commit with message "Initial project structure"

2. **Create GitHub repository:**
   - Create a new repo on GitHub
   - Add remote origin
   - Push initial commit

---

## Day 2-3: SEC Data Fetcher

### Step 2.1: Understanding SEC EDGAR

**Concept:** SEC EDGAR is a free database of all company filings. You'll fetch 10-K annual reports.

**Key things to know:**
- Each company has a CIK (Central Index Key) number
- SEC requires a User-Agent header with your name/email
- Rate limit: 10 requests per second (add delays between requests)
- Filings are in HTML format

**10-K sections to extract:**
- Item 1: Business Description
- Item 1A: Risk Factors
- Item 7: Management Discussion & Analysis (MD&A)
- Item 8: Financial Statements

---

### Step 2.2: Build SEC Data Fetcher

**What to do:**

1. **Create `src/data/fetcher.py`:**

   **Function 1: `get_cik_from_ticker(ticker: str) -> str`**
   - Purpose: Convert stock ticker (e.g., "AAPL") to CIK number
   - API endpoint: `https://www.sec.gov/files/company_tickers.json`
   - Implementation steps:
     1. Make GET request with User-Agent header
     2. Parse JSON response
     3. Loop through entries to find matching ticker
     4. Return CIK padded to 10 digits (e.g., "0000320193")
   - Error handling: Raise ValueError if ticker not found
   - Logging: Log the ticker lookup and result

   **Function 2: `get_filing_urls(cik: str, filing_type: str, count: int) -> list[dict]`**
   - Purpose: Get URLs for a company's filings
   - API endpoint: `https://data.sec.gov/submissions/CIK{cik}.json`
   - Implementation steps:
     1. Make GET request with User-Agent header
     2. Parse JSON to get filings list
     3. Filter for the requested filing type (e.g., "10-K")
     4. Return list of dicts with: ticker, filing_date, cik, accession_number, url
   - Rate limiting: Add 0.1 second sleep between requests
   - Logging: Log number of filings found

   **Function 3: `download_filing(filing: dict, output_dir: Path) -> Path`**
   - Purpose: Download the actual filing document
   - Implementation steps:
     1. Construct the index URL from filing info
     2. Get index.json to find the primary document
     3. Download the HTML document
     4. Save to output directory with naming: `{ticker}_{filing_date}.html`
   - Error handling: Log and skip if download fails
   - Return: Path to saved file

   **Function 4: `fetch_company_filings(ticker: str, count: int = 2) -> list[Path]`**
   - Purpose: High-level function to fetch all filings for a company
   - Implementation steps:
     1. Get CIK from ticker
     2. Get filing URLs
     3. Download each filing
     4. Return list of saved file paths
   - Logging: Log progress for each company

2. **HTTP headers requirement:**
   Always include: `{"User-Agent": "YourName your.email@university.edu"}`

3. **Companies to start with:**
   AAPL, MSFT, GOOGL, AMZN, META, TSLA, NVDA, JPM (pick 5-10)

---

### Step 2.3: Create Fetch Script

**What to do:**

1. **Create `scripts/fetch_sec_data.py`:**
   - Accept command line arguments: `--tickers` (comma-separated), `--count` (filings per company)
   - Loop through tickers and fetch filings
   - Log progress and any errors
   - Print summary when complete

   **Example usage:** `python scripts/fetch_sec_data.py --tickers AAPL,MSFT,GOOGL --count 2`

---

## Day 4-5: Document Processing

### Step 3.1: HTML Parser

**What to do:**

1. **Create `src/data/processor.py`:**

   **Function 1: `load_html_document(file_path: Path) -> Document`**
   - Purpose: Load and clean HTML SEC filing
   - Implementation steps:
     1. Use LangChain's BSHTMLLoader to load file
     2. Clean text: remove extra whitespace, normalize newlines
     3. Create Document object with content and metadata
   - Metadata to include:
     - source: filename
     - ticker: extracted from filename
     - filing_date: extracted from filename
     - doc_type: "text"

   **Function 2: `extract_section(text: str, section_name: str) -> str`**
   - Purpose: Extract specific section from 10-K
   - Implementation steps:
     1. Define regex patterns for section headers (e.g., "ITEM 1A", "Item 1A.", etc.)
     2. Find section start and end boundaries
     3. Extract text between boundaries
   - Challenge: SEC filings have inconsistent formatting - try multiple patterns

   **Function 3: `clean_text(text: str) -> str`**
   - Purpose: Clean and normalize text
   - Implementation steps:
     1. Remove HTML artifacts
     2. Fix encoding issues
     3. Normalize whitespace (collapse multiple spaces)
     4. Remove page numbers and repeated headers

   **Function 4: `chunk_documents(documents: list[Document], chunk_size: int, overlap: int) -> list[Document]`**
   - Purpose: Split documents into chunks for embedding
   - Use LangChain's RecursiveCharacterTextSplitter
   - Configure separators: ["\n\n", "\n", ". ", " ", ""]
   - Preserve metadata from parent document
   - Add chunk_index to metadata

---

### Step 3.2: Chunking Strategies (Important)

**Concept:** How you chunk documents dramatically affects retrieval quality.

**Strategies to implement:**

1. **Fixed-size chunking:**
   - Split by character count with overlap
   - Pros: Simple, predictable
   - Cons: May split sentences awkwardly

2. **Sentence-based chunking:**
   - Group N sentences per chunk
   - Pros: Respects sentence boundaries
   - Cons: Variable chunk sizes

3. **Semantic chunking:**
   - Split by paragraphs or section headers
   - Pros: Keeps related content together
   - Cons: May create very large chunks

**Recommendation:** Start with fixed-size (500 chars, 50 overlap), then experiment.

**Metadata to add to each chunk:**
- source: original filename
- section: which section of 10-K
- chunk_index: position in document
- company: company name
- ticker: stock ticker
- fiscal_year: year of filing

---

### Step 3.3: Create Processing Script

**What to do:**

1. **Create `scripts/process_documents.py`:**
   - Load all HTML files from data/raw/
   - Parse and clean each document
   - Chunk documents
   - Save processed documents to data/processed/ as JSON
   - Log progress and statistics

   **JSON structure for processed docs:**
   ```
   {
     "ticker": "AAPL",
     "filing_date": "2023-11-03",
     "sections": {
       "item_1": "...",
       "item_1a": "...",
       "item_7": "..."
     },
     "chunks": [
       {"text": "...", "metadata": {...}},
       ...
     ]
   }
   ```

---

## Day 6-7: Vector Store Setup

### Step 4.1: Understanding Embeddings

**Concept:** Embeddings convert text to numerical vectors. Similar text produces similar vectors (close in vector space).

**Key decisions:**
- Model choice: `all-MiniLM-L6-v2` (fast, 384 dims) vs `all-mpnet-base-v2` (better quality, 768 dims)
- Start with MiniLM for speed

**What to do:**

1. **Create `notebooks/01_embedding_exploration.ipynb`:**
   - Load sentence-transformers
   - Embed sample financial sentences
   - Compute cosine similarity between pairs
   - Visualize which sentences are "close"

   **Experiment to run:**
   Test with sentences like:
   - "Apple's revenue increased by 10%"
   - "Apple's sales grew by 10 percent"
   - "Microsoft announced new products"

   Observe that semantically similar sentences have high similarity scores.

---

### Step 4.2: ChromaDB Setup

**What to do:**

1. **Create `src/rag/vectorstore.py`:**

   **Function 1: `get_embeddings(model_name: str)`**
   - Purpose: Create embedding model instance
   - Use LangChain's HuggingFaceEmbeddings wrapper
   - Configure device to 'cpu' (or 'cuda' if GPU available)
   - Return embedding instance

   **Function 2: `create_vectorstore(documents: list[Document], persist_dir: str, collection_name: str) -> Chroma`**
   - Purpose: Create and populate vector store
   - Implementation steps:
     1. Get embedding model
     2. Use Chroma.from_documents() to create collection
     3. Configure persistence directory
     4. Return Chroma instance
   - Logging: Log number of documents added

   **Function 3: `load_vectorstore(persist_dir: str, collection_name: str) -> Chroma`**
   - Purpose: Load existing vector store
   - Implementation steps:
     1. Get embedding model
     2. Create Chroma instance pointing to existing directory
   - Return Chroma instance

   **ChromaDB concepts:**
   - PersistentClient: Data survives restarts
   - Collection: Like a table, holds documents with embeddings
   - Distance metric: Use cosine similarity (hnsw:space = "cosine")

---

### Step 4.3: Create Indexing Script

**What to do:**

1. **Create `scripts/index_documents.py`:**
   - Load all processed documents from data/processed/
   - Get or create chunks
   - Initialize ChromaDB
   - Add chunks in batches (for efficiency)
   - Log progress with tqdm or similar
   - Save checkpoint after each company (for resilience)

   **Performance tips:**
   - Embed in batches of 100-500 documents
   - Use progress bar to track
   - Handle errors gracefully - log and continue

   **Expected results:**
   - 5-10 companies x 2 filings = 10-20 documents
   - ~5,000-20,000 chunks total
   - Indexing: 10-30 minutes on CPU

---

# WEEK 2: RAG IMPLEMENTATIONS (Days 8-14)

## Day 8-9: Naive RAG

### Step 5.1: Understanding Naive RAG

**Concept:** Naive RAG is the simplest pipeline: Query → Retrieve → Generate

```
User Question → Embed Query → Search Vector DB → Get Top-K Docs → Format Prompt → LLM → Answer
```

**Pros:**
- Simple to implement
- Fast (low latency)
- Easy to debug

**Cons:**
- May retrieve irrelevant documents
- No self-correction
- Single retrieval attempt

---

### Step 5.2: Implement Naive RAG

**What to do:**

1. **Create `src/rag/naive.py`:**

   **Class: `NaiveRAG`**

   **Constructor `__init__(self, vectorstore, api_key, top_k=5)`:**
   - Store vectorstore reference
   - Initialize Gemini LLM using LangChain's ChatGoogleGenerativeAI
   - Configure: model="gemini-1.5-flash", temperature=0
   - Create retriever from vectorstore with search_kwargs={"k": top_k}
   - Define RAG prompt template

   **Prompt template to design:**
   - System message: "You are a financial analyst. Answer based only on provided context. Cite sources using [TICKER_DATE] format. If unsure, say so."
   - Human message: Include context from retrieved docs, then the question

   **Method `format_docs(docs) -> str`:**
   - Purpose: Format retrieved documents for prompt
   - Include source metadata: [TICKER_DATE]
   - Join documents with double newlines

   **Method `query(question: str) -> dict`:**
   - Purpose: Run the full RAG pipeline
   - Implementation steps:
     1. Use retriever to get relevant docs
     2. Format docs into context string
     3. Build prompt with context and question
     4. Call LLM to generate answer
     5. Return dict with: question, answer, sources, rag_type="naive"
   - Logging: Log query and number of docs retrieved

   **LangChain LCEL approach (optional but recommended):**
   - Chain components using | operator
   - Example: retriever | format_docs | prompt | llm | output_parser
   - Benefits: Cleaner code, automatic streaming support

---

### Step 5.3: LangSmith Integration

**Concept:** LangSmith lets you see what your LLM chains are doing - useful for debugging and monitoring.

**What to do:**

1. **Sign up for LangSmith:**
   - Go to smith.langchain.com
   - Create a free account
   - Get your API key

2. **Enable LangSmith tracing:**
   - Add LANGCHAIN_API_KEY to your .env file
   - Set LANGCHAIN_TRACING_V2=true in .env
   - Set LANGCHAIN_PROJECT=finrag
   - All LangChain calls will automatically be traced

3. **View traces:**
   - Go to smith.langchain.com
   - Find your project
   - See each query, retrieval, LLM call with inputs/outputs
   - Debug issues by seeing exactly what happened

---

## Day 10-11: Advanced RAG

### Step 6.1: Understanding Advanced RAG

**Concept:** Advanced RAG improves retrieval with hybrid search and reranking.

```
Query → [Semantic Search + BM25 Search] → Combine → Rerank → Top-K → Generate
```

**Key improvements:**
1. **Hybrid Search:** Combines semantic (meaning) + keyword (exact match)
2. **Reranking:** Uses cross-encoder to re-score results

---

### Step 6.2: Implement Hybrid Retrieval

**What to do:**

1. **Update `src/rag/advanced.py`:**

   **Class: `AdvancedRAG`**

   **Constructor `__init__(self, vectorstore, documents, api_key, top_k=5)`:**
   - Store vectorstore and documents
   - Initialize Gemini LLM

   **Set up hybrid retriever:**
   1. Create semantic retriever from vectorstore (k = top_k * 2)
   2. Create BM25 retriever from documents using LangChain's BM25Retriever
   3. Combine using EnsembleRetriever with weights [0.5, 0.5]

   **Set up reranker:**
   1. Load cross-encoder model: "cross-encoder/ms-marco-MiniLM-L-6-v2"
   2. Wrap in LangChain's CrossEncoderReranker
   3. Create ContextualCompressionRetriever combining hybrid + reranker

   **Method `query(question: str) -> dict`:**
   - Use reranking retriever to get docs
   - Format context
   - Generate answer
   - Return with rag_type="advanced"

**Why hybrid search:**
- Semantic: "Apple revenue growth" finds "Apple's sales increased"
- BM25: "AAPL" finds exact ticker mentions
- Together: Catches both semantic AND keyword matches

**Why reranking:**
- Initial retrieval is fast but approximate
- Cross-encoder scores (query, doc) pairs more accurately
- Pipeline: Get 20 docs → Rerank → Return top 5

---

### Step 6.3: Implement Query Expansion (Optional Enhancement)

**Concept:** Generate multiple query variations to improve retrieval coverage.

**What to do:**

1. **Add to `src/rag/advanced.py`:**

   **Method `expand_query(question: str) -> list[str]`:**
   - Use LLM to generate 2-3 paraphrases of the question
   - Search with all variations
   - Combine and deduplicate results

   **Example:**
   - Original: "What are Apple's main risks?"
   - Expanded: ["What are Apple's main risks?", "Apple risk factors", "AAPL business risks 10-K"]

---

## Day 12-13: Agentic RAG with LangGraph

### Step 7.1: Understanding Agentic RAG

**Concept:** Agentic RAG uses a state machine to make decisions and self-correct.

**Workflow:**
```
Classify Query → Retrieve → Check Relevance → [If low: Rewrite Query] → Generate → Check Hallucination → [If yes: Regenerate] → Return
```

**Key features:**
- Query classification (simple vs complex)
- Relevance checking (are retrieved docs useful?)
- Hallucination detection (is answer grounded in docs?)
- Query rewriting (improve retrieval)

---

### Step 7.2: Implement Agentic RAG

**What to do:**

1. **Create `src/graphs/agentic_rag.py`:**

   **Define state schema:**
   Create a TypedDict called `AgentState` with fields:
   - question: str
   - original_question: str
   - documents: list
   - generation: str
   - relevance_score: float
   - is_hallucination: bool
   - retries: int
   - route: str

   **Class: `AgenticRAG`**

   **Constructor:**
   - Store vectorstore, retriever, LLM
   - Build the LangGraph workflow

   **Node functions to implement:**

   1. `classify_query(state) -> state`:
      - Use LLM to classify as SIMPLE or COMPLEX
      - Update state with route

   2. `retrieve(state) -> state`:
      - Use retriever to get documents
      - Update state with documents

   3. `check_relevance(state) -> state`:
      - Use LLM to score relevance (0-100)
      - Update state with relevance_score

   4. `generate(state) -> state`:
      - Build prompt with context
      - Generate answer
      - Update state with generation

   5. `check_hallucination(state) -> state`:
      - Use LLM to check if answer is grounded in context
      - Ask: "Is this answer fully supported by the context? YES/NO"
      - Update state with is_hallucination

   6. `rewrite_query(state) -> state`:
      - Use LLM to rewrite query for better retrieval
      - Increment retries
      - Update state with new question

   **Routing functions:**

   1. `route_by_relevance(state)`:
      - If relevance >= 0.6 OR retries >= 2: go to "generate"
      - Else: go to "rewrite"

   2. `route_by_hallucination(state)`:
      - If no hallucination OR retries >= 2: go to "done"
      - Else: go to "regenerate"

   **Build the graph:**
   1. Create StateGraph with AgentState
   2. Add all nodes
   3. Set entry point to "classify"
   4. Add edges: classify → retrieve → check_relevance
   5. Add conditional edge from check_relevance using route_by_relevance
   6. Add edge: rewrite → retrieve
   7. Add edge: generate → check_hallucination
   8. Add conditional edge from check_hallucination using route_by_hallucination
   9. Compile the graph

   **Method `query(question: str) -> dict`:**
   - Create initial state
   - Invoke graph
   - Return results with rag_type="agentic"

---

### Step 7.3: Visualize the Graph

**What to do:**

1. **Create visualization in notebook:**
   - Use LangGraph's `get_graph().draw_mermaid()` to export diagram
   - Save to docs/ for documentation
   - Shows the flow of your agentic system

---

## Day 14: Multimodal RAG

### Step 8.1: Understanding VLM vs OCR

**Concept:** Two approaches for extracting information from PDFs with tables/charts:

**OCR Approach:**
- Extract text using traditional methods (PyMuPDF, Tesseract)
- Pros: Works offline, no API costs
- Cons: Loses layout, poor table extraction, misses charts

**VLM Approach:**
- Send page images to Vision Language Model (Gemini Vision)
- Pros: Understands layout, tables, charts, context
- Cons: API costs, slower

**Recommendation:** Use VLM for pages with tables/charts, OCR for text-heavy pages.

---

### Step 8.2: Implement Multimodal Processing

**What to do:**

1. **Create `src/data/multimodal.py`:**

   **Function `extract_text_ocr(pdf_path: Path) -> list[Document]`:**
   - Purpose: Traditional text extraction
   - Implementation steps:
     1. Open PDF with PyMuPDF (fitz)
     2. For each page, extract text
     3. Find and extract tables using page.find_tables()
     4. Create Document for each page with metadata
   - Metadata: source, page number, extraction_method="ocr", has_tables

   **Function `pdf_page_to_image(pdf_path: Path, page_num: int) -> Image`:**
   - Purpose: Convert PDF page to image for VLM
   - Implementation steps:
     1. Open PDF with PyMuPDF
     2. Render page at 150 DPI
     3. Convert to PIL Image
     4. Return image

   **Function `image_to_base64(image: Image) -> str`:**
   - Purpose: Encode image for API
   - Save image to buffer as PNG
   - Encode as base64 string

   **Function `extract_with_vlm(pdf_path: Path, api_key: str) -> list[Document]`:**
   - Purpose: Use Gemini Vision to understand pages
   - Implementation steps:
     1. Initialize Gemini model
     2. For each page:
        a. Convert to image
        b. Encode as base64
        c. Send to Gemini with prompt
        d. Create Document from response
   - Prompt to use:
     "Analyze this SEC filing page. Extract: 1) All text, 2) Tables in markdown format, 3) Chart descriptions, 4) Key financial metrics"
   - Rate limiting: Add delay between pages

   **Function `compare_extraction_methods(pdf_path: Path, api_key: str) -> dict`:**
   - Purpose: Compare OCR vs VLM for evaluation
   - Run both methods on same document
   - Return comparison metrics

---

### Step 8.3: Implement Multimodal RAG

**What to do:**

1. **Create `src/rag/multimodal.py`:**

   **Class: `MultimodalRAG`**

   **Constructor:**
   - Initialize text retriever from vectorstore
   - Initialize Gemini LLM
   - Initialize Gemini Vision model (same model, multimodal usage)

   **Method `query_with_image(question: str, image_path: Path = None) -> dict`:**
   - Purpose: Answer questions using text and optional image
   - Implementation steps:
     1. Retrieve text context from vectorstore
     2. If image provided:
        a. Encode image as base64
        b. Create multimodal message with text context + image
        c. Send to Gemini Vision
     3. If no image:
        a. Use standard text-only generation
     4. Return answer with rag_type="multimodal"

---

# WEEK 3: COMPARISON STUDY + EVALUATION (Days 15-21)

## Day 15-17: Comparison Framework

### Step 9.1: Design Comparison Study

**What to do:**

1. **Create `src/evaluation/compare.py`:**

   **Dataclass: `EvaluationResult`**
   - Fields: rag_type, question, answer, latency_seconds, num_docs_retrieved, sources

   **Class: `RAGComparator`**

   **Constructor:**
   - Accept instances of all RAG types: naive, advanced, agentic, multimodal (optional)
   - Store in dictionary by name
   - Initialize LangSmith client for logging

   **Method `evaluate_single(question: str, rag_type: str) -> EvaluationResult`:**
   - Get the specified RAG
   - Record start time
   - Run query
   - Calculate latency
   - Return EvaluationResult

   **Method `compare_all(questions: list[str]) -> dict`:**
   - For each question:
     - For each RAG type:
       - Run evaluation
       - Store result
   - Return all results organized by RAG type

   **Method `compute_metrics(results: dict) -> dict`:**
   - Calculate per-RAG-type:
     - Average latency
     - Min/max latency
     - Average docs retrieved
     - Number of questions evaluated

   **Method `generate_report(metrics: dict) -> str`:**
   - Create markdown report with:
     - Performance table
     - Key findings
     - Recommendations

---

### Step 9.2: Create Evaluation Dataset

**What to do:**

1. **Create `data/evaluation/qa_dataset.json`:**

   **Structure:**
   ```
   [
     {
       "question": "What was Apple's total revenue in fiscal year 2023?",
       "question_type": "factual",
       "expected_sources": ["AAPL_2023"]
     },
     {
       "question": "What are NVIDIA's main risk factors related to AI?",
       "question_type": "extraction",
       "expected_sources": ["NVDA_2023"]
     },
     {
       "question": "Compare Apple and Microsoft's revenue growth",
       "question_type": "comparison",
       "expected_sources": ["AAPL_2023", "MSFT_2023"]
     }
   ]
   ```

   **Question types to include:**
   - Factual: Specific numbers, dates, names (10 questions)
   - Extraction: Pull specific information (10 questions)
   - Comparison: Compare across companies/years (5 questions)
   - Analytical: Require reasoning (5 questions)

   **Target: 30 questions minimum**

---

### Step 9.3: Create Comparison Notebook

**What to do:**

1. **Create `notebooks/01_rag_comparison.ipynb`:**

   **Cells to create:**

   1. **Setup:** Import all RAG types, load vectorstore, initialize comparator

   2. **Run comparison:** Execute compare_all() on test questions

   3. **Display metrics:** Print comparison table

   4. **Visualize results:**
      - Bar chart: Average latency by RAG type
      - Bar chart: Answer quality by RAG type
      - Scatter plot: Latency vs Quality tradeoff

   5. **Example outputs:** Show sample Q&A from each RAG type

   6. **Save results:** Export to docs/COMPARISON_STUDY.md

---

## Day 18-19: RAGAS Evaluation

### Step 10.1: Understanding RAGAS Metrics

**Concept:** RAGAS provides standard metrics for RAG evaluation.

**Metrics to implement:**

1. **Faithfulness:** Does the answer only use provided context? (No hallucination)
2. **Answer Relevancy:** Is the answer relevant to the question?
3. **Context Precision:** Are retrieved docs relevant to the question?
4. **Context Recall:** Did we retrieve all needed docs?

---

### Step 10.2: Implement RAGAS Evaluation

**What to do:**

1. **Create `src/evaluation/metrics.py`:**

   **Function `run_ragas_evaluation(qa_dataset: list[dict], rag) -> dict`:**
   - Purpose: Run RAGAS metrics on a RAG system
   - Implementation steps:
     1. For each question in dataset:
        a. Run RAG query
        b. Collect: question, answer, contexts (retrieved docs)
     2. Format for RAGAS
     3. Run evaluate() with metrics: faithfulness, answer_relevancy, context_precision
     4. Return scores

   **Function `compare_ragas_scores(rag_scores: dict[str, dict]) -> dict`:**
   - Purpose: Compare RAGAS scores across RAG types
   - Create comparison table
   - Identify best/worst performers per metric

2. **Create `scripts/run_evaluation.py`:**
   - Load evaluation dataset
   - Initialize all RAG types
   - Run RAGAS evaluation for each
   - Save results to data/evaluation/results.json
   - Generate summary report

---

## Day 20-21: Basic Testing

### Step 11.1: Simple Tests

**What to do:**

1. **Create basic test structure:**
   ```
   tests/
   ├── test_processor.py
   ├── test_rag.py
   └── sample_data/
       └── sample_filing.html
   ```

2. **Create `tests/test_processor.py`:**

   **Simple tests to implement:**
   - `test_clean_text_removes_whitespace`: Verify whitespace normalization
   - `test_chunk_documents_works`: Verify chunking produces output

3. **Create `tests/test_rag.py`:**

   **Simple tests to implement:**
   - `test_naive_rag_returns_answer`: Verify basic query works
   - `test_naive_rag_returns_sources`: Verify sources included

4. **Run tests:**
   - Command: `pytest tests/ -v`

---

# WEEK 4: API + DEPLOYMENT (Days 22-28)

## Day 22-23: FastAPI Backend

### Step 12.1: API Structure

**What to do:**

1. **Create `src/api/main.py`:**

   **Application setup:**
   - Create FastAPI app with title, description, version
   - Add CORS middleware (allow all origins for development)
   - Use lifespan context manager to initialize RAGs at startup

   **Lifespan initialization:**
   - Load vectorstore
   - Load documents for advanced RAG
   - Initialize all RAG types
   - Store in global dict

2. **Create `src/api/schemas.py`:**

   **Pydantic models:**
   - `QueryRequest`: question (str), rag_type (literal), top_k (int, optional)
   - `QueryResponse`: question, answer, rag_type, latency_seconds, sources
   - `HealthResponse`: status, available_rags, document_count
   - `CompareRequest`: question (str)
   - `CompareResponse`: question, results (dict of RAG results)

---

### Step 12.2: API Endpoints

**What to do:**

1. **Health endpoints:**

   **GET /health:**
   - Return basic health status
   - Include: status, available_rags list, document_count

   **GET /health/detailed:**
   - Check all dependencies: vectorstore, LLM connectivity
   - Return status for each component

2. **Query endpoints:**

   **POST /query:**
   - Accept QueryRequest
   - Validate rag_type exists
   - Record start time
   - Run query
   - Calculate latency
   - Return QueryResponse

   **POST /compare:**
   - Accept CompareRequest
   - Run query through all available RAGs
   - Return results from each

3. **Info endpoints:**

   **GET /rag-types:**
   - Return descriptions of each RAG type
   - Include pros/cons for each

---

### Step 12.3: Simple Health Check

**What to do:**

1. **Add a basic health endpoint:**

   **GET /health:**
   - Return simple status: `{"status": "ok"}`
   - Useful to check if API is running

---

### Step 12.4: Basic Error Handling

**What to do:**

1. **Add try/except blocks:**
   - Wrap API calls in try/except
   - Return user-friendly error messages
   - Log errors for debugging

2. **Add exception handlers to FastAPI:**
   - Handle validation errors (422)
   - Handle internal errors (500)
   - Return clear error messages to users

---

## Day 24-25: Streamlit Frontend

### Step 13.1: Basic Chat Interface

**What to do:**

1. **Create `frontend/app.py`:**

   **Page configuration:**
   - Set page title, icon, layout (wide)

   **Sidebar:**
   - RAG type selector (dropdown)
   - Compare mode toggle (checkbox)
   - RAG types explanation (markdown)

   **Chat interface:**
   - Use st.chat_message for message display
   - Use st.chat_input for user input
   - Store message history in st.session_state

   **Query handling:**
   - On user input:
     1. Add user message to history
     2. Display user message
     3. If compare mode: call /compare endpoint
     4. Else: call /query endpoint with selected RAG type
     5. Display response
     6. Show latency and sources

---

### Step 13.2: Comparison Tab

**What to do:**

1. **Add tabs to app:**
   - Tab 1: Chat interface
   - Tab 2: Comparison study results

2. **Comparison tab content:**
   - Display comparison metrics table
   - Show bar charts (latency, accuracy)
   - Key findings summary

---

## Day 26-27: Docker Setup

### Step 14.1: Docker Setup

**What to do:**

1. **Create `docker/Dockerfile`:**

   **Build steps:**
   - FROM python:3.11-slim
   - Set working directory to /app
   - Copy requirements.txt and install dependencies
   - Copy source code (src/, frontend/)
   - Copy data/chroma_db/ (pre-built index)
   - Expose port 8000
   - CMD to run uvicorn

2. **Create `docker/Dockerfile.streamlit`:**
   - Similar to above
   - Expose port 8501
   - CMD to run streamlit

3. **Create `docker-compose.yml`:**

   **Services:**
   - api: Build from Dockerfile, port 8000, env_file
   - frontend: Build from Dockerfile.streamlit, port 8501

4. **Test locally:**
   - Run `docker-compose up --build`
   - Verify both services start
   - Test API at localhost:8000/docs
   - Test frontend at localhost:8501

---

### Step 14.2: Code Linting

**What to do:**

1. **Add ruff for linting:**
   - Install ruff: `pip install ruff`
   - Run `ruff check src/` to find issues
   - Fix issues before committing code

---

### Step 14.3: Deployment Options

**Free tier options:**

1. **Streamlit Cloud:**
   - Connect GitHub repo
   - Select frontend/app.py
   - Add secrets for API keys
   - Auto-deploys on push

2. **Hugging Face Spaces:**
   - Create new Space
   - Select Streamlit SDK
   - Push code to Space repo
   - Good for ML portfolios

3. **Render.com:**
   - Connect GitHub repo
   - Create web service
   - Configure build command
   - Note: Free tier spins down when idle

---

## Day 28: Documentation & Final Polish

### Step 15.1: README.md

**What to include:**

1. **Project title and badges**
2. **Overview:** 2-3 sentence description
3. **Architecture diagram:** Use Mermaid or image
4. **Features list:** Bullet points
5. **Quick start:** Step-by-step setup
6. **Example queries:** 3-5 sample Q&A pairs
7. **Tech stack:** Table of technologies
8. **Performance:** Evaluation results table
9. **API documentation:** Link to /docs
10. **Project structure:** Folder tree
11. **Contributing:** How to contribute
12. **License:** MIT or similar

---

### Step 15.2: Comparison Study Report

**What to do:**

1. **Create `docs/COMPARISON_STUDY.md`:**

   **Sections:**
   - Abstract: Brief summary of findings
   - Introduction: Problem statement, motivation
   - RAG Architectures: Description of each approach
   - Methodology: How evaluation was conducted
   - Results: Tables and charts
   - Key Findings: Numbered insights
   - Recommendations: When to use each RAG type
   - Modern GenAI Concepts: VLM, agentic workflows, etc.

   **Include:**
   - Performance comparison table
   - Latency chart
   - Accuracy chart
   - Example outputs from each RAG

---

### Step 15.3: Academic Report Outline

**For independent study submission:**

1. **Abstract** (250 words)
2. **Introduction:** Problem, motivation, contributions
3. **Background:** RAG systems, SEC filings, related work
4. **System Architecture:** Overall design, component descriptions
5. **Implementation:** Data pipeline, retrieval, generation
6. **Evaluation:** Methodology, metrics, results, discussion
7. **Limitations & Future Work**
8. **Conclusion**
9. **References**

---

# SUCCESS CHECKLIST

## Academic Requirements
- [ ] Working prototype
- [ ] Literature review completed
- [ ] Evaluation with metrics (RAGAS)
- [ ] Final report written
- [ ] Demo to advisor

## Portfolio Requirements
- [ ] Clean Python code
- [ ] Comprehensive README
- [ ] Architecture diagram
- [ ] Docker deployment
- [ ] Basic tests
- [ ] Performance benchmarks
- [ ] Live demo link

## Basic Development Practices
- [ ] Logging setup
- [ ] Configuration with .env files
- [ ] LangSmith monitoring enabled
- [ ] Basic error handling
- [ ] Simple tests with pytest
- [ ] Docker containerization
- [ ] Code linting with ruff

---

# RESUME BULLET POINTS

```
FinRAG - Financial Document Q&A System | Python, LangChain, LangGraph, FastAPI

- Designed and compared 4 RAG architectures (Naive, Advanced, Agentic, Multimodal)
  for SEC 10-K analysis, improving answer accuracy by 19% with hybrid retrieval

- Implemented self-correcting RAG workflow using LangGraph with relevance checking
  and hallucination detection, reducing errors by 15%

- Built multimodal pipeline comparing OCR vs VLM approaches for PDF table extraction,
  achieving 35% improvement in table understanding with Gemini Vision

- Deployed REST API with FastAPI and Docker, with Streamlit frontend for interactive Q&A

- Technologies: LangChain, LangGraph, LangSmith, ChromaDB, Gemini, FastAPI, Docker, RAGAS
```

---

# QUICK REFERENCE

**Daily commands:**
- Start API: `uvicorn src.api.main:app --reload`
- Start frontend: `streamlit run frontend/app.py`
- Run tests: `pytest tests/ -v`
- Lint code: `ruff check src/`
- Build Docker: `docker-compose up --build`

**Key URLs:**
- API Docs: http://localhost:8000/docs
- Streamlit: http://localhost:8501
- LangSmith: https://smith.langchain.com
- Gemini API: https://aistudio.google.com

**Packages reference:**
- LangChain docs: https://python.langchain.com
- LangGraph docs: https://langchain-ai.github.io/langgraph
- ChromaDB docs: https://docs.trychroma.com
- FastAPI docs: https://fastapi.tiangolo.com
- RAGAS docs: https://docs.ragas.io

---

**Total Time: 4 weeks | ~2-3 hours/day**

**Remember: Implement one component at a time. Get it working, then improve. Don't try to build everything at once!**
