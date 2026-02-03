import requests
import time
import re
from pathlib import Path
from src.config.logging_config import logger

# SEC requires this header for all requests
# HEADERS = {"User-Agent": "sangram.more@colorado.edu"}
# SEC requires both name and email for identification.
HEADERS = {"User-Agent": "Sangram More sangram.more@colorado.edu"}


def get_cik_from_ticker(ticker: str) -> str:
    """
    Convert stock ticker to their respective CIK number.
    
    Args:
        ticker: Stock symbol (e.g., "AAPL")
    
    Returns:
        CIK padded to 10 digits (e.g., "0000320193")
    
    Raises:
        ValueError: If ticker not found
    """

    logger.info(f"Looking up CIK for ticker: {ticker}")

    # Making GET request
    url = "https://www.sec.gov/files/company_tickers.json"
    response = requests.get(url, headers=HEADERS)
    response.raise_for_status() # Raises error if request fails

    # Parsing JSON
    data = response.json()

    # Looping through all entries to find the matching ticker
    ticker_upper = ticker.upper()
    for entry in data.values():
        if entry["ticker"] == ticker_upper:
            # return CIK number padded to 10 digits.
            cik = str(entry["cik_str"]).zfill(10)
            logger.info(f"Found CIK for {ticker}: {cik}")
            return cik
    
    # Handling errors.
    logger.error(f"Ticker not found: {ticker}")
    raise ValueError(f"Ticker '{ticker}' not found in SEC database")

# ----------------------------------------------------------------------------

def get_10k_document_url(cik: str, accession_number: str) -> str | None:
    """
    Get the URL for the human-readable 10-K document from the filing index.
    """
    accession_no_dashes = accession_number.replace("-", "")
    index_url = f"https://www.sec.gov/Archives/edgar/data/{cik}/{accession_no_dashes}/index.json"
    
    try:
        response = requests.get(index_url, headers=HEADERS)
        response.raise_for_status()
        data = response.json()
        
        items = data.get("directory", {}).get("item", [])
        
        # Look for 10-K document (exclude exhibits)
        for item in items:
            name = item.get("name", "").lower()
            # Must end in .htm, contain "10k" or "10-k", but NOT be an exhibit
            if name.endswith(".htm"):
                if ("10k" in name or "10-k" in name):
                    # Skip exhibit files
                    if "exhibit" in name or "ex-" in name or "ex10" in name or "ex21" in name:
                        continue
                    return f"https://www.sec.gov/Archives/edgar/data/{cik}/{accession_no_dashes}/{item['name']}"
        
        # Fallback: find the largest .htm file that's not an exhibit
        htm_files = []
        for item in items:
            name = item.get("name", "").lower()
            if name.endswith(".htm"):
                # Skip exhibits
                if "exhibit" in name or "ex-" in name or "ex10" in name or "ex21" in name:
                    continue
                htm_files.append(item)
        
        if htm_files:
            largest = max(htm_files, key=lambda x: int(x.get("size", 0)))
            return f"https://www.sec.gov/Archives/edgar/data/{cik}/{accession_no_dashes}/{largest['name']}"
        
        return None
    except Exception as e:
        logger.warning(f"Could not get filing index: {e}")
        return None

# ----------------------------------------------------------------------------

def get_filing_urls(cik: str, filing_type: str = "10-K", count: int = 2) -> list[dict]:
    """
    Get URLs from a company's SEC filings.

    Args:
        cik: company's CIK number (usually a 10 digit number)
        filing_type: Type of filing (eg: "10-K", "10-Q") 
        count: Number of filings to be returned 
    
    Returns:
        returns list of dictionaries with filing date, cik, accession number, url
     
    """

    logger.info(f"Fetching {filing_type} filings for CIK: {cik}")

    # Making GET Request
    url = f"https://data.sec.gov/submissions/CIK{cik}.json"
    response = requests.get(url, headers=HEADERS)
    response.raise_for_status()

    # Parsing json
    data = response.json()

    # Get company info.
    ticker = data.get("tickers", ["UNKNOWN"])[0]

    # Get filing data.
    recent_filings = data["filings"]["recent"]

    # Apply filter to get requested file type from collected results
    results = []

    for i in range(len(recent_filings["form"])):
        if recent_filings["form"][i] == filing_type:
            accession_number = recent_filings["accessionNumber"][i]
            filing_date = recent_filings["filingDate"][i]
            
            # Use helper to get correct document URL
            doc_url = get_10k_document_url(cik, accession_number)
            
            # Fallback to primaryDocument if helper fails
            if not doc_url:
                primary_doc = recent_filings["primaryDocument"][i]
                accession_no_dashes = accession_number.replace("-", "")
                doc_url = f"https://www.sec.gov/Archives/edgar/data/{cik}/{accession_no_dashes}/{primary_doc}"
            
            # Rate limiting for index requests
            time.sleep(0.1)

            
            results.append({
                "ticker": ticker,
                "cik": cik,
                "filing_date": filing_date,
                "accession_number": accession_number,
                "url": doc_url
            })
            
            # Stop if we have enough data
            if len(results) >= count:
                break

    # requests rate limiting.

    time.sleep(0.1)
    
    logger.info(f"Found {len(results)} {filing_type} filings for {ticker}")
    return results

# ----------------------------------------------------------------------------

def download_filing(filing: dict, output_dir: Path) -> Path | None:
    """
    Download the actual filing document and associated images (charts).

    Args:
        filing: Dictionary with keys (ticker, cik, filing_date, accession_number, url)
        output_dir: Directory location to save the file

    Returns:
        Path to saved file, or none if download fails
    """

    ticker = filing["ticker"]
    filing_date = filing["filing_date"]
    url = filing["url"]

    logger.info(f"Downloading {ticker} filing from {filing_date}")

    try:
        # Downloading the html document.
        response = requests.get(url, headers=HEADERS)
        response.raise_for_status()

        # Create output directory if it does not already exists.
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save with naming format: {ticker}_{filing_date}.html
        filename = f"{ticker}_{filing_date}.html"
        file_path = output_dir / filename

        # writing content to the file
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(response.text)

        logger.info(f"Saved file to {file_path}")

        # Download associated images (charts)
        base_url = url.rsplit("/", 1)[0]  # Get directory URL
        image_refs = re.findall(r'src=["\']([^"\']+\.(?:jpg|jpeg|png|gif))["\']', response.text, re.IGNORECASE)

        for img_ref in image_refs:
            # Skip external URLs
            if img_ref.startswith("http"):
                continue

            img_url = f"{base_url}/{img_ref}"
            img_path = output_dir / img_ref

            try:
                img_response = requests.get(img_url, headers=HEADERS)
                img_response.raise_for_status()

                with open(img_path, "wb") as img_file:
                    img_file.write(img_response.content)

                logger.info(f"Downloaded image: {img_ref}")
                time.sleep(0.1)  # Rate limiting
            except requests.RequestException as e:
                logger.warning(f"Failed to download image {img_ref}: {e}")

        # Rate Limiting
        time.sleep(0.1)

        return file_path

    except requests.RequestException as e:
        # Error handling, logging and returning none
        logger.error(f"Failed to download {ticker} dated {filing_date}: {e}")
        return None

# ----------------------------------------------------------------------------

def fetch_company_filings(ticker: str, filing_type: str = "10-K", count: int = 2, output_dir: Path = Path("./data/raw")) -> list[Path]:
    """
    A high level function to fetch all the filings for a company.

    Args:
        ticker: Stock symbol (eg: "AAPL")
        filing_type: Type of filing recoard (eg: "10-K", "10-Q")
        count: Number of filings to be downloaded.
        output_dir: Directiory to save file into.

    Returns:
        Returns a list of Paths to saved files
    """
    logger.info(f"Starting fetch for {ticker} - {count} {filing_type} filings")

    # Get CIK number from ticker
    try:
        cik = get_cik_from_ticker(ticker)
    except ValueError as e:
        logger.error(f"Failed to get CIK number for {ticker}: {e}")
        return[]
    
    # Get filing urls
    filings = get_filing_urls(cik, filing_type=filing_type, count=count)

    if not filings:
        logger.warning(f"No {filing_type} filings found for {ticker}")
        return []
    
    # Download each filings
    saved_paths = []

    for i, filing in enumerate(filings, 1):
        logger.info(f"Downloading {i}/{len(filings)}: {filing['filing_date']}")
        path = download_filing(filing, output_dir)

        if path:
            saved_paths.append(path)
        else:
            logger.warning(f"Skipping {ticker}: {len(saved_paths)}/{len(filings)} files downloaded")

    # Logging summary
    logger.info(f"Completed {ticker}: {len(saved_paths)}/{len(filings)} files downloaded")
    return saved_paths

# ----------------------------------------------------------------------------

"""
Standard SEC URL Response:

{
  "cik": "320193",
  "name": "Apple Inc.",
  "tickers": ["AAPL"],
  "filings": {
    "recent": {
      "accessionNumber": ["0000320193-23-000106", "0000320193-23-000077", ...],
      "filingDate": ["2023-11-03", "2023-08-04", ...],
      "form": ["10-K", "10-Q", ...],
      "primaryDocument": ["aapl-20230930.htm", ...]
    }
  }
}

"""

# ----------------------------------------------------------------------------

# =============================================================================
# TEST CODE - Comment out or remove in production
# =============================================================================
if __name__ == "__main__":

    # This only runs when you execute: python src/data/fetcher.py
    # It won't run when the module is imported elsewhere

    # Companies to fetch (pick 5-10 from the list)
    # COMPANIES = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "NVDA", "JPM"]
    COMPANIES = ["AAPL", "MSFT", "GOOGL"]
    
    all_paths = []
    
    for ticker in COMPANIES:
        print(f"\n{'='*50}")
        print(f"Fetching {ticker}...")
        print('='*50)
        
        paths = fetch_company_filings(
            ticker=ticker,
            filing_type="10-K",
            count=2,
            output_dir=Path("./data/raw")
        )
        
        all_paths.extend(paths)
        
        print(f"Downloaded {len(paths)} files:")
        for p in paths:
            print(f"  - {p.name} ({p.stat().st_size / 1024:.1f} KB)")
    
    # Final summary
    print(f"\n{'='*50}")
    print(f"TOTAL: Downloaded {len(all_paths)} files for {len(COMPANIES)} companies")
    print('='*50)



# -------------------------------------------------------------------------------
    # test_tickers = ["AAPL", "MSFT", "GOOGL", "INVALID_TICKER"]
    # for ticker in test_tickers:
    #     try:
    #         cik = get_cik_from_ticker(ticker)
    #         print(f"✓ {ticker} -> {cik}")
    #     except ValueError as e:
    #         print(f"✗ {ticker} -> Error: {e}")
# -------------------------------------------------------------------------------
    # Test get_filing_urls
    # cik = get_cik_from_ticker("AAPL")
    # filings = get_filing_urls(cik, filing_type="10-K", count=2)
    
    # print("\n10-K Filings:")
    # for f in filings:
    #     print(f"  {f['filing_date']}: {f['url']}")
# --------------------------------------------------------------------------------
# Test download_filing
    # cik = get_cik_from_ticker("AAPL")
    # filings = get_filing_urls(cik, filing_type="10-K", count=1)
    
    # if filings:
    #     output_dir = Path("./data/raw")
    #     saved_path = download_filing(filings[0], output_dir)
        
    #     if saved_path:
    #         print(f"\n✓ Downloaded to: {saved_path}")
    #         print(f"  File size: {saved_path.stat().st_size / 1024:.1f} KB")
    #     else:
    #         print("✗ Download failed")
# --------------------------------------------------------------------------------
# Test fetch_company_filings
    # tickers = ["AAPL", "MSFT"]
    
    # for ticker in tickers:
    #     print(f"\n{'='*50}")
    #     print(f"Fetching {ticker}...")
    #     print('='*50)
        
    #     paths = fetch_company_filings(
    #         ticker=ticker,
    #         filing_type="10-K",
    #         count=2,
    #         output_dir=Path("./data/raw")
    #     )
        
    #     print(f"\nDownloaded {len(paths)} files:")
    #     for p in paths:
    #         print(f"  - {p.name} ({p.stat().st_size / 1024:.1f} KB)")
# --------------------------------------------------------------------------------