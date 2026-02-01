"""
SEC Data Fetcher CLI

Usage:
    python scripts/fetch_sec_data.py --tickers AAPL,MSFT,GOOGL --count 2
"""

import argparse
from pathlib import Path
import sys

# Add project root to path so we can import src modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.fetcher import fetch_company_filings
from src.config.logging_config import logger

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Fetch SEC 10-K filings for specified companies"
    )
    
    parser.add_argument(
        "--tickers",
        type=str,
        required=True,
        help="Comma-separated list of stock tickers (e.g., AAPL,MSFT,GOOGL)"
    )
    
    parser.add_argument(
        "--count",
        type=int,
        default=2,
        help="Number of filings to fetch per company (default: 2)"
    )
    
    parser.add_argument(
        "--filing-type",
        type=str,
        default="10-K",
        help="Type of filing to fetch (default: 10-K)"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./data/raw",
        help="Output directory for downloaded files (default: ./data/raw)"
    )
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    
    # Parse comma-separated tickers
    tickers = [t.strip().upper() for t in args.tickers.split(",")]
    output_dir = Path(args.output_dir)
    
    logger.info(f"Starting SEC data fetch")
    logger.info(f"Tickers: {tickers}")
    logger.info(f"Filings per company: {args.count}")
    logger.info(f"Filing type: {args.filing_type}")
    logger.info(f"Output directory: {output_dir}")
    
    # Track results
    all_paths = []
    failed_tickers = []
    
    # Fetch filings for each ticker
    for ticker in tickers:
        print(f"\n{'='*50}")
        print(f"Fetching {ticker}...")
        print('='*50)
        
        try:
            paths = fetch_company_filings(
                ticker=ticker,
                filing_type=args.filing_type,
                count=args.count,
                output_dir=output_dir
            )
            
            if paths:
                all_paths.extend(paths)
                print(f"✓ Downloaded {len(paths)} files:")
                for p in paths:
                    print(f"  - {p.name} ({p.stat().st_size / 1024:.1f} KB)")
            else:
                failed_tickers.append(ticker)
                print(f"✗ No files downloaded for {ticker}")
                
        except Exception as e:
            failed_tickers.append(ticker)
            logger.error(f"Error fetching {ticker}: {e}")
            print(f"✗ Error: {e}")
    
    # Print summary
    print(f"\n{'='*50}")
    print("SUMMARY")
    print('='*50)
    print(f"Total files downloaded: {len(all_paths)}")
    print(f"Successful tickers: {len(tickers) - len(failed_tickers)}/{len(tickers)}")
    
    if failed_tickers:
        print(f"Failed tickers: {', '.join(failed_tickers)}")
    
    if all_paths:
        total_size = sum(p.stat().st_size for p in all_paths) / (1024 * 1024)
        print(f"Total size: {total_size:.2f} MB")
        print(f"Files saved to: {output_dir.absolute()}")


if __name__ == "__main__":
    main()