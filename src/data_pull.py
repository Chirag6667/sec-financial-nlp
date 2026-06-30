import os
import logging
from sec_edgar_downloader import Downloader
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("logs/data_pull.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

TICKERS = ['AAPl', 'MSFT', 'GOOGL', "JPM", "TSLA"] 
FILINGS = ["10-K", "8-K"] 
START_DATE = "2020-01-01"
END_DATE = "2024-12-31"

def pull_filings():
    dl = Downloader("ChiragJain", "jainchirag.231@gmail.com", "data/raw")
    for ticker in TICKERS:
        for filing_type in FILINGS:
            logger.info(f"Pulling {filing_type} for {ticker}")
            try:
                dl.get(filing_type, ticker, after=START_DATE, before=END_DATE, limit=20)
                logger.info(f"Successfully pulled {filing_type} filings for {ticker}")
            except Exception as e:
                logger.error(f"Error pulling {filing_type} filings for {ticker}: {e}")

if __name__ == "__main__":
    pull_filings()