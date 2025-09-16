# enrichment/core/pipelines/options_feature_engineering.py
import logging
import pandas as pd
from datetime import date
from concurrent.futures import ThreadPoolExecutor, as_completed
from google.cloud import bigquery
from .. import config
from .. import options_analysis_helper as helper

logging.basicConfig(level=logging.INFO)

def _get_tickers_with_options() -> list[str]:
    """Gets all tickers that have options data for today."""
    client = bigquery.Client(project=config.PROJECT_ID)
    query = f"""
        SELECT DISTINCT ticker
        FROM `{config.CHAIN_TABLE}`
        WHERE fetch_date = CURRENT_DATE()
    """
    try:
        df = client.query(query).to_dataframe()
        return df['ticker'].tolist()
    except Exception as e:
        logging.error(f"Failed to get tickers with options: {e}")
        return []

def _get_full_chain_for_ticker(ticker: str, bq_client: bigquery.Client) -> pd.DataFrame:
    """Fetches the full options chain for a given ticker for today."""
    query = f"""
        SELECT *
        FROM `{config.CHAIN_TABLE}`
        WHERE ticker = @ticker AND fetch_date = CURRENT_DATE()
    """
    job_config = bigquery.QueryJobConfig(
        query_parameters=[bigquery.ScalarQueryParameter("ticker", "STRING", ticker)]
    )
    return bq_client.query(query, job_config=job_config).to_dataframe()

def _process_ticker(ticker: str, bq_client: bigquery.Client):
    """Worker function to process one ticker."""
    try:
        logging.info(f"Processing ticker: {ticker}")
        chain_df = _get_full_chain_for_ticker(ticker, bq_client)
        if not chain_df.empty:
            helper.build_and_upsert_for_ticker(
                bq=bq_client,
                ticker=ticker,
                snapshot_date=date.today(),
                full_chain_df=chain_df
            )
            logging.info(f"Successfully processed ticker: {ticker}")
    except Exception as e:
        logging.error(f"Error processing ticker {ticker}: {e}", exc_info=True)

def run_pipeline():
    """Main pipeline to run the options feature engineering."""
    logging.info("--- Starting Options Feature Engineering Pipeline ---")
    bq_client = bigquery.Client(project=config.PROJECT_ID)
    tickers = _get_tickers_with_options()

    if not tickers:
        logging.warning("No tickers with options found for today.")
        return

    with ThreadPoolExecutor(max_workers=16) as executor:
        futures = [executor.submit(_process_ticker, ticker, bq_client) for ticker in tickers]
        for future in as_completed(futures):
            future.result() # Raise exceptions if any

    logging.info("--- Backfilling IV Industry Average for consistency ---")
    helper.backfill_iv_industry_avg_for_date(bq=bq_client, run_date=date.today())
    logging.info("--- Options Feature Engineering Pipeline Finished ---")