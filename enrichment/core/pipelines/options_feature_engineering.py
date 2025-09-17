# enrichment/core/pipelines/options_feature_engineering.py
import logging
import pandas as pd
from datetime import date
from typing import Optional # <-- ADD THIS LINE
from concurrent.futures import ThreadPoolExecutor, as_completed
from google.cloud import bigquery, storage
from .. import config
from .. import options_analysis_helper as helper

logging.basicConfig(level=logging.INFO)

TICKER_LIST_BUCKET = "profit-scout-data"
TICKER_LIST_PATH = "tickerlist.txt"

def _get_tickers(storage_client: storage.Client) -> list[str]:
    """Gets all tickers from the tickerlist.txt file in GCS."""
    try:
        bucket = storage_client.bucket(TICKER_LIST_BUCKET)
        blob = bucket.blob(TICKER_LIST_PATH)
        content = blob.download_as_text(encoding="utf-8")
        tickers = [line.strip().upper() for line in content.splitlines() if line.strip()]
        return tickers
    except Exception as e:
        logging.error(f"Failed to load tickers from GCS: {e}")
        return []

def _fetch_all_data(tickers: list[str], bq_client: bigquery.Client) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Fetches all options chain and price data for all tickers in a single batch.
    """
    logging.info(f"Fetching bulk data for {len(tickers)} tickers...")
    
    chain_query = f"""
        SELECT *
        FROM `{config.CHAIN_TABLE}`
        WHERE ticker IN UNNEST(@tickers) AND fetch_date = CURRENT_DATE()
    """
    
    price_history_query = f"""
        SELECT ticker, date, open, high, low, adj_close AS close, volume
        FROM `{config.PRICE_TABLE_ID}`
        WHERE ticker IN UNNEST(@tickers) AND date >= DATE_SUB(CURRENT_DATE(), INTERVAL 400 DAY)
    """
    
    job_config = bigquery.QueryJobConfig(
        query_parameters=[bigquery.ArrayQueryParameter("tickers", "STRING", tickers)]
    )
    
    chain_df = bq_client.query(chain_query, job_config=job_config).to_dataframe()
    price_history_df = bq_client.query(price_history_query, job_config=job_config).to_dataframe()
    
    logging.info(f"Fetched {len(chain_df)} chain records and {len(price_history_df)} price records.")
    return chain_df, price_history_df

def _process_ticker(ticker: str, snapshot_date: date, chain_df: pd.DataFrame, price_history_df: pd.DataFrame) -> Optional[dict]:
    """
    Worker function to process data for one ticker IN MEMORY.
    """
    try:
        if chain_df.empty:
            logging.warning(f"No options chain data found for {ticker} for today. Skipping feature engineering.")
            return None
            
        uprice = None
        if "underlying_price" in chain_df.columns:
            u = pd.to_numeric(chain_df["underlying_price"], errors="coerce").dropna()
            if not u.empty: uprice = float(u.iloc[0])

        if uprice is None and not price_history_df.empty:
            uprice = price_history_df['close'].iloc[-1]

        iv_avg = helper.compute_iv_avg_atm(chain_df, uprice, snapshot_date) if uprice else None
        
        hv_30 = helper.compute_hv30(None, ticker, snapshot_date, price_history_df=price_history_df)
        tech = helper.compute_technicals_and_deltas(price_history_df)
        
        iv_signal = None
        if iv_avg is not None and hv_30 is not None:
            try: iv_signal = "high" if iv_avg > (hv_30 + 0.10) else "low"
            except Exception: pass
        
        latest_price_row = price_history_df.sort_values('date').iloc[-1]
        
        final_row = {
            "ticker": ticker, "date": snapshot_date,
            "open": latest_price_row.get('open'),
            "high": latest_price_row.get('high'),
            "low": latest_price_row.get('low'),
            "adj_close": latest_price_row.get('close'),
            "volume": latest_price_row.get('volume'),
            "iv_avg": iv_avg, "hv_30": hv_30,
            "iv_signal": iv_signal, **tech
        }
        return final_row
        
    except Exception as e:
        logging.error(f"Error processing ticker {ticker}: {e}", exc_info=True)
        return None

def run_pipeline():
    """Main pipeline to run the options feature engineering."""
    logging.info("--- Starting Options Feature Engineering Pipeline ---")
    bq_client = bigquery.Client(project=config.PROJECT_ID)
    storage_client = storage.Client(project=config.PROJECT_ID)

    tickers = _get_tickers(storage_client)
    if not tickers:
        logging.warning("No tickers found from tickerlist.txt. Exiting.")
        return

    all_chains_df, all_prices_df = _fetch_all_data(tickers, bq_client)

    chains_by_ticker = {ticker: group for ticker, group in all_chains_df.groupby('ticker')}
    prices_by_ticker = {ticker: group.sort_values('date') for ticker, group in all_prices_df.groupby('ticker')}

    results = []
    snapshot_date = date.today()

    with ThreadPoolExecutor(max_workers=16) as executor:
        future_to_ticker = {
            executor.submit(
                _process_ticker,
                ticker,
                snapshot_date,
                chains_by_ticker.get(ticker, pd.DataFrame()),
                prices_by_ticker.get(ticker, pd.DataFrame())
            ): ticker for ticker in tickers
        }
        for future in as_completed(future_to_ticker):
            result = future.result()
            if result:
                results.append(result)

    if not results:
        logging.warning("No results were generated after processing. Exiting.")
        return

    logging.info(f"Bulk upserting {len(results)} records to BigQuery...")
    helper.upsert_analysis_rows(bq_client, results, enrich_ohlcv=False)
    
    logging.info("--- Backfilling IV Industry Average for consistency (if needed) ---")
    helper.backfill_iv_industry_avg_for_date(bq=bq_client, run_date=snapshot_date)
    
    logging.info("--- Options Feature Engineering Pipeline Finished ---")