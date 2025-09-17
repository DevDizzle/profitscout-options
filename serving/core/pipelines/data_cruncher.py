# serving/core/pipelines/data_cruncher.py
import logging
import pandas as pd
import json
from datetime import date, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from google.cloud import bigquery
from typing import Dict, Any, Optional

from .. import config, gcs

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - [%(threadName)s] - %(message)s")
OUTPUT_PREFIX = "prep/"
MAX_WORKERS = 16

# --- Helper Functions ---

def _get_work_list() -> pd.DataFrame:
    """Queries stock_metadata for the list of tickers to process."""
    logging.info("Fetching work list from stock_metadata...")
    client = bigquery.Client(project=config.SOURCE_PROJECT_ID)
    query = f"""
        SELECT ticker
        FROM `{config.BUNDLER_STOCK_METADATA_TABLE_ID}`
        WHERE ticker IS NOT NULL
        GROUP BY ticker
    """
    df = client.query(query).to_dataframe()
    logging.info(f"Work list created for {len(df)} tickers.")
    return df

def _delete_old_prep_files(ticker: str):
    """Deletes all prep JSON files for a given ticker."""
    prefix = f"{OUTPUT_PREFIX}{ticker}_"
    blobs_to_delete = gcs.list_blobs(config.GCS_BUCKET_NAME, prefix)
    for blob_name in blobs_to_delete:
        try:
            gcs.delete_blob(config.GCS_BUCKET_NAME, blob_name)
        except Exception as e:
            logging.error(f"[{ticker}] Failed to delete old prep file {blob_name}: {e}")

def _fetch_and_calculate_kpis(ticker: str) -> Optional[str]:
    """
    Fetches enriched data for a single ticker and calculates the 4 main KPIs.
    """
    client = bigquery.Client(project=config.SOURCE_PROJECT_ID)
    run_date_str = date.today().strftime('%Y-%m-%d')
    final_json = {"ticker": ticker, "runDate": run_date_str, "kpis": {}}

    try:
        enriched_query = f"""
            SELECT *
            FROM `{config.SOURCE_PROJECT_ID}.{config.BIGQUERY_DATASET}.options_analysis_input`
            WHERE ticker = @ticker
            ORDER BY date DESC
            LIMIT 1
        """
        job_config = bigquery.QueryJobConfig(query_parameters=[bigquery.ScalarQueryParameter("ticker", "STRING", ticker)])
        enriched_df = client.query(enriched_query, job_config=job_config).to_dataframe()

        if enriched_df.empty:
            logging.warning(f"[{ticker}] No enriched analysis data found.")
            return None

        latest_row = enriched_df.iloc[0]

        # --- KPI 1: Trend Strength ---
        price = latest_row.get('adj_close')
        sma50 = latest_row.get('latest_sma50')
        if pd.notna(price) and pd.notna(sma50):
            signal = "bullish" if price > sma50 else "bearish"
            final_json["kpis"]["trendStrength"] = {
                "value": "Above 50D MA" if signal == "bullish" else "Below 50D MA",
                "price": round(price, 2),
                "sma50": round(sma50, 2),
                "signal": signal,
                "tooltip": "Compares current price to the 50-day moving average to identify the current trend."
            }

        # --- KPI 2: RSI (Relative Strength Index) ---
        rsi = latest_row.get('latest_rsi')
        if pd.notna(rsi):
            signal = "overbought" if rsi > 70 else "oversold" if rsi < 30 else "neutral"
            final_json["kpis"]["rsi"] = {
                "value": round(rsi, 2),
                "signal": signal,
                "tooltip": "Indicates if a stock is overbought (>70) or oversold (<30)."
            }

        # --- KPI 3: Volume Surge ---
        volume = latest_row.get('volume')
        avg_volume = latest_row.get('avg_volume_30d')
        if pd.notna(volume) and pd.notna(avg_volume) and avg_volume > 0:
            surge_pct = (volume / avg_volume - 1) * 100
            signal = "high" if surge_pct > 50 else "normal"
            final_json["kpis"]["volumeSurge"] = {
                "value": f"+{round(surge_pct)}%",
                "signal": signal,
                "tooltip": "Today's volume versus its 30-day average."
            }

        # --- KPI 4: 30-Day Historical Volatility (HV) ---
        hv_30 = latest_row.get('hv_30')
        if pd.notna(hv_30):
            final_json["kpis"]["historicalVolatility"] = {
                "value": f"{round(hv_30 * 100, 1)}%",
                "signal": "high" if hv_30 > 0.5 else "low" if hv_30 < 0.2 else "moderate",
                "tooltip": "The stock's actual (realized) volatility over the last 30 days."
            }

        _delete_old_prep_files(ticker)
        output_blob_name = f"{OUTPUT_PREFIX}{ticker}_{run_date_str}.json"
        gcs.write_text(config.GCS_BUCKET_NAME, output_blob_name, json.dumps(final_json, indent=2))
        logging.info(f"[{ticker}] Successfully generated and uploaded prep JSON with 4 KPIs.")
        return output_blob_name

    except Exception as e:
        logging.error(f"[{ticker}] Failed during KPI calculation: {e}", exc_info=True)
        return None

def run_pipeline():
    """Orchestrates the data crunching pipeline."""
    logging.info("--- Starting Data Cruncher (Prep Stage) Pipeline ---")

    work_list_df = _get_work_list()
    if work_list_df.empty:
        logging.warning("No tickers in work list. Exiting.")
        return

    processed_count = 0
    with ThreadPoolExecutor(max_workers=MAX_WORKERS, thread_name_prefix='CruncherWorker') as executor:
        future_to_ticker = {
            executor.submit(_fetch_and_calculate_kpis, row['ticker']): row['ticker']
            for _, row in work_list_df.iterrows()
        }
        for future in as_completed(future_to_ticker):
            try:
                if future.result():
                    processed_count += 1
            except Exception as exc:
                logging.error(f"Worker generated an unhandled exception: {exc}", exc_info=True)

    logging.info(f"--- Data Cruncher Pipeline Finished. Processed {processed_count} of {len(work_list_df)} tickers. ---")