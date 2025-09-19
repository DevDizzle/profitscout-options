# serving/core/pipelines/data_bundler.py
import logging
import pandas as pd
import json
from collections import defaultdict
from typing import Any, Dict, List
from google.cloud import bigquery, storage
from concurrent.futures import ThreadPoolExecutor, as_completed
from .. import config, bq, gcs

def _copy_blob(blob, source_bucket, destination_bucket, overwrite: bool = False):
    """
    Worker function to copy a single blob.
    """
    try:
        destination_blob = destination_bucket.blob(blob.name)
        
        if not overwrite and destination_blob.exists():
            logging.info(f"Skipping copy, blob already exists: {blob.name}")
            return None

        source_blob = source_bucket.blob(blob.name)
        token, _, _ = destination_blob.rewrite(source_blob)
        while token is not None:
            token, _, _ = destination_blob.rewrite(source_blob, token=token)

        action = "Overwrote" if overwrite else "Copied"
        logging.info(f"Successfully {action} {blob.name}")
        return blob.name
    except Exception as e:
        logging.error(f"Failed to copy {blob.name}: {e}", exc_info=True)
        return None


def _sync_gcs_data():
    """
    Copies all necessary asset files from the source to the destination bucket.
    """
    storage_client = storage.Client()
    source_bucket = storage_client.bucket(config.GCS_BUCKET_NAME, user_project=config.SOURCE_PROJECT_ID)
    destination_bucket = storage_client.bucket(config.DESTINATION_GCS_BUCKET_NAME, user_project=config.DESTINATION_PROJECT_ID)
    
    prefixes_to_sync = [
        "price-chart-json/",
        "sec-business/", "headline-news/", "sec-mda/",
        "financial-statements/", "earnings-call-transcripts/",
        "recommendations/", "pages/", "dashboards/", "images/",
        "fundamentals-analysis/"
    ]
    prefixes_to_overwrite = ["technicals/"]

    logging.info("Starting GCS data sync...")
    
    copied_count = 0
    with ThreadPoolExecutor(max_workers=config.MAX_WORKERS_BUNDLER) as executor:
        blobs_to_sync = [blob for prefix in prefixes_to_sync for blob in source_bucket.list_blobs(prefix=prefix)]
        future_to_blob_sync = {
            executor.submit(_copy_blob, blob, source_bucket, destination_bucket, overwrite=False): blob 
            for blob in blobs_to_sync
        }
        
        blobs_to_overwrite = [blob for prefix in prefixes_to_overwrite for blob in source_bucket.list_blobs(prefix=prefix)]
        future_to_blob_overwrite = {
            executor.submit(_copy_blob, blob, source_bucket, destination_bucket, overwrite=True): blob
            for blob in blobs_to_overwrite
        }
        
        for future in as_completed({**future_to_blob_sync, **future_to_blob_overwrite}):
            if future.result():
                copied_count += 1

    logging.info(f"GCS data sync finished. Copied or overwrote {copied_count} files.")


def _get_latest_daily_files_map() -> Dict[str, Dict[str, str]]:
    """Lists daily files from GCS once and creates a map of the latest file URI for each ticker."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(config.DESTINATION_GCS_BUCKET_NAME, user_project=config.DESTINATION_PROJECT_ID)
    daily_prefixes = {
        "news": "headline-news/",
        "recommendation_analysis": "recommendations/",
        "pages_json": "pages/",
        "dashboard_json": "dashboards/"
    }
    latest_files = defaultdict(dict)
    
    for key, prefix in daily_prefixes.items():
        blobs = bucket.list_blobs(prefix=prefix)
        ticker_files = defaultdict(list)
        for blob in blobs:
            try:
                ticker = blob.name.split('/')[-1].split('_')[0]
                ticker_files[ticker].append(blob.name)
            except IndexError: continue
        
        for ticker, names in ticker_files.items():
            latest_name = max(names)
            latest_files[ticker][key] = f"gs://{config.DESTINATION_GCS_BUCKET_NAME}/{latest_name}"
            
    return latest_files

def _get_latest_kpis() -> Dict[str, Dict[str, Any]]:
    """Reads all recent prep files to get the latest price and 30-day change for each ticker."""
    blobs = gcs.list_blobs(config.GCS_BUCKET_NAME, prefix="prep/")
    latest_kpis = {}
    
    for blob_name in blobs:
        try:
            content = gcs.read_blob(config.GCS_BUCKET_NAME, blob_name)
            if not content: continue
            data = json.loads(content)
            ticker = data.get("ticker")
            if not ticker: continue
            
            if ticker not in latest_kpis or data.get("runDate") > latest_kpis[ticker].get("runDate"):
                kpis = data.get("kpis", {})
                latest_kpis[ticker] = {
                    "price": kpis.get("trendStrength", {}).get("price"),
                    "thirty_day_change_pct": kpis.get("thirtyDayChange", {}).get("value"),
                    "runDate": data.get("runDate")
                }
        except (json.JSONDecodeError, KeyError) as e:
            logging.warning(f"Could not process KPI file {blob_name}: {e}")
            continue
            
    return latest_kpis

def _get_ticker_work_list() -> pd.DataFrame:
    """Gets the base metadata for the latest quarter for each ticker."""
    client = bigquery.Client(project=config.SOURCE_PROJECT_ID)
    query = f"""
        SELECT ticker, company_name, industry, sector, quarter_end_date
        FROM (
            SELECT *, ROW_NUMBER() OVER(PARTITION BY ticker ORDER BY quarter_end_date DESC) as rn
            FROM `{config.BUNDLER_STOCK_METADATA_TABLE_ID}`
            WHERE ticker IS NOT NULL AND quarter_end_date IS NOT NULL
        ) WHERE rn = 1
    """
    return client.query(query).to_dataframe()

def _get_weighted_scores() -> pd.DataFrame:
    """Fetches the latest weighted_score for each ticker."""
    client = bigquery.Client(project=config.SOURCE_PROJECT_ID)
    query = f"""
        SELECT ticker, weighted_score FROM (
            SELECT ticker, weighted_score, ROW_NUMBER() OVER(PARTITION BY ticker ORDER BY run_date DESC) as rn
            FROM `{config.BUNDLER_SCORES_TABLE_ID}`
            WHERE weighted_score IS NOT NULL
        ) WHERE rn = 1
    """
    return client.query(query).to_dataframe()

def _assemble_final_metadata(work_list_df: pd.DataFrame, scores_df: pd.DataFrame, daily_files_map: Dict, kpis_map: Dict) -> List[Dict[str, Any]]:
    """Joins metadata and adds GCS asset URIs and KPIs."""
    if scores_df.empty: return []
    merged_df = pd.merge(work_list_df, scores_df, on="ticker", how="inner")
    final_records = []
    
    for _, row in merged_df.iterrows():
        ticker = row["ticker"]
        quarterly_date_str = row["quarter_end_date"].strftime('%Y-%m-%d')
        record = row.to_dict()

        record["news"] = daily_files_map.get(ticker, {}).get("news")
        record["recommendation_analysis"] = daily_files_map.get(ticker, {}).get("recommendation_analysis")
        record["pages_json"] = daily_files_map.get(ticker, {}).get("pages_json")
        record["dashboard_json"] = daily_files_map.get(ticker, {}).get("dashboard_json")
        
        record["technicals"] = f"gs://{config.DESTINATION_GCS_BUCKET_NAME}/technicals/{ticker}_technicals.json"
        record["profile"] = f"gs://{config.DESTINATION_GCS_BUCKET_NAME}/sec-business/{ticker}_{quarterly_date_str}.json"
        record["mda"] = f"gs://{config.DESTINATION_GCS_BUCKET_NAME}/sec-mda/{ticker}_{quarterly_date_str}.json"
        record["financials"] = f"gs://{config.DESTINATION_GCS_BUCKET_NAME}/financial-statements/{ticker}_{quarterly_date_str}.json"
        record["earnings_transcript"] = f"gs://{config.DESTINATION_GCS_BUCKET_NAME}/earnings-call-transcripts/{ticker}_{quarterly_date_str}.json"
        record["fundamentals"] = f"gs://{config.DESTINATION_GCS_BUCKET_NAME}/fundamentals-analysis/{ticker}_{quarterly_date_str}.json"
        
        record["image_uri"] = f"gs://{config.DESTINATION_GCS_BUCKET_NAME}/images/{ticker}.png"

        # --- THIS IS THE FIX ---
        # The logic is now simpler because we expect clean floats from the KPI file.
        ticker_kpis = kpis_map.get(ticker, {})
        
        try:
            record["price"] = float(ticker_kpis.get("price")) if ticker_kpis.get("price") is not None else None
            record["thirty_day_change_pct"] = float(ticker_kpis.get("thirty_day_change_pct")) if ticker_kpis.get("thirty_day_change_pct") is not None else None
            record["weighted_score"] = float(row.get("weighted_score")) if row.get("weighted_score") is not None else None
        except (ValueError, TypeError):
            # Fallback in case of unexpected non-numeric data
            record["price"] = record.get("price")
            record["thirty_day_change_pct"] = record.get("thirty_day_change_pct")
            record["weighted_score"] = record.get("weighted_score")

        weighted_score = record["weighted_score"]
        if weighted_score is not None:
            if weighted_score > 0.62:
                record["recommendation"] = "BUY"
            elif weighted_score >= 0.43:
                record["recommendation"] = "HOLD"
            else:
                record["recommendation"] = "SELL"
        else:
            record["recommendation"] = None
        
        final_records.append(record)
    return final_records

def run_pipeline():
    """Orchestrates the final assembly and loading of asset metadata."""
    logging.info("--- Starting Data Bundler (Final Assembly) Pipeline ---")
    
    _sync_gcs_data()
    daily_files_map = _get_latest_daily_files_map()
    kpis_map = _get_latest_kpis()
    
    work_list_df = _get_ticker_work_list()
    if work_list_df.empty:
        logging.warning("No tickers in work list. Shutting down.")
        return
        
    scores_df = _get_weighted_scores()
    final_metadata = _assemble_final_metadata(work_list_df, scores_df, daily_files_map, kpis_map)
    
    if not final_metadata:
        logging.warning("No complete records to load to BigQuery.")
        return
        
    df = pd.DataFrame(final_metadata)
    
    bq.upsert_df_to_bq(df, config.BUNDLER_ASSET_METADATA_TABLE_ID, config.DESTINATION_PROJECT_ID)
    
    logging.info(f"--- Data Bundler (Final Assembly) Pipeline Finished ---")