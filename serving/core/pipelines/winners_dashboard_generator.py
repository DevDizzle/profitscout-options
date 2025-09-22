# serving/core/pipelines/winners_dashboard_generator.py
import logging
import pandas as pd
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from google.cloud import bigquery
from .. import config, gcs, bq

# --- Configuration ---
RECOMMENDATION_PREFIX = "recommendations/"
SIGNALS_TABLE_ID = f"{config.SOURCE_PROJECT_ID}.{config.BIGQUERY_DATASET}.options_analysis_signals"
ASSET_METADATA_TABLE_ID = f"{config.DESTINATION_PROJECT_ID}.{config.BIGQUERY_DATASET}.asset_metadata"
OUTPUT_TABLE_ID = f"{config.SOURCE_PROJECT_ID}.{config.BIGQUERY_DATASET}.winners_dashboard"

# --- Main Logic ---

def _get_strong_stock_recommendations() -> pd.DataFrame:
    """
    Fetches all companion JSON files and filters them for strong bullish or bearish signals.
    """
    all_rec_jsons = gcs.list_blobs(config.GCS_BUCKET_NAME, prefix=RECOMMENDATION_PREFIX)
    json_paths = [path for path in all_rec_jsons if path.endswith('.json')]
    
    if not json_paths:
        logging.warning("No recommendation JSON files found.")
        return pd.DataFrame()

    def read_json_blob(blob_name):
        try:
            content = gcs.read_blob(config.GCS_BUCKET_NAME, blob_name)
            return json.loads(content) if content else None
        except Exception as e:
            logging.error(f"Failed to read or parse {blob_name}: {e}")
            return None

    all_data = []
    with ThreadPoolExecutor(max_workers=16) as executor:
        future_to_path = {executor.submit(read_json_blob, path): path for path in json_paths}
        for future in as_completed(future_to_path):
            data = future.result()
            if data:
                all_data.append(data)

    if not all_data:
        logging.warning("Could not parse any recommendation data.")
        return pd.DataFrame()

    df = pd.DataFrame(all_data)
    
    strong_signals = [
        "Strongly Bullish", "Moderately Bullish",
        "Strongly Bearish", "Moderately Bearish"
    ]
    filtered_df = df[df['outlook_signal'].isin(strong_signals)]
    
    latest_df = filtered_df.sort_values('run_date', ascending=False).drop_duplicates('ticker')
    
    return latest_df

def _get_strong_options_setups() -> pd.DataFrame:
    """
    Queries the options signals table to find all tickers that have at least
    one 'Strong' setup in the most recent run.
    """
    client = bigquery.Client(project=config.SOURCE_PROJECT_ID)
    query = f"""
        SELECT DISTINCT ticker
        FROM `{SIGNALS_TABLE_ID}`
        WHERE setup_quality_signal = 'Strong'
          AND run_date = (SELECT MAX(run_date) FROM `{SIGNALS_TABLE_ID}`)
    """
    try:
        df = client.query(query).to_dataframe()
        return df
    except Exception as e:
        logging.error(f"Failed to query strong options setups: {e}")
        return pd.DataFrame()

def _get_asset_metadata(tickers: list) -> pd.DataFrame:
    """
    Fetches the asset metadata for a specific list of tickers.
    """
    client = bigquery.Client(project=config.SOURCE_PROJECT_ID)
    # --- THIS IS THE FIX ---
    # Simplified the query to a direct select since we only have one row per ticker.
    query = f"""
        SELECT
            ticker,
            company_name,
            image_uri,
            price AS last_close,
            thirty_day_change_pct,
            industry,
            weighted_score
        FROM `{ASSET_METADATA_TABLE_ID}`
        WHERE ticker IN UNNEST(@tickers)
    """
    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ArrayQueryParameter("tickers", "STRING", tickers),
        ]
    )
    try:
        df = client.query(query, job_config=job_config).to_dataframe()
        return df
    except Exception as e:
        logging.error(f"Failed to query asset metadata: {e}")
        return pd.DataFrame()

def run_pipeline():
    """
    Orchestrates the creation of the 'winners_dashboard' table by joining
    strong stock recommendations with strong options setups.
    """
    logging.info("--- Starting Winners Dashboard Generation Pipeline ---")

    strong_recs_df = _get_strong_stock_recommendations()
    if strong_recs_df.empty:
        logging.warning("No strong stock recommendations found. Aborting.")
        return

    strong_options_df = _get_strong_options_setups()
    if strong_options_df.empty:
        logging.warning("No tickers with strong options setups found. Aborting.")
        return

    winners_df = pd.merge(strong_recs_df, strong_options_df, on='ticker', how='inner')
    if winners_df.empty:
        logging.warning("No tickers matched between strong recommendations and strong options. Final table will be empty.")
        # We still proceed to write an empty dataframe to truncate the table.
        
    # Only fetch metadata if we have winners to enrich
    if not winners_df.empty:
        winner_tickers = winners_df['ticker'].tolist()
        asset_metadata_df = _get_asset_metadata(winner_tickers)
        if not asset_metadata_df.empty:
            final_df = pd.merge(winners_df, asset_metadata_df, on='ticker', how='left')
        else:
            logging.warning("No asset metadata found for winning tickers. Final table may have missing columns.")
            final_df = winners_df
    else:
        final_df = winners_df

    # Rename for consistency with the final schema
    if "price" in final_df.columns:
        final_df = final_df.rename(columns={"price": "last_close"})

    final_columns = [
        "image_uri", "company_name", "ticker", "outlook_signal",
        "last_close", "thirty_day_change_pct", "industry", "run_date", "weighted_score"
    ]
    
    # Ensure all required columns exist, adding them with None if they don't
    for col in final_columns:
        if col not in final_df.columns:
            final_df[col] = None
            
    final_df = final_df[final_columns]

    logging.info(f"Found {len(final_df)} winning tickers. Loading to BigQuery...")
    bq.load_df_to_bq(final_df, OUTPUT_TABLE_ID, config.SOURCE_PROJECT_ID, write_disposition="WRITE_TRUNCATE")
    
    logging.info(f"--- Winners Dashboard Generation Pipeline Finished ---")