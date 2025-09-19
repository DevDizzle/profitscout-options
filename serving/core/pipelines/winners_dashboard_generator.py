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
ASSET_METADATA_TABLE_ID = f"{config.SOURCE_PROJECT_ID}.{config.BIGQUERY_DATASET}.asset_metadata"
OUTPUT_TABLE_ID = f"{config.DESTINATION_PROJECT_ID}.{config.BIGQUERY_DATASET}.winners_dashboard"

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
    
    # Filter for the specific signals you want on the dashboard
    strong_signals = [
        "Strongly Bullish", "Moderately Bullish",
        "Strongly Bearish", "Moderately Bearish"
    ]
    filtered_df = df[df['outlook_signal'].isin(strong_signals)]
    
    # Get the latest recommendation for each ticker
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
        WHERE setup_quality_signal LIKE '🟢 Strong'
          AND run_date = (SELECT MAX(run_date) FROM `{SIGNALS_TABLE_ID}`)
    """
    try:
        df = client.query(query).to_dataframe()
        return df
    except Exception as e:
        logging.error(f"Failed to query strong options setups: {e}")
        return pd.DataFrame()

def _get_asset_metadata() -> pd.DataFrame:
    """
    Fetches the latest asset metadata for all tickers.
    """
    client = bigquery.Client(project=config.SOURCE_PROJECT_ID)
    query = f"""
        SELECT
            ticker,
            company_name,
            image_uri,
            price AS last_close,
            thirty_day_change_pct,
            industry
        FROM (
            SELECT *, ROW_NUMBER() OVER(PARTITION BY ticker ORDER BY updated_at DESC) as rn
            FROM `{ASSET_METADATA_TABLE_ID}`
        )
        WHERE rn = 1
    """
    try:
        df = client.query(query).to_dataframe()
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

    # 1. Get the filtered list of stocks with strong recommendations
    strong_recs_df = _get_strong_stock_recommendations()
    if strong_recs_df.empty:
        logging.warning("No strong stock recommendations found. Aborting.")
        return

    # 2. Get the list of tickers with strong options setups
    strong_options_df = _get_strong_options_setups()
    if strong_options_df.empty:
        logging.warning("No tickers with strong options setups found. Aborting.")
        return

    # 3. Find the intersection: tickers that are in both lists
    winners_df = pd.merge(strong_recs_df, strong_options_df, on='ticker', how='inner')
    if winners_df.empty:
        logging.warning("No tickers matched between strong recommendations and strong options. Final table will be empty.")
        # Still proceed to truncate the table to ensure old data is cleared
    
    # 4. Enrich with the final metadata for the dashboard
    asset_metadata_df = _get_asset_metadata()
    if not asset_metadata_df.empty:
        final_df = pd.merge(winners_df, asset_metadata_df, on='ticker', how='left')
    else:
        logging.warning("No asset metadata found. Final table may have missing columns.")
        final_df = winners_df

    # 5. Select and rename columns to match the final table schema
    final_df = final_df.rename(columns={"price": "last_close"})
    final_columns = [
        "image_uri", "company_name", "ticker", "outlook_signal",
        "last_close", "thirty_day_change_pct", "industry", "run_date"
    ]
    
    # Ensure all required columns exist, adding them with None if they don't
    for col in final_columns:
        if col not in final_df.columns:
            final_df[col] = None
            
    final_df = final_df[final_columns]

    # 6. Load the data into the BigQuery table
    logging.info(f"Found {len(final_df)} winning tickers. Loading to BigQuery...")
    bq.load_df_to_bq(final_df, OUTPUT_TABLE_ID, config.DESTINATION_PROJECT_ID, write_disposition="WRITE_TRUNCATE")
    
    logging.info(f"--- Winners Dashboard Generation Pipeline Finished ---")