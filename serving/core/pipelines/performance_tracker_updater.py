# serving/core/pipelines/performance_tracker_updater.py
import logging
from datetime import date
import pandas as pd
from google.cloud import bigquery
from .. import config, bq

# --- Configuration ---
SIGNALS_TABLE_ID = f"{config.SOURCE_PROJECT_ID}.{config.BIGQUERY_DATASET}.options_analysis_signals"
OPTIONS_CHAIN_TABLE_ID = f"{config.SOURCE_PROJECT_ID}.{config.BIGQUERY_DATASET}.options_chain"
TRACKER_TABLE_ID = f"{config.SOURCE_PROJECT_ID}.{config.BIGQUERY_DATASET}.performance_tracker"

def _get_new_and_active_contracts(bq_client: bigquery.Client) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    1. Fetches today's new "Strong" signals that are not already in the tracker.
    2. Fetches all contracts currently marked as 'Active' in the tracker.
    """
    today_iso = date.today().isoformat()

    # --- THIS IS THE FIX ---
    # The query now explicitly casts the `run_date` field to a DATE before comparing.
    new_signals_query = f"""
        SELECT
            s.contract_symbol,
            s.ticker,
            CAST(s.run_date AS DATE) as run_date,
            CAST(s.expiration_date AS DATE) as expiration_date,
            s.option_type,
            s.strike_price,
            s.stock_price_trend_signal,
            s.setup_quality_signal
        FROM `{SIGNALS_TABLE_ID}` s
        LEFT JOIN `{TRACKER_TABLE_ID}` t ON s.contract_symbol = t.contract_symbol
        WHERE CAST(s.run_date AS DATE) = @today
          AND s.setup_quality_signal = 'Strong'
          AND (
            (s.stock_price_trend_signal LIKE '%Bullish%' AND s.option_type = 'call') OR
            (s.stock_price_trend_signal LIKE '%Bearish%' AND s.option_type = 'put')
          )
          AND t.contract_symbol IS NULL
    """

    active_contracts_query = f"""
        SELECT
            contract_symbol,
            run_date,
            expiration_date,
            initial_price
        FROM `{TRACKER_TABLE_ID}`
        WHERE status = 'Active'
    """

    job_config = bigquery.QueryJobConfig(query_parameters=[bigquery.ScalarQueryParameter("today", "DATE", today_iso)])
    new_signals_df = bq_client.query(new_signals_query, job_config=job_config).to_dataframe()
    active_contracts_df = bq_client.query(active_contracts_query).to_dataframe()

    return new_signals_df, active_contracts_df

def _get_current_prices(bq_client: bigquery.Client, contracts_df: pd.DataFrame) -> pd.DataFrame:
    """
    Fetches the latest mid-price for a given list of contract symbols.
    """
    if contracts_df.empty:
        return pd.DataFrame()

    contract_symbols = contracts_df['contract_symbol'].unique().tolist()

    query = f"""
        SELECT
            contract_symbol,
            (bid + ask) / 2 AS current_price
        FROM `{OPTIONS_CHAIN_TABLE_ID}`
        WHERE contract_symbol IN UNNEST(@contract_symbols)
          AND fetch_date = (SELECT MAX(fetch_date) FROM `{OPTIONS_CHAIN_TABLE_ID}`)
          AND bid > 0 AND ask > 0
    """

    job_config = bigquery.QueryJobConfig(
        query_parameters=[bigquery.ArrayQueryParameter("contract_symbols", "STRING", contract_symbols)]
    )
    prices_df = bq_client.query(query, job_config=job_config).to_dataframe()
    return prices_df

def _upsert_with_merge(bq_client: bigquery.Client, df: pd.DataFrame):
    """
    Performs a MERGE operation to insert new rows and update existing ones.
    """
    if df.empty:
        logging.info("No data to upsert. Skipping MERGE operation.")
        return

    # Ensure date columns are in the correct format for BigQuery MERGE
    for col in ['run_date', 'expiration_date']:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col]).dt.date

    temp_table_id = f"{TRACKER_TABLE_ID}_temp_staging"
    
    load_job_config = bigquery.LoadJobConfig(write_disposition="WRITE_TRUNCATE")
    bq_client.load_table_from_dataframe(df, temp_table_id, job_config=load_job_config).result()

    merge_sql = f"""
    MERGE `{TRACKER_TABLE_ID}` T
    USING `{temp_table_id}` S ON T.contract_symbol = S.contract_symbol
    WHEN MATCHED THEN
        UPDATE SET
            T.current_price = S.current_price,
            T.percent_gain = S.percent_gain,
            T.status = S.status,
            T.last_updated = CURRENT_TIMESTAMP()
    WHEN NOT MATCHED THEN
        INSERT (
            contract_symbol, ticker, run_date, expiration_date, option_type, strike_price,
            stock_price_trend_signal, setup_quality_signal, initial_price, current_price,
            percent_gain, status, last_updated
        ) VALUES (
            S.contract_symbol, S.ticker, S.run_date, S.expiration_date, S.option_type, S.strike_price,
            S.stock_price_trend_signal, S.setup_quality_signal, S.initial_price, S.current_price,
            S.percent_gain, S.status, CURRENT_TIMESTAMP()
        )
    """
    try:
        merge_job = bq_client.query(merge_sql)
        merge_job.result()
        logging.info(f"MERGE complete. {merge_job.num_dml_affected_rows} rows affected in {TRACKER_TABLE_ID}.")
    finally:
        bq_client.delete_table(temp_table_id, not_found_ok=True)


def run_pipeline():
    """
    Orchestrates the daily update of the performance tracker table.
    """
    logging.info("--- Starting Performance Tracker Update Pipeline ---")
    bq_client = bigquery.Client(project=config.SOURCE_PROJECT_ID)
    today = date.today()

    new_signals_df, active_contracts_df = _get_new_and_active_contracts(bq_client)

    if not new_signals_df.empty:
        initial_prices_df = _get_current_prices(bq_client, new_signals_df)
        new_signals_df = pd.merge(new_signals_df, initial_prices_df, on='contract_symbol', how='left')
        new_signals_df.rename(columns={'current_price': 'initial_price'}, inplace=True)
        new_signals_df['current_price'] = new_signals_df['initial_price']
        new_signals_df['status'] = 'Active'
        new_signals_df['percent_gain'] = 0.0
    
    if not active_contracts_df.empty:
        current_prices_df = _get_current_prices(bq_client, active_contracts_df)
        active_contracts_df = pd.merge(active_contracts_df, current_prices_df, on='contract_symbol', how='left')

        active_contracts_df['percent_gain'] = (
            (active_contracts_df['current_price'] - active_contracts_df['initial_price']) /
             active_contracts_df['initial_price'] * 100
        ).where(active_contracts_df['initial_price'] != 0)

        active_contracts_df['status'] = 'Active'
        active_contracts_df['expiration_date'] = pd.to_datetime(active_contracts_df['expiration_date']).dt.date
        active_contracts_df.loc[active_contracts_df['expiration_date'] < today, 'status'] = 'Expired'
        active_contracts_df.loc[active_contracts_df['current_price'].isnull(), 'status'] = 'Delisted'

    final_df = pd.concat([new_signals_df, active_contracts_df], ignore_index=True)

    if final_df.empty:
        logging.info("No new or active contracts to update today.")
        logging.info("--- Performance Tracker Update Pipeline Finished ---")
        return
        
    logging.info(f"Preparing to upsert {len(final_df)} records into the performance tracker.")
    _upsert_with_merge(bq_client, final_df)

    logging.info("--- Performance Tracker Update Pipeline Finished ---")