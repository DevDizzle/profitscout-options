# serving/core/pipelines/performance_tracker_updater.py
import logging
from datetime import date
import pandas as pd
from google.cloud import bigquery
from .. import config, bq

# --- Configuration ---
WINNERS_TABLE_ID = f"{config.DESTINATION_PROJECT_ID}.{config.BIGQUERY_DATASET}.winners_dashboard"
SIGNALS_TABLE_ID = f"{config.SOURCE_PROJECT_ID}.{config.BIGQUERY_DATASET}.options_analysis_signals"
OPTIONS_CHAIN_TABLE_ID = f"{config.SOURCE_PROJECT_ID}.{config.BIGQUERY_DATASET}.options_chain"
PRICE_DATA_TABLE_ID = f"{config.SOURCE_PROJECT_ID}.{config.BIGQUERY_DATASET}.price_data"
OUTPUT_TABLE_ID = f"{config.DESTINATION_PROJECT_ID}.{config.BIGQUERY_DATASET}.performance_tracker"

# --- Main Logic ---

def _get_new_and_active_winners(bq_client: bigquery.Client) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Identifies today's new winners and fetches the list of previously identified, still-active winners.
    """
    today_iso = date.today().isoformat()
    
    # Query to find today's new winners that are not already being tracked as 'Active'
    new_winners_query = f"""
        SELECT
            w.ticker,
            s.contract_symbol,
            s.option_type,
            s.expiration_date,
            s.strike_price,
            p.adj_close AS stock_price_initial,
            (s.bid + s.ask) / 2 AS initial_price
        FROM `{WINNERS_TABLE_ID}` w
        JOIN `{SIGNALS_TABLE_ID}` s ON w.ticker = s.ticker
        LEFT JOIN `{PRICE_DATA_TABLE_ID}` p ON w.ticker = p.ticker AND p.date = s.run_date
        LEFT JOIN `{OUTPUT_TABLE_ID}` t ON s.contract_symbol = t.contract_symbol AND t.status = 'Active'
        WHERE w.run_date = @today
          AND s.run_date = @today
          AND s.setup_quality_signal LIKE '🟢 Strong'
          AND t.contract_symbol IS NULL -- This ensures we only get contracts not already tracked
    """
    
    # Query to get the list of contracts that are already being tracked and are 'Active'
    active_winners_query = f"""
        SELECT DISTINCT
            contract_symbol,
            ticker,
            recommendation_date,
            initial_price,
            stock_price_initial
        FROM `{OUTPUT_TABLE_ID}`
        WHERE status = 'Active'
    """

    job_config = bigquery.QueryJobConfig(query_parameters=[bigquery.ScalarQueryParameter("today", "DATE", today_iso)])
    
    new_winners_df = bq_client.query(new_winners_query, job_config=job_config).to_dataframe()
    active_winners_df = bq_client.query(active_winners_query).to_dataframe()
    
    return new_winners_df, active_winners_df

def _get_current_prices(bq_client: bigquery.Client, contracts_df: pd.DataFrame) -> pd.DataFrame:
    """
    Fetches the latest stock and option prices for a given list of contracts.
    """
    if contracts_df.empty:
        return pd.DataFrame()
        
    contract_symbols = contracts_df['contract_symbol'].tolist()
    
    query = f"""
        SELECT
            c.contract_symbol,
            p.adj_close AS stock_price_snapshot,
            (c.bid + c.ask) / 2 AS snapshot_price
        FROM `{OPTIONS_CHAIN_TABLE_ID}` c
        LEFT JOIN `{PRICE_DATA_TABLE_ID}` p ON c.ticker = p.ticker AND c.fetch_date = p.date
        WHERE c.contract_symbol IN UNNEST(@contract_symbols)
          AND c.fetch_date = (SELECT MAX(fetch_date) FROM `{OPTIONS_CHAIN_TABLE_ID}`)
    """
    
    job_config = bigquery.QueryJobConfig(
        query_parameters=[bigquery.ArrayQueryParameter("contract_symbols", "STRING", contract_symbols)]
    )
    
    prices_df = bq_client.query(query, job_config=job_config).to_dataframe()
    return prices_df

def run_pipeline():
    """
    Orchestrates the daily update of the performance tracker table.
    """
    logging.info("--- Starting Performance Tracker Update Pipeline ---")
    bq_client = bigquery.Client(project=config.SOURCE_PROJECT_ID)
    today = date.today()

    # 1. Identify new winners and find existing active trades
    new_winners_df, active_winners_df = _get_new_and_active_winners(bq_client)
    
    # 2. Prepare today's snapshot for the new winners
    if not new_winners_df.empty:
        new_winners_df['recommendation_date'] = today
        new_winners_df['snapshot_date'] = today
        new_winners_df['stock_price_snapshot'] = new_winners_df['stock_price_initial']
        new_winners_df['snapshot_price'] = new_winners_df['initial_price']
        new_winners_df['status'] = 'Active'
    
    # 3. Fetch current prices for the active old winners
    current_prices_df = _get_current_prices(bq_client, active_winners_df)
    
    # 4. Prepare today's snapshot for the active old winners
    updated_winners_df = pd.DataFrame()
    if not active_winners_df.empty and not current_prices_df.empty:
        merged_df = pd.merge(active_winners_df, current_prices_df, on='contract_symbol', how='left')
        merged_df['snapshot_date'] = today
        
        # Check if the contract has expired or if we lost pricing data
        merged_df['status'] = 'Active'
        merged_df.loc[pd.to_datetime(merged_df['expiration_date']) < pd.to_datetime(today), 'status'] = 'Expired'
        merged_df.loc[merged_df['snapshot_price'].isnull(), 'status'] = 'Delisted'
        
        updated_winners_df = merged_df

    # 5. Combine all of today's snapshots into one DataFrame
    final_df = pd.concat([new_winners_df, updated_winners_df], ignore_index=True)
    
    if final_df.empty:
        logging.info("No new or updated winners to track today.")
        logging.info("--- Performance Tracker Update Pipeline Finished ---")
        return
        
    # 6. Ensure schema matches the destination table
    final_columns = [
        "recommendation_date", "snapshot_date", "ticker", "contract_symbol", "option_type",
        "expiration_date", "strike_price", "stock_price_initial", "stock_price_snapshot",
        "initial_price", "snapshot_price", "status"
    ]
    for col in final_columns:
        if col not in final_df.columns:
            final_df[col] = None
    final_df = final_df[final_columns]
    
    # 7. Append today's data to the performance tracker table
    logging.info(f"Adding {len(final_df)} new performance snapshots to the tracker.")
    bq.load_df_to_bq(final_df, OUTPUT_TABLE_ID, config.DESTINATION_PROJECT_ID, write_disposition="WRITE_APPEND")

    logging.info("--- Performance Tracker Update Pipeline Finished ---")