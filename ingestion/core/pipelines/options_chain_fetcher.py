# ingestion/core/pipelines/options_chain_fetcher.py
import logging
import pandas as pd
import numpy as np  # For any potential calculations, though we'll use SQL where possible
from datetime import date
from concurrent.futures import ThreadPoolExecutor, as_completed
from google.cloud import bigquery
from .. import config
from ..clients.polygon import PolygonClient

OPTIONS_TABLE = f"{config.PROJECT_ID}.{config.BIGQUERY_DATASET}.options_chain"
PRICE_TABLE_ID = f"{config.PROJECT_ID}.{config.BIGQUERY_DATASET}.price_data"  # Assuming based on your schema example

BUY_THRESHOLD = 0.62
SELL_THRESHOLD = 0.44


def _truncate_options_chain(bq_client: bigquery.Client):
    """Remove ALL previous rows so only today's snapshot remains."""
    q = f"TRUNCATE TABLE `{config.PROJECT_ID}.{config.BIGQUERY_DATASET}.options_chain`"
    job = bq_client.query(q)
    job.result()
    logging.info("Truncated %s", OPTIONS_TABLE)


def _get_buy_sell_universe(client: bigquery.Client) -> pd.DataFrame:
    """
    Select ALL tickers for the most recent run_date where:
      - BUY  if weighted_score >  BUY_THRESHOLD
      - SELL if weighted_score <  SELL_THRESHOLD
    Emits: ticker, signal ('BUY' or 'SELL'), weighted_score, run_date
    """
    q = f"""
    WITH latest_run AS (
      SELECT MAX(run_date) AS run_date
      FROM `{config.PROJECT_ID}.{config.BIGQUERY_DATASET}.analysis_scores`
      WHERE weighted_score IS NOT NULL
    ),
    latest AS (
      SELECT s.ticker, s.weighted_score, r.run_date
      FROM `{config.PROJECT_ID}.{config.BIGQUERY_DATASET}.analysis_scores` AS s
      JOIN latest_run AS r ON s.run_date = r.run_date
      WHERE s.weighted_score IS NOT NULL
    ),
    buys AS (
      SELECT ticker, 'BUY' AS signal, weighted_score, run_date
      FROM latest
      WHERE weighted_score > {BUY_THRESHOLD}
    ),
    sells AS (
      SELECT ticker, 'SELL' AS signal, weighted_score, run_date
      FROM latest
      WHERE weighted_score < {SELL_THRESHOLD}
    )
    SELECT * FROM buys
    UNION ALL
    SELECT * FROM sells
    """
    df = client.query(q).to_dataframe()
    if not df.empty:
        used = df["run_date"].iloc[0]
        n_buy = (df["signal"] == "BUY").sum()
        n_sell = (df["signal"] == "SELL").sum()
        logging.info(
            "Using latest run_date=%s; selected %d BUY and %d SELL tickers",
            used, n_buy, n_sell,
        )
    else:
        logging.warning("No BUY/SELL tickers found for latest run_date.")
    return df


def _coerce_and_align(df: pd.DataFrame, ticker: str, today: date) -> pd.DataFrame:
    """
    Ensures the DataFrame matches the options_chain schema and types.
    """
    rename = {
        "contract_symbol": "contract_symbol",
        "option_type": "option_type",
        "expiration_date": "expiration_date",
        "strike": "strike",
        "last_price": "last_price",
        "bid": "bid",
        "ask": "ask",
        "volume": "volume",
        "open_interest": "open_interest",
        "implied_volatility": "implied_volatility",
        "delta": "delta",
        "theta": "theta",
        "vega": "vega",
        "gamma": "gamma",
        "underlying_price": "underlying_price",
    }
    df = df.rename(columns=rename)

    # Normalize option_type to lowercase (call/put)
    if "option_type" in df.columns:
        df["option_type"] = df["option_type"].astype(str).str.lower()

    df["ticker"] = ticker
    df["fetch_date"] = today

    num_cols = [
        "strike",
        "last_price",
        "bid",
        "ask",
        "volume",
        "open_interest",
        "implied_volatility",
        "delta",
        "theta",
        "vega",
        "gamma",
        "underlying_price",
    ]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df["expiration_date"] = pd.to_datetime(df["expiration_date"], errors="coerce").dt.date
    df = df.dropna(subset=["expiration_date", "contract_symbol"])

    cols = [
        "ticker",
        "contract_symbol",
        "option_type",
        "expiration_date",
        "strike",
        "last_price",
        "bid",
        "ask",
        "volume",
        "open_interest",
        "implied_volatility",
        "delta",
        "theta",
        "vega",
        "gamma",
        "underlying_price",
        "fetch_date",
    ]
    return df.reindex(columns=cols)

def _compute_and_insert_iv_metrics(bq_client: bigquery.Client, df: pd.DataFrame, ticker: str, today: date):
    """
    Computes stock-level IV metrics from the options chain DF and inserts/updates them in price_data.
    - iv_avg: Average ATM IV for 7-90 DTE contracts.
    - iv_percentile: Percentile rank over past year (assuming historical iv_avg in price_data).
    - hv_30: 30-day historical volatility from price_data.
    - iv_industry_avg: Average IV of industry peers (assume a metadata table with peers; query and avg).
    - iv_signal: Derived signal (e.g., 'high' if iv_percentile > 50 or iv_avg > hv_30 + 0.10).
    Uses MERGE to update/insert the row for today.
    """
    if df.empty:
        logging.warning(f"[{ticker}] Empty DF for IV computation.")
        return

    underlying_price = df['underlying_price'].iloc[0]
    df['dte'] = (pd.to_datetime(df['expiration_date']) - pd.to_datetime(today)).dt.days
    atm_df = df[(df['dte'].between(7, 90)) & 
                (abs(df['strike'] - underlying_price) / underlying_price <= 0.05)]

    if atm_df.empty:
        logging.warning(f"[{ticker}] No ATM contracts for IV avg.")
        return

    iv_avg = atm_df['implied_volatility'].mean()

    # Compute HV_30 from price_data (annualized std dev of log returns)
    hv_query = f"""
        WITH returns AS (
            SELECT date, adj_close,
                   LOG(adj_close / LAG(adj_close) OVER (PARTITION BY ticker ORDER BY date)) AS log_return
            FROM `{PRICE_TABLE_ID}`
            WHERE ticker = '{ticker}' AND date >= DATE_SUB('{today}', INTERVAL 30 DAY)
        )
        SELECT STDDEV_SAMP(log_return) * SQRT(252) AS hv_30
        FROM returns
    """
    hv_df = bq_client.query(hv_query).to_dataframe()
    hv_30 = hv_df['hv_30'].iloc[0] if not hv_df.empty and pd.notnull(hv_df['hv_30'].iloc[0]) else None

    # Compute IV percentile from historical iv_avg in price_data (past 252 trading days ~1 year)
    percentile_query = f"""
        SELECT PERCENT_RANK() OVER (ORDER BY iv_avg) * 100 AS iv_percentile
        FROM `{PRICE_TABLE_ID}`
        WHERE ticker = '{ticker}' AND date >= DATE_SUB('{today}', INTERVAL 365 DAY) AND iv_avg IS NOT NULL
        ORDER BY date DESC
        LIMIT 1
    """
    percentile_df = bq_client.query(percentile_query).to_dataframe()
    iv_percentile = percentile_df['iv_percentile'].iloc[0] if not percentile_df.empty else None

    # Compute industry avg IV (assume a stock_metadata table with 'related_tickers' array or similar; avg their iv_avg)
    # For simplicity, query peers' latest iv_avg; adjust if no table
    industry_query = f"""
        WITH peers AS (
            SELECT STRING_AGG(related_ticker, ',') AS peer_list
            FROM `{config.PROJECT_ID}.{config.BIGQUERY_DATASET}.stock_metadata`  -- Assume table with peers
            WHERE ticker = '{ticker}'
        ),
        peer_iv AS (
            SELECT AVG(p.iv_avg) AS iv_industry_avg
            FROM UNNEST(SPLIT((SELECT peer_list FROM peers), ',')) AS peer_ticker
            JOIN `{PRICE_TABLE_ID}` p ON p.ticker = peer_ticker AND p.date = (SELECT MAX(date) FROM `{PRICE_TABLE_ID}` WHERE ticker = peer_ticker)
        )
        SELECT iv_industry_avg FROM peer_iv
    """
    industry_df = bq_client.query(industry_query).to_dataframe()
    iv_industry_avg = industry_df['iv_industry_avg'].iloc[0] if not industry_df.empty else None

    # Derive signal
    iv_signal = 'high' if (iv_percentile and iv_percentile > 50) or (hv_30 and iv_avg > hv_30 + 0.10) else 'low'

    # MERGE into price_data
    merge_q = f"""
        MERGE `{PRICE_TABLE_ID}` T
        USING (SELECT '{ticker}' AS ticker, DATE('{today}') AS date, {iv_avg or 'NULL'} AS iv_avg, 
               {hv_30 or 'NULL'} AS hv_30, {iv_percentile or 'NULL'} AS iv_percentile, 
               {iv_industry_avg or 'NULL'} AS iv_industry_avg, '{iv_signal}' AS iv_signal)
        S ON T.ticker = S.ticker AND T.date = S.date
        WHEN MATCHED THEN
            UPDATE SET iv_avg = S.iv_avg, hv_30 = S.hv_30, iv_percentile = S.iv_percentile, 
                       iv_industry_avg = S.iv_industry_avg, iv_signal = S.iv_signal
        WHEN NOT MATCHED THEN
            INSERT (ticker, date, iv_avg, hv_30, iv_percentile, iv_industry_avg, iv_signal) 
            VALUES (S.ticker, S.date, S.iv_avg, S.hv_30, S.iv_percentile, S.iv_industry_avg, S.iv_signal)
    """
    bq_client.query(merge_q).result()
    logging.info(f"[{ticker}] Inserted/updated IV metrics in price_data for {today}.")


def _fetch_and_load_chain_for_ticker(
    client: PolygonClient, bq_client: bigquery.Client, ticker: str, signal: str
):
    """
    Fetch Polygon option chain (≤90d) for ticker, filter by direction, then append to BigQuery.
      - BUY  => keep CALLs
      - SELL => keep PUTs
    """
    today = date.today()
    logging.info("[%s] Fetching Polygon chain (≤90d).", ticker)
    raw = client.fetch_options_chain(ticker, max_days=90)
    if not raw:
        logging.warning("[%s] No contracts returned.", ticker)
        return

    df = _coerce_and_align(pd.DataFrame(raw), ticker, today)

    # Compute and insert IV metrics before filtering (uses full chain)
    _compute_and_insert_iv_metrics(bq_client, df, ticker, today)

    desired = "call" if signal == "BUY" else "put"
    before = len(df)
    df = df[df["option_type"] == desired]
    logging.info("[%s] Direction=%s; kept %d/%d contracts.", ticker, desired.upper(), len(df), before)

    if df.empty:
        logging.info("[%s] Nothing to load after %s filter.", ticker, desired.upper())
        return

    logging.info("[%s] Loading %d contracts into %s", ticker, len(df), OPTIONS_TABLE)
    job = bq_client.load_table_from_dataframe(
        df, OPTIONS_TABLE, job_config=bigquery.LoadJobConfig(write_disposition="WRITE_APPEND")
    )
    job.result()


def run_pipeline(polygon_client: PolygonClient | None = None, bq_client: bigquery.Client | None = None):
    """
    Main entry:
      - TRUNCATE table (keep only today's snapshot)
      - Build BUY+SELL universe from latest run_date
      - Fetch Polygon chains (≤90d)
      - Filter CALLs for BUY and PUTs for SELL
      - Append to BigQuery
    """
    logging.info("--- Starting Options Chain Fetcher (Polygon) ---")
    bq_client = bq_client or bigquery.Client(project=config.PROJECT_ID)
    polygon_client = polygon_client or PolygonClient(api_key=config.POLYGON_API_KEY)

    _truncate_options_chain(bq_client)

    work = _get_buy_sell_universe(bq_client)
    if work.empty:
        logging.warning("No tickers identified. Exiting.")
        return

    tickers = ", ".join(sorted(set(work["ticker"].tolist())))
    logging.info("Tickers selected (%d): %s", len(work), tickers)

    with ThreadPoolExecutor(max_workers=16) as ex:
        futures = {
            ex.submit(_fetch_and_load_chain_for_ticker, polygon_client, bq_client, r.ticker, r.signal): r.ticker
            for _, r in work.iterrows()
        }
        for fut in as_completed(futures):
            try:
                fut.result()
            except Exception as e:
                logging.error("[%s] Worker failed: %s", futures[fut], e, exc_info=True)

    logging.info("--- Options Chain Fetcher Finished ---")