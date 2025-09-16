# ingestion/core/pipelines/options_chain_fetcher.py
import logging
import time
import pandas as pd
from datetime import date
from concurrent.futures import ThreadPoolExecutor, as_completed
from google.cloud import bigquery

from .. import config
from ..clients.polygon import PolygonClient

OPTIONS_TABLE = f"{config.PROJECT_ID}.{config.BIGQUERY_DATASET}.options_chain"
PRICE_TABLE_ID = f"{config.PROJECT_ID}.{config.BIGQUERY_DATASET}.price_data"  # kept for underlying fallback

BUY_THRESHOLD = 0.62
SELL_THRESHOLD = 0.44


def _truncate_options_chain(bq_client: bigquery.Client):
    """Remove ALL previous rows so only today's snapshot remains, with retries."""
    q = f"TRUNCATE TABLE `{OPTIONS_TABLE}`"

    # Retry TRUNCATE (rarely, BigQuery can be momentarily busy)
    for attempt in range(3):
        try:
            job = bq_client.query(q)
            job.result()  # wait for DDL completion
            break
        except Exception:
            if attempt == 2:
                raise
            time.sleep(2 ** attempt)  # 1s, 2s backoff

    # Poll availability to avoid transient 404 "Table is truncated" on immediate writes
    for _ in range(10):
        try:
            bq_client.query(f"SELECT 1 FROM `{OPTIONS_TABLE}` LIMIT 0").result()
            logging.info("Truncated %s and verified availability.", OPTIONS_TABLE)
            return
        except Exception:
            time.sleep(0.5)

    logging.warning("Truncated %s but could not quickly verify availability.", OPTIONS_TABLE)


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
        "strike", "last_price", "bid", "ask", "volume", "open_interest",
        "implied_volatility", "delta", "theta", "vega", "gamma", "underlying_price",
    ]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df["expiration_date"] = pd.to_datetime(df["expiration_date"], errors="coerce").dt.date
    df = df.dropna(subset=["expiration_date", "contract_symbol"])

    cols = [
        "ticker", "contract_symbol", "option_type", "expiration_date",
        "strike", "last_price", "bid", "ask", "volume", "open_interest",
        "implied_volatility", "delta", "theta", "vega", "gamma",
        "underlying_price", "fetch_date",
    ]
    return df.reindex(columns=cols)


def _fill_underlying_from_bq_if_needed(bq_client: bigquery.Client, df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """
    If any missing underlying_price values remain (e.g., Options-only plan), fill from latest adj_close.
    """
    if df.empty or not df["underlying_price"].isna().any():
        return df

    q = f"""
      SELECT adj_close
      FROM `{PRICE_TABLE_ID}`
      WHERE ticker = @ticker
      ORDER BY date DESC
      LIMIT 1
    """
    job = bq_client.query(
        q,
        job_config=bigquery.QueryJobConfig(
            query_parameters=[bigquery.ScalarQueryParameter("ticker", "STRING", ticker)]
        ),
    )
    rows = list(job)
    if not rows or rows[0]["adj_close"] is None:
        logging.warning("[%s] No fallback adj_close in %s", ticker, PRICE_TABLE_ID)
        return df

    fallback = float(rows[0]["adj_close"])
    df["underlying_price"] = df["underlying_price"].fillna(fallback)
    logging.info("[%s] Filled missing underlying_price from price_data: %.4f", ticker, fallback)
    return df


def _fetch_and_load_chain_for_ticker(
    client: PolygonClient, bq_client: bigquery.Client, ticker: str, signal: str
):
    """
    Fetch Polygon option chain (≤90d) for ticker, then append filtered contracts to BigQuery.
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

    # Backfill underlying from BQ if plan/snapshot didn’t include it
    df = _fill_underlying_from_bq_if_needed(bq_client, df, ticker)

    # Direction filter only (IV computations removed)
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
