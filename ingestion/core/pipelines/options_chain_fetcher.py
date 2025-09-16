# ingestion/core/pipelines/options_chain_fetcher.py
import logging
import time
import pandas as pd
import numpy as np
from datetime import date
from concurrent.futures import ThreadPoolExecutor, as_completed
from google.cloud import bigquery

from .. import config
from ..clients.polygon import PolygonClient

OPTIONS_TABLE = f"{config.PROJECT_ID}.{config.BIGQUERY_DATASET}.options_chain"
PRICE_TABLE_ID = f"{config.PROJECT_ID}.{config.BIGQUERY_DATASET}.price_data"

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


def _compute_peer_or_market_iv_avg(
    bq_client: bigquery.Client, ticker: str
) -> float | None:
    """
    Compute a sector/industry baseline IV:
      1) Find this ticker's industry (prefer) or sector from stock_metadata.
      2) Among names in that group, average each peer's *latest* iv_avg from price_data (exclude self).
      3) If no peers, fall back to market-wide average of each ticker's latest iv_avg (exclude self).
    """
    project = config.PROJECT_ID
    dataset = config.BIGQUERY_DATASET
    meta_tbl = f"`{project}.{dataset}.stock_metadata`"

    # Peer baseline: same industry if present, else sector
    peer_q = f"""
    WITH me AS (
      SELECT industry, sector
      FROM {meta_tbl}
      WHERE ticker = @ticker
      LIMIT 1
    ),
    latest AS (
      SELECT p.ticker, p.iv_avg, p.date,
             ROW_NUMBER() OVER (PARTITION BY p.ticker ORDER BY p.date DESC) AS rn
      FROM `{PRICE_TABLE_ID}` p
      JOIN {meta_tbl} m USING (ticker)
      WHERE p.iv_avg IS NOT NULL
        AND (
             (SELECT industry FROM me) IS NOT NULL AND m.industry = (SELECT industry FROM me)
          OR (SELECT industry FROM me) IS NULL AND (SELECT sector FROM me) IS NOT NULL AND m.sector = (SELECT sector FROM me)
        )
    )
    SELECT AVG(iv_avg) AS iv_industry_avg
    FROM latest
    WHERE rn = 1 AND ticker != @ticker
    """
    try:
        df = bq_client.query(
            peer_q,
            job_config=bigquery.QueryJobConfig(
                query_parameters=[bigquery.ScalarQueryParameter("ticker", "STRING", ticker)]
            ),
        ).to_dataframe()
        if not df.empty and pd.notnull(df["iv_industry_avg"].iloc[0]):
            return float(df["iv_industry_avg"].iloc[0])
    except Exception as e:
        logging.info("[%s] Peer IV baseline unavailable; will fall back. %s", ticker, e)

    # Market-wide fallback: average of latest iv_avg across all tickers (exclude self)
    fallback_q = f"""
    WITH latest AS (
      SELECT ticker, iv_avg,
             ROW_NUMBER() OVER (PARTITION BY ticker ORDER BY date DESC) AS rn
      FROM `{PRICE_TABLE_ID}`
      WHERE iv_avg IS NOT NULL
    )
    SELECT AVG(iv_avg) AS iv_industry_avg
    FROM latest
    WHERE rn = 1 AND ticker != @ticker
    """
    try:
        df2 = bq_client.query(
            fallback_q,
            job_config=bigquery.QueryJobConfig(
                query_parameters=[bigquery.ScalarQueryParameter("ticker", "STRING", ticker)]
            ),
        ).to_dataframe()
        if not df2.empty and pd.notnull(df2["iv_industry_avg"].iloc[0]):
            return float(df2["iv_industry_avg"].iloc[0])
    except Exception as e:
        logging.info("[%s] Market-wide IV avg fallback failed: %s", ticker, e)

    return None


def _compute_and_insert_iv_metrics(bq_client: bigquery.Client, df: pd.DataFrame, ticker: str, today: date):
    """
    Computes stock-level IV metrics from the options chain DF and inserts/updates them in price_data.
    - iv_avg: Average ATM IV for 7–90 DTE contracts (±5% moneyness).
    - hv_30: 30-day historical volatility (annualized) from price_data.
    - iv_percentile: Percentile rank over the last ~year (if history exists).
    - iv_industry_avg: Peers' avg IV (if available), with market-wide fallback.
    """
    if df.empty:
        logging.warning(f"[{ticker}] Empty DF for IV computation.")
        return

    underlying_price = df["underlying_price"].iloc[0]
    if pd.isna(underlying_price):
        logging.warning(f"[{ticker}] Missing underlying_price after backfills; skipping IV computation.")
        return

    df = df.copy()
    df["dte"] = (pd.to_datetime(df["expiration_date"]) - pd.to_datetime(today)).dt.days
    atm_df = df[(df["dte"].between(7, 90)) &
                (np.abs(df["strike"] - underlying_price) / underlying_price <= 0.05)]

    if atm_df.empty:
        logging.warning(f"[{ticker}] No ATM contracts for IV average.")
        return

    iv_avg = float(atm_df["implied_volatility"].mean())

    # HV_30 from daily log returns (annualized)
    hv_query = f"""
        WITH returns AS (
            SELECT date, adj_close,
                   LOG(adj_close / LAG(adj_close) OVER (PARTITION BY ticker ORDER BY date)) AS log_return
            FROM `{PRICE_TABLE_ID}`
            WHERE ticker = @ticker AND date >= DATE_SUB(@today, INTERVAL 30 DAY)
        )
        SELECT STDDEV_SAMP(log_return) * SQRT(252) AS hv_30
        FROM returns
    """
    hv_df = bq_client.query(
        hv_query,
        job_config=bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("ticker", "STRING", ticker),
                bigquery.ScalarQueryParameter("today", "DATE", str(today)),
            ]
        ),
    ).to_dataframe()
    hv_30 = float(hv_df["hv_30"].iloc[0]) if not hv_df.empty and pd.notnull(hv_df["hv_30"].iloc[0]) else None

    # IV percentile over trailing ~year (if you persist past iv_avg values in price_data)
    percentile_query = f"""
        WITH hist AS (
          SELECT iv_avg
          FROM `{PRICE_TABLE_ID}`
          WHERE ticker = @ticker AND date >= DATE_SUB(@today, INTERVAL 365 DAY) AND iv_avg IS NOT NULL
        ),
        ranked AS (
          SELECT iv_avg,
                 PERCENT_RANK() OVER (ORDER BY iv_avg) * 100 AS pct
          FROM hist
        )
        SELECT pct
        FROM ranked
        ORDER BY iv_avg DESC
        LIMIT 1
    """
    percentile_df = bq_client.query(
        percentile_query,
        job_config=bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("ticker", "STRING", ticker),
                bigquery.ScalarQueryParameter("today", "DATE", str(today)),
            ]
        ),
    ).to_dataframe()
    iv_percentile = float(percentile_df["pct"].iloc[0]) if not percentile_df.empty else None

    # Compute industry (or market-wide) avg IV safely
    iv_industry_avg = _compute_peer_or_market_iv_avg(bq_client, ticker)

    # Simple signal
    iv_signal = "high" if (iv_percentile and iv_percentile > 50) or (hv_30 and iv_avg > (hv_30 + 0.10)) else "low"

    merge_q = f"""
        MERGE `{PRICE_TABLE_ID}` T
        USING (
          SELECT
            @ticker AS ticker,
            DATE(@today) AS date,
            @iv_avg AS iv_avg,
            @hv_30 AS hv_30,
            @iv_percentile AS iv_percentile,
            @iv_industry_avg AS iv_industry_avg,
            @iv_signal AS iv_signal
        ) S
        ON T.ticker = S.ticker AND T.date = S.date
        WHEN MATCHED THEN UPDATE SET
          iv_avg = S.iv_avg,
          hv_30 = S.hv_30,
          iv_percentile = S.iv_percentile,
          iv_industry_avg = S.iv_industry_avg,
          iv_signal = S.iv_signal
        WHEN NOT MATCHED THEN INSERT (ticker, date, iv_avg, hv_30, iv_percentile, iv_industry_avg, iv_signal)
          VALUES (S.ticker, S.date, S.iv_avg, S.hv_30, S.iv_percentile, S.iv_industry_avg, S.iv_signal)
    """
    bq_client.query(
        merge_q,
        job_config=bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("ticker", "STRING", ticker),
                bigquery.ScalarQueryParameter("today", "DATE", str(today)),
                bigquery.ScalarQueryParameter("iv_avg", "FLOAT64", iv_avg),
                bigquery.ScalarQueryParameter("hv_30", "FLOAT64", hv_30),
                bigquery.ScalarQueryParameter("iv_percentile", "FLOAT64", iv_percentile),
                bigquery.ScalarQueryParameter("iv_industry_avg", "FLOAT64", iv_industry_avg),
                bigquery.ScalarQueryParameter("iv_signal", "STRING", iv_signal),
            ]
        ),
    ).result()
    logging.info(f"[{ticker}] Inserted/updated IV metrics in price_data for {today}.")


def _fetch_and_load_chain_for_ticker(
    client: PolygonClient, bq_client: bigquery.Client, ticker: str, signal: str
):
    """
    Fetch Polygon option chain (≤90d) for ticker, compute IV metrics, then append to BigQuery.
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

    # Compute and insert IV metrics before direction filtering (uses full chain)
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
      - Compute IV metrics (ATM 7–90 DTE)
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
