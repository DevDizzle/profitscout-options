# enrichment/core/analysis/options_analysis_helper.py
from __future__ import annotations
import datetime as dt
import math
import random
import time
import uuid
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import pandas_ta as ta
from google.api_core.exceptions import BadRequest
from google.cloud import bigquery

# CONFIG
PROJECT = "profitscout-lx6bb"
DATASET = "profit_scout"
PRICE_TABLE_ID = f"{PROJECT}.{DATASET}.price_data"
TARGET_TABLE_ID = f"{PROJECT}.{DATASET}.options_analysis_input"
META_TABLE_ID = f"{PROJECT}.{DATASET}.stock_metadata"
RSI_LEN, SMA50, SMA200 = 14, 50, 200
ATM_DTE_MIN, ATM_DTE_MAX, ATM_MNY_PCT = 7, 90, 0.05

def ensure_table_exists(bq: bigquery.Client) -> None:
    """Create the options_analysis_input table if it doesn't exist."""
    ddl = f"""
    CREATE TABLE IF NOT EXISTS `{TARGET_TABLE_ID}` (
        ticker STRING, date DATE, open FLOAT64, high FLOAT64, low FLOAT64,
        adj_close FLOAT64, volume INT64, iv_avg FLOAT64, hv_30 FLOAT64,
        iv_industry_avg FLOAT64, iv_signal STRING, latest_rsi FLOAT64,
        latest_macd FLOAT64, latest_sma50 FLOAT64, latest_sma200 FLOAT64,
        close_30d_delta_pct FLOAT64, rsi_30d_delta FLOAT64, macd_30d_delta FLOAT64,
        close_90d_delta_pct FLOAT64, rsi_90d_delta FLOAT64, macd_90d_delta FLOAT64
    ) PARTITION BY date CLUSTER BY ticker
    """
    bq.query(ddl).result()

def _safe_float(x) -> Optional[float]:
    try:
        if x is None: return None
        xf = float(x)
        if math.isnan(xf) or math.isinf(xf): return None
        return xf
    except Exception: return None

def _normalize_row(row: Dict) -> Dict:
    """Coerce required fields/shape for upsert."""
    out = dict(row)
    if not out.get("ticker"): raise ValueError("Row missing 'ticker'")
    d = out.get("date")
    if isinstance(d, (dt.date, dt.datetime)): out["date"] = d.strftime("%Y-%m-%d")
    elif not isinstance(d, str): raise ValueError("Row missing or invalid 'date'")
    return out

def _fetch_ohlcv_for_keys(bq: bigquery.Client, keys: List[Dict[str, str]]) -> Dict[tuple, Dict]:
    if not keys: return {}
    tickers, min_date, max_date = list({k["ticker"] for k in keys}), min(k["date"] for k in keys), max(k["date"] for k in keys)
    q = f"""
        SELECT ticker, CAST(date AS STRING) AS date_str, open, high, low, adj_close, volume
        FROM `{PRICE_TABLE_ID}`
        WHERE ticker IN UNNEST(@tickers) AND date BETWEEN @min_date AND @max_date
    """
    params = [bigquery.ArrayQueryParameter("tickers", "STRING", tickers),
              bigquery.ScalarQueryParameter("min_date", "DATE", min_date),
              bigquery.ScalarQueryParameter("max_date", "DATE", max_date)]
    rows = bq.query(q, job_config=bigquery.QueryJobConfig(query_parameters=params)).result()
    return {(r["ticker"], r["date_str"]): r for r in rows}

def _spot_on_or_before(bq: bigquery.Client, ticker: str, as_of: dt.date) -> Optional[float]:
    """Get spot (adj_close) on or before as_of date."""
    q = f"SELECT adj_close FROM `{PRICE_TABLE_ID}` WHERE ticker = @ticker AND date <= @as_of ORDER BY date DESC LIMIT 1"
    params = [bigquery.ScalarQueryParameter("ticker", "STRING", ticker),
              bigquery.ScalarQueryParameter("as_of", "DATE", str(as_of))]
    rows = bq.query(q, job_config=bigquery.QueryJobConfig(query_parameters=params)).result()
    for r in rows: return _safe_float(r["adj_close"])
    return None

def _create_staging_table(bq: bigquery.Client) -> str:
    staging_id = f"{PROJECT}.{DATASET}._tmp_options_analysis_{uuid.uuid4().hex}"
    schema = [
        bigquery.SchemaField("ticker", "STRING"), bigquery.SchemaField("date", "DATE"),
        bigquery.SchemaField("open", "FLOAT"), bigquery.SchemaField("high", "FLOAT"),
        bigquery.SchemaField("low", "FLOAT"), bigquery.SchemaField("adj_close", "FLOAT"),
        bigquery.SchemaField("volume", "INT64"), bigquery.SchemaField("iv_avg", "FLOAT"),
        bigquery.SchemaField("hv_30", "FLOAT"), bigquery.SchemaField("iv_industry_avg", "FLOAT"),
        bigquery.SchemaField("iv_signal", "STRING"), bigquery.SchemaField("latest_rsi", "FLOAT"),
        bigquery.SchemaField("latest_macd", "FLOAT"), bigquery.SchemaField("latest_sma50", "FLOAT"),
        bigquery.SchemaField("latest_sma200", "FLOAT"),
        bigquery.SchemaField("close_30d_delta_pct", "FLOAT"),
        bigquery.SchemaField("rsi_30d_delta", "FLOAT"), bigquery.SchemaField("macd_30d_delta", "FLOAT"),
        bigquery.SchemaField("close_90d_delta_pct", "FLOAT"),
        bigquery.SchemaField("rsi_90d_delta", "FLOAT"), bigquery.SchemaField("macd_90d_delta", "FLOAT"),
    ]
    bq.create_table(bigquery.Table(staging_id, schema=schema))
    return staging_id

def _merge_from_staging(bq: bigquery.Client, staging_id: str, present_cols: List[str]) -> None:
    non_keys = [c for c in present_cols if c not in ("ticker", "date")]
    if not non_keys:
        bq.delete_table(staging_id, not_found_ok=True)
        return
    set_clause = ", ".join([f"T.{c} = COALESCE(S.{c}, T.{c})" for c in non_keys])
    insert_cols, insert_vals = ", ".join(present_cols), ", ".join([f"S.{c}" for c in present_cols])
    sql = f"""
    MERGE `{TARGET_TABLE_ID}` T USING `{staging_id}` S ON T.ticker = S.ticker AND T.date = S.date
    WHEN MATCHED THEN UPDATE SET {set_clause}
    WHEN NOT MATCHED THEN INSERT ({insert_cols}) VALUES ({insert_vals})
    """
    for attempt in range(6):
        try:
            bq.query(sql).result()
            break
        except BadRequest as e:
            if "Could not serialize access" in getattr(e, "message", str(e)):
                time.sleep((2 ** attempt) + random.random())
                continue
            raise
    try: bq.delete_table(staging_id, not_found_ok=True)
    except Exception: pass

def upsert_analysis_rows(bq: bigquery.Client, rows: List[Dict], enrich_ohlcv: bool = True) -> None:
    """Upsert rows into options_analysis_input, enriching OHLCV from price_data if desired."""
    if not rows: return
    ensure_table_exists(bq)
    norm = [_normalize_row(r) for r in rows]
    if enrich_ohlcv:
        keys, ohlcv = [{"ticker": r["ticker"], "date": r["date"]} for r in norm], _fetch_ohlcv_for_keys(bq, keys)
        for r in norm:
            if (k := (r["ticker"], r["date"])) in ohlcv:
                r.update(ohlcv[k])
    present_cols = sorted({k for r in norm for k, v in r.items() if v is not None} | {"ticker", "date"})
    staging_id = _create_staging_table(bq)
    bq.load_table_from_json(norm, staging_id).result()
    _merge_from_staging(bq, staging_id, present_cols)

def compute_iv_avg_atm(full_chain_df: pd.DataFrame, underlying_price: Optional[float], as_of: dt.date) -> Optional[float]:
    if full_chain_df is None or full_chain_df.empty or not underlying_price: return None
    df = full_chain_df.copy()
    df["expiration_date"] = pd.to_datetime(df["expiration_date"], errors="coerce").dt.date
    df["implied_volatility"] = pd.to_numeric(df["implied_volatility"], errors="coerce")
    df["strike"] = pd.to_numeric(df["strike"], errors="coerce")
    df.dropna(subset=["expiration_date", "implied_volatility", "strike"], inplace=True)
    if df.empty: return None
    dte = (pd.to_datetime(df["expiration_date"]) - pd.to_datetime(as_of)).dt.days
    mny = np.abs(df["strike"] - underlying_price) / underlying_price
    atm_df = df.loc[(dte >= ATM_DTE_MIN) & (dte <= ATM_DTE_MAX) & (mny <= ATM_MNY_PCT)]
    if atm_df.empty: return None
    return _safe_float(atm_df["implied_volatility"].mean())

def _fetch_history_for_technicals(bq: bigquery.Client, ticker: str, as_of: dt.date) -> pd.DataFrame:
    q = f"""
        SELECT date, open, high, low, adj_close AS close, volume
        FROM `{PRICE_TABLE_ID}`
        WHERE ticker = @ticker AND date <= @as_of AND date >= DATE_SUB(@as_of, INTERVAL 400 DAY)
        ORDER BY date ASC
    """
    params = [bigquery.ScalarQueryParameter("ticker", "STRING", ticker),
              bigquery.ScalarQueryParameter("as_of", "DATE", str(as_of))]
    df = bq.query(q, job_config=bigquery.QueryJobConfig(query_parameters=params)).to_dataframe()
    if df.empty: return df
    df["date"] = pd.to_datetime(df["date"])
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df.ffill().bfill().dropna(inplace=True)
    return df

def compute_technicals_and_deltas(price_hist: pd.DataFrame) -> Dict[str, Optional[float]]:
    out_keys = ["latest_rsi", "latest_macd", "latest_sma50", "latest_sma200",
                "close_30d_delta_pct", "rsi_30d_delta", "macd_30d_delta",
                "close_90d_delta_pct", "rsi_90d_delta", "macd_90d_delta"]
    if price_hist is None or price_hist.empty: return {k: None for k in out_keys}
    df = price_hist.copy()
    df[f"RSI_{RSI_LEN}"] = ta.rsi(close=df["close"], length=RSI_LEN)
    macd = ta.macd(close=df["close"], fast=12, slow=26, signal=9)
    df["MACD_12_26_9"] = macd.iloc[:, 0] if isinstance(macd, pd.DataFrame) and not macd.empty else np.nan
    df["SMA_50"] = ta.sma(close=df["close"], length=SMA50)
    df["SMA_200"] = ta.sma(close=df["close"], length=SMA200)
    valid = df.dropna(subset=["close", "RSI_14", "MACD_12_26_9", "SMA_50", "SMA_200"])
    if valid.empty: return {k: None for k in out_keys}
    latest = valid.iloc[-1]
    out = {"latest_rsi": _safe_float(latest["RSI_14"]), "latest_macd": _safe_float(latest["MACD_12_26_9"]),
           "latest_sma50": _safe_float(latest["SMA_50"]), "latest_sma200": _safe_float(latest["SMA_200"])}
    try:
        if len(valid) >= 31:
            ago_30 = valid.iloc[-31]
            out["close_30d_delta_pct"] = _safe_float((latest["close"] - ago_30["close"]) / ago_30["close"] * 100.0)
            out["rsi_30d_delta"] = _safe_float(latest["RSI_14"] - ago_30["RSI_14"])
            out["macd_30d_delta"] = _safe_float(latest["MACD_12_26_9"] - ago_30["MACD_12_26_9"])
        if len(valid) >= 91:
            ago_90 = valid.iloc[-91]
            out["close_90d_delta_pct"] = _safe_float((latest["close"] - ago_90["close"]) / ago_90["close"] * 100.0)
            out["rsi_90d_delta"] = _safe_float(latest["RSI_14"] - ago_90["RSI_14"])
            out["macd_90d_delta"] = _safe_float(latest["MACD_12_26_9"] - ago_90["MACD_12_26_9"])
    except Exception: pass
    return out

def compute_hv30(bq: bigquery.Client, ticker: str, as_of: dt.date) -> Optional[float]:
    q = f"""
        WITH returns AS (
            SELECT LOG(adj_close / LAG(adj_close) OVER (ORDER BY date)) AS lr
            FROM `{PRICE_TABLE_ID}`
            WHERE ticker = @ticker AND date > DATE_SUB(@as_of, INTERVAL 31 DAY) AND date <= @as_of
        )
        SELECT STDDEV_SAMP(lr) * SQRT(252) AS hv_30 FROM returns
    """
    params = [bigquery.ScalarQueryParameter("ticker", "STRING", ticker),
              bigquery.ScalarQueryParameter("as_of", "DATE", str(as_of))]
    df = bq.query(q, job_config=bigquery.QueryJobConfig(query_parameters=params)).to_dataframe()
    if df.empty or pd.isna(df["hv_30"].iloc[0]): return None
    return _safe_float(df["hv_30"].iloc[0])

def compute_iv_industry_avg(bq: bigquery.Client, ticker: str, as_of: dt.date) -> Optional[float]:
    peer_q = f"""
        WITH me AS (SELECT industry, sector FROM `{META_TABLE_ID}` WHERE ticker = @ticker LIMIT 1),
        peers AS (
            SELECT m.ticker FROM `{META_TABLE_ID}` m, me
            WHERE (me.industry IS NOT NULL AND m.industry = me.industry) OR
                  (me.industry IS NULL AND me.sector IS NOT NULL AND m.sector = me.sector)
        ),
        latest AS (
            SELECT a.ticker, a.iv_avg FROM `{TARGET_TABLE_ID}` a JOIN peers p USING (ticker)
            WHERE a.date = @as_of AND a.iv_avg IS NOT NULL
        )
        SELECT AVG(iv_avg) AS iv_industry_avg FROM latest WHERE ticker != @ticker
    """
    params = [bigquery.ScalarQueryParameter("ticker", "STRING", ticker),
              bigquery.ScalarQueryParameter("as_of", "DATE", str(as_of))]
    df = bq.query(peer_q, job_config=bigquery.QueryJobConfig(query_parameters=params)).to_dataframe()
    if not df.empty and pd.notnull(df["iv_industry_avg"].iloc[0]):
        return float(df["iv_industry_avg"].iloc[0])
    market_q = f"SELECT AVG(iv_avg) AS iv_industry_avg FROM `{TARGET_TABLE_ID}` WHERE date = @as_of AND iv_avg IS NOT NULL AND ticker != @ticker"
    df2 = bq.query(market_q, job_config=bigquery.QueryJobConfig(query_parameters=params)).to_dataframe()
    if not df2.empty and pd.notnull(df2["iv_industry_avg"].iloc[0]):
        return float(df2["iv_industry_avg"].iloc[0])
    return None

def build_and_upsert_for_ticker(bq: bigquery.Client, ticker: str, snapshot_date: dt.date, full_chain_df: pd.DataFrame) -> None:
    uprice = None
    if full_chain_df is not None and not full_chain_df.empty and "underlying_price" in full_chain_df.columns:
        u = pd.to_numeric(full_chain_df["underlying_price"], errors="coerce").dropna()
        if not u.empty: uprice = float(u.iloc[0])
    if uprice is None: uprice = _spot_on_or_before(bq, ticker, snapshot_date)
    iv_avg, hv_30 = compute_iv_avg_atm(full_chain_df, uprice, snapshot_date) if uprice else None, compute_hv30(bq, ticker, snapshot_date)
    hist = _fetch_history_for_technicals(bq, ticker, snapshot_date)
    tech = compute_technicals_and_deltas(hist)
    iv_signal = None
    if iv_avg is not None and hv_30 is not None:
        try: iv_signal = "high" if iv_avg > (hv_30 + 0.10) else "low"
        except Exception: pass
    iv_industry_avg = compute_iv_industry_avg(bq, ticker, snapshot_date)
    row = {"ticker": ticker, "date": snapshot_date, "iv_avg": iv_avg, "hv_30": hv_30,
           "iv_signal": iv_signal, "iv_industry_avg": iv_industry_avg, **tech}
    upsert_analysis_rows(bq, [row], enrich_ohlcv=True)

def backfill_iv_industry_avg_for_date(bq: bigquery.Client, run_date: Optional[dt.date] = None) -> None:
    """After all tickers for a day are upserted, recompute iv_industry_avg consistently."""
    run_date = run_date or dt.date.today()
    sql = f"""
    DECLARE run_date DATE DEFAULT @run_date;
    MERGE `{TARGET_TABLE_ID}` T USING (
        WITH latest AS (
            SELECT a.ticker, a.date, a.iv_avg, m.industry, m.sector
            FROM `{TARGET_TABLE_ID}` a LEFT JOIN `{META_TABLE_ID}` m ON a.ticker = m.ticker
            WHERE a.date = run_date
        ),
        peers AS (
            SELECT l1.ticker, AVG(l2.iv_avg) AS peer_avg
            FROM latest l1 JOIN latest l2 ON l1.ticker != l2.ticker
            AND ((l1.industry IS NOT NULL AND l2.industry = l1.industry) OR
                 (l1.industry IS NULL AND l1.sector IS NOT NULL AND l2.sector = l1.sector) OR
                 (l1.industry IS NULL AND l1.sector IS NULL))
            WHERE l2.iv_avg IS NOT NULL GROUP BY l1.ticker
        ),
        market AS (SELECT AVG(iv_avg) AS mkt_avg FROM latest WHERE iv_avg IS NOT NULL)
        SELECT l.ticker, l.date, COALESCE(p.peer_avg, m.mkt_avg) AS new_iv_industry_avg
        FROM latest l LEFT JOIN peers p ON p.ticker = l.ticker CROSS JOIN market m
    ) S ON T.ticker = S.ticker AND T.date = S.date
    WHEN MATCHED THEN UPDATE SET T.iv_industry_avg = S.new_iv_industry_avg
    """
    bq.query(sql, job_config=bigquery.QueryJobConfig(query_parameters=[
        bigquery.ScalarQueryParameter("run_date", "DATE", str(run_date))])).result()