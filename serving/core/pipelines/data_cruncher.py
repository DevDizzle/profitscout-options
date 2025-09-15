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
    """Queries stock_metadata for the list of tickers, industry, sector, and their latest quarter."""
    logging.info("Fetching work list from stock_metadata...")
    client = bigquery.Client(project=config.SOURCE_PROJECT_ID)
    query = f"""
        SELECT ticker, industry, sector, MAX(quarter_end_date) as latest_quarter
        FROM `{config.BUNDLER_STOCK_METADATA_TABLE_ID}`
        WHERE ticker IS NOT NULL AND quarter_end_date IS NOT NULL AND industry IS NOT NULL AND sector IS NOT NULL
        GROUP BY ticker, industry, sector
    """
    df = client.query(query).to_dataframe()
    logging.info(f"Work list created for {len(df)} tickers.")
    return df

def _calculate_industry_averages(work_list_df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    """Calculates industry-level average KPIs based on enriched price_data."""
    logging.info("Calculating industry-level average KPIs...")
    if work_list_df.empty:
        return {}

    client = bigquery.Client(project=config.SOURCE_PROJECT_ID)

    industry_query = f"""
        WITH latest_prices AS (
            SELECT ticker, MAX(date) AS max_date
            FROM `{config.PROJECT_ID}.{config.BIGQUERY_DATASET}.price_data`
            GROUP BY ticker
        ),
        enriched AS (
            SELECT
                p.ticker,
                p.close_30d_delta_pct,
                p.iv_avg,
                m.industry
            FROM latest_prices lp
            JOIN `{config.PROJECT_ID}.{config.BIGQUERY_DATASET}.price_data` p
                ON lp.ticker = p.ticker AND lp.max_date = p.date
            JOIN `{config.BUNDLER_STOCK_METADATA_TABLE_ID}` m ON p.ticker = m.ticker
            WHERE p.close_30d_delta_pct IS NOT NULL AND m.industry IS NOT NULL
        )
        SELECT
            industry,
            AVG(close_30d_delta_pct) AS avg_30d_change_pct,
            AVG(iv_avg) AS avg_iv
        FROM enriched
        GROUP BY industry
    """
    industry_df = client.query(industry_query).to_dataframe()
    industry_averages = industry_df.set_index('industry').to_dict(orient='index')
    return industry_averages

def _delete_old_prep_files(ticker: str):
    """Deletes all prep JSON files for a given ticker."""
    prefix = f"{OUTPUT_PREFIX}{ticker}_"
    blobs_to_delete = gcs.list_blobs(config.GCS_BUCKET_NAME, prefix)
    for blob_name in blobs_to_delete:
        try:
            gcs.delete_blob(config.GCS_BUCKET_NAME, blob_name)
        except Exception as e:
            logging.error(f"[{ticker}] Failed to delete old prep file {blob_name}: {e}")

def _fetch_and_calculate_kpis(ticker: str, latest_quarter: date, industry: str, industry_averages: Dict) -> Optional[str]:
    """
    Fetches enriched data from price_data for a single ticker, calculates KPIs, and returns the generated JSON file path.
    """
    client = bigquery.Client(project=config.SOURCE_PROJECT_ID)
    run_date = date.today()
    run_date_str = run_date.strftime('%Y-%m-%d')

    final_json = {
        "ticker": ticker, "runDate": run_date_str, "kpis": {}
    }

    try:
        # Fetch enriched data from price_data (latest row)
        enriched_query = f"""
            WITH latest AS (
                SELECT MAX(date) AS max_date
                FROM `{config.PROJECT_ID}.{config.BIGQUERY_DATASET}.price_data`
                WHERE ticker = @ticker
            )
            SELECT
                p.date,
                p.adj_close,
                p.iv_avg,
                p.iv_percentile,
                p.hv_30,
                p.latest_rsi,
                p.rsi_30d_delta,
                p.latest_macd,
                p.macd_30d_delta,
                p.close_30d_delta_pct
            FROM latest l
            JOIN `{config.PROJECT_ID}.{config.BIGQUERY_DATASET}.price_data` p
                ON p.date = l.max_date AND p.ticker = @ticker
        """
        job_config = bigquery.QueryJobConfig(query_parameters=[bigquery.ScalarQueryParameter("ticker", "STRING", ticker)])
        enriched_df = client.query(enriched_query, job_config=job_config).to_dataframe()

        if enriched_df.empty:
            logging.warning(f"[{ticker}] No enriched price data found.")
            return None

        latest_row = enriched_df.iloc[0]
        industry_avg = industry_averages.get(industry, {})

        # KPI 1: Current Price
        # Assuming daily change from previous day (query prev_close if needed; for simplicity, assume we add it or compute)
        prev_close_query = f"""
            SELECT adj_close AS prev_close
            FROM `{config.PROJECT_ID}.{config.BIGQUERY_DATASET}.price_data`
            WHERE ticker = @ticker AND date = DATE_SUB(@run_date, INTERVAL 1 DAY)
        """
        prev_df = client.query(prev_close_query, job_config=bigquery.QueryJobConfig(
            query_parameters=[bigquery.ScalarQueryParameter("ticker", "STRING", ticker), bigquery.ScalarQueryParameter("run_date", "DATE", run_date)]
        )).to_dataframe()
        prev_close = prev_df['prev_close'].iloc[0] if not prev_df.empty else latest_row['adj_close']
        daily_change_pct = ((latest_row['adj_close'] - prev_close) / prev_close * 100) if prev_close != 0 else 0.0
        price_signal = "positive" if daily_change_pct > 0 else "negative"
        final_json["kpis"]["price"] = {"value": round(latest_row['adj_close'], 2), "dailyChangePct": round(daily_change_pct, 2), "signal": price_signal}

        # KPI 2: Implied Volatility (IV)
        iv_vs = latest_row['iv_percentile'] if pd.notna(latest_row['iv_percentile']) else latest_row['hv_30']
        iv_signal = "high" if (latest_row['iv_avg'] > (iv_vs or 0) + 10) else "low"  # Simple derivation; adjust as needed
        final_json["kpis"]["impliedVolatility"] = {"value": round(latest_row['iv_avg'], 2), "vsContext": round(iv_vs or 0, 2), "signal": iv_signal}

        # KPI 3: RSI
        rsi_signal = "oversold" if latest_row['latest_rsi'] < 30 else "overbought" if latest_row['latest_rsi'] > 70 else "neutral"
        final_json["kpis"]["rsi"] = {"value": round(latest_row['latest_rsi'], 2), "thirtyDayDelta": round(latest_row['rsi_30d_delta'], 2), "signal": rsi_signal}

        # KPI 4: MACD
        macd_signal = "bullish" if latest_row['latest_macd'] > 0 and latest_row['macd_30d_delta'] > 0 else "bearish" if latest_row['latest_macd'] < 0 else "neutral"
        final_json["kpis"]["macd"] = {"value": round(latest_row['latest_macd'], 2), "thirtyDayDelta": round(latest_row['macd_30d_delta'], 2), "signal": macd_signal}

        # KPI 5: 30-Day Price Change
        industry_avg_30d = industry_avg.get("avg_30d_change_pct")
        change_signal = "outperform" if pd.notna(industry_avg_30d) and latest_row['close_30d_delta_pct'] > industry_avg_30d else "underperform"
        final_json["kpis"]["thirtyDayPriceChange"] = {"value": round(latest_row['close_30d_delta_pct'], 2), "vsIndustry": round(industry_avg_30d, 2) if pd.notna(industry_avg_30d) else "unavailable", "signal": change_signal}

        _delete_old_prep_files(ticker)
        output_blob_name = f"{OUTPUT_PREFIX}{ticker}_{run_date_str}.json"
        # The prep file itself is still written to the source bucket for the dashboard_generator to find
        gcs.write_text(config.GCS_BUCKET_NAME, output_blob_name, json.dumps(final_json, indent=2), "application/json")
        logging.info(f"[{ticker}] Successfully generated and uploaded prep JSON with KPIs.")
        return output_blob_name

    except Exception as e:
        logging.error(f"[{ticker}] Failed during KPI calculation: {e}", exc_info=True)
        return None

def run_pipeline():
    """Orchestrates the data crunching pipeline with multi-threading."""
    logging.info("--- Starting Data Cruncher (Prep Stage) Pipeline ---")

    work_list_df = _get_work_list()
    if work_list_df.empty:
        logging.warning("No tickers in work list. Exiting.")
        return

    industry_averages = _calculate_industry_averages(work_list_df)

    processed_count = 0
    with ThreadPoolExecutor(max_workers=MAX_WORKERS, thread_name_prefix='CruncherWorker') as executor:
        future_to_ticker = {
            executor.submit(_fetch_and_calculate_kpis, row['ticker'], row['latest_quarter'], row['industry'], industry_averages): row['ticker']
            for _, row in work_list_df.iterrows()
        }
        for future in as_completed(future_to_ticker):
            try:
                if future.result():
                    processed_count += 1
            except Exception as exc:
                logging.error(f"Worker generated an unhandled exception: {exc}", exc_info=True)

    logging.info(f"--- Data Cruncher Pipeline Finished. Processed {processed_count} of {len(work_list_df)} tickers. ---")

if __name__ == "__main__":
    run_pipeline()