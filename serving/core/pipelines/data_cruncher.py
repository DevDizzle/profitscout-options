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
    """Calculates industry-level average KPIs."""
    logging.info("Calculating industry-level average KPIs...")
    if work_list_df.empty:
        return {}

    client = bigquery.Client(project=config.SOURCE_PROJECT_ID)

    price_query = f"""
        WITH PriceChanges AS (
            SELECT
                ticker,
                (adj_close - prev_close_30d) / NULLIF(prev_close_30d, 0) * 100 AS change_pct_30d
            FROM (
                SELECT
                    ticker,
                    date,
                    adj_close,
                    LAG(adj_close, 30) OVER (PARTITION BY ticker ORDER BY date) AS prev_close_30d,
                    ROW_NUMBER() OVER (PARTITION BY ticker ORDER BY date DESC) AS rn
                FROM `{config.PRICE_DATA_TABLE_ID}`
            )
            WHERE rn = 1
        )
        SELECT
            m.industry,
            AVG(pc.change_pct_30d) as avg_change_pct_30d
        FROM PriceChanges pc
        JOIN `{config.BUNDLER_STOCK_METADATA_TABLE_ID}` m ON pc.ticker = m.ticker
        WHERE m.industry IS NOT NULL
        GROUP BY m.industry
    """
    price_change_df = client.query(price_query).to_dataframe()
    industry_price_change_avg = price_change_df.set_index('industry')['avg_change_pct_30d'].to_dict()

    all_eps_growth = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_row = {
            executor.submit(gcs.read_blob, config.GCS_BUCKET_NAME, f"financial-statements/{row['ticker']}_{row['latest_quarter'].strftime('%Y-%m-%d')}.json"): row
            for _, row in work_list_df.iterrows()
        }
        for future in as_completed(future_to_row):
            row = future_to_row[future]
            industry = row['industry']
            json_content = future.result()
            if not json_content: continue

            try:
                data = json.loads(json_content)
                reports = data.get("quarterly_reports", [])
                if len(reports) >= 2:
                    latest_eps = reports[0].get("income_statement", {}).get("eps")
                    prev_eps = reports[1].get("income_statement", {}).get("eps")
                    if latest_eps is not None and prev_eps is not None and prev_eps != 0:
                        eps_qoq = (latest_eps - prev_eps) / abs(prev_eps) * 100
                        all_eps_growth.append({"industry": industry, "eps_qoq": eps_qoq})
            except (json.JSONDecodeError, KeyError):
                continue

    if all_eps_growth:
        eps_growth_df = pd.DataFrame(all_eps_growth)
        industry_eps_growth_avg = eps_growth_df.groupby('industry')['eps_qoq'].mean().to_dict()
    else:
        industry_eps_growth_avg = {}

    industry_averages = {}
    all_industries = set(industry_price_change_avg.keys()) | set(industry_eps_growth_avg.keys())
    for industry in all_industries:
        industry_averages[industry] = {
            "avg_change_pct_30d": industry_price_change_avg.get(industry),
            "avg_eps_qoq": industry_eps_growth_avg.get(industry)
        }
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
    Fetches all data for a single ticker, calculates KPIs, and returns the generated JSON file path.
    """
    client = bigquery.Client(project=config.SOURCE_PROJECT_ID)
    run_date = date.today()
    run_date_str = run_date.strftime('%Y-%m-%d')

    final_json = {
        "ticker": ticker, "runDate": run_date_str, "kpis": {},
        "chartUris": {}, "aiAnalystRecommendationUri": None
    }

    try:
        price_query = f"SELECT date, adj_close FROM `{config.PRICE_DATA_TABLE_ID}` WHERE ticker = @ticker AND date <= @run_date ORDER BY date DESC LIMIT 31"
        job_config = bigquery.QueryJobConfig(query_parameters=[bigquery.ScalarQueryParameter("ticker", "STRING", ticker), bigquery.ScalarQueryParameter("run_date", "DATE", run_date)])
        price_df = client.query(price_query, job_config=job_config).to_dataframe()
        if len(price_df) >= 1:
            latest_close = price_df['adj_close'].iloc[0]
            prev_close = price_df['adj_close'].iloc[1] if len(price_df) > 1 else None
            daily_change = (latest_close - prev_close) / prev_close * 100 if prev_close and prev_close != 0 else 0.0
            close_30d_ago = price_df['adj_close'].iloc[-1] if len(price_df) >= 30 else None
            change_30d = (latest_close - close_30d_ago) / close_30d_ago * 100 if close_30d_ago and close_30d_ago != 0 else 0.0
            industry_avg_30d = industry_averages.get(industry, {}).get("avg_change_pct_30d")
            final_json["kpis"]["price"] = {"value": round(latest_close, 2), "dailyChangePct": round(daily_change, 2), "signal": "positive" if daily_change > 0 else "negative"}
            final_json["kpis"]["thirtyDayChange"] = {"value": round(change_30d, 2), "vsIndustry": round(industry_avg_30d, 2) if pd.notna(industry_avg_30d) else "unavailable", "signal": "outperform" if pd.notna(industry_avg_30d) and change_30d > industry_avg_30d else "underperform"}

        financials_blob_name = f"financial-statements/{ticker}_{latest_quarter.strftime('%Y-%m-%d')}.json"
        json_content = gcs.read_blob(config.GCS_BUCKET_NAME, financials_blob_name)
        if json_content:
            data = json.loads(json_content)
            reports = data.get("quarterly_reports", [])
            if len(reports) >= 2:
                latest_rev = reports[0].get("income_statement", {}).get("revenue")
                prev_rev = reports[1].get("income_statement", {}).get("revenue")
                latest_eps = reports[0].get("income_statement", {}).get("eps")
                prev_eps = reports[1].get("income_statement", {}).get("eps")
                if latest_rev is not None and prev_rev is not None and prev_rev != 0:
                    rev_qoq = (latest_rev - prev_rev) / abs(prev_rev) * 100
                    final_json["kpis"]["revenueQoQ"] = {"value": round(rev_qoq, 2), "signal": "strong" if rev_qoq > 5 else "weak" if rev_qoq < 0 else "moderate"}
                if latest_eps is not None and prev_eps is not None and prev_eps != 0:
                    eps_qoq = (latest_eps - prev_eps) / abs(prev_eps) * 100
                    industry_avg_eps = industry_averages.get(industry, {}).get("avg_eps_qoq")
                    final_json["kpis"]["epsGrowth"] = {"value": round(eps_qoq, 2), "vsIndustry": round(industry_avg_eps, 2) if pd.notna(industry_avg_eps) else "unavailable", "signal": "outperform" if pd.notna(industry_avg_eps) and eps_qoq > industry_avg_eps else "underperform"}

        score_query = f"SELECT weighted_score FROM `{config.SCORES_TABLE_ID}` WHERE ticker = @ticker ORDER BY run_date DESC LIMIT 1"
        job_config_score = bigquery.QueryJobConfig(query_parameters=[bigquery.ScalarQueryParameter("ticker", "STRING", ticker)])
        score_df = client.query(score_query, job_config=job_config_score).to_dataframe()
        if not score_df.empty:
            score = score_df['weighted_score'].iloc[0]
            rec = "BUY" if score > 0.62 else "SELL" if score < 0.44 else "HOLD"
            final_json["kpis"]["aiScore"] = {"value": round((score - 0.5) * 20, 2), "recommendation": rec}

        # --- THIS IS THE MODIFIED SECTION ---
        chart_date = run_date.strftime('%Y-%m-%d')
        chart_folders = { "90dayPrice": "90-Day-Chart/", "revenueTrend": "Revenue-Chart/", "rsiMacd": "Momentum-Chart/" }
        for key, folder in chart_folders.items():
            final_json["chartUris"][key] = f"gs://{config.DESTINATION_GCS_BUCKET_NAME}/{folder}{ticker}_{chart_date}.webp"

        final_json["aiAnalystRecommendationUri"] = f"gs://{config.DESTINATION_GCS_BUCKET_NAME}/{config.RECOMMENDATION_PREFIX}{ticker}_recommendation_{run_date_str}.md"
        # --- END MODIFIED SECTION ---

        _delete_old_prep_files(ticker)
        output_blob_name = f"{OUTPUT_PREFIX}{ticker}_{run_date_str}.json"
        # The prep file itself is still written to the source bucket for the dashboard_generator to find
        gcs.write_text(config.GCS_BUCKET_NAME, output_blob_name, json.dumps(final_json, indent=2), "application/json")
        logging.info(f"[{ticker}] Successfully generated and uploaded prep JSON with destination URIs.")
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