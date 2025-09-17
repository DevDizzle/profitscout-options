# serving/core/pipelines/dashboard_generator.py
import logging
import pandas as pd
import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from google.cloud import bigquery
from datetime import date
from typing import Dict, Any, List, Optional

from .. import config, gcs
from ..clients import vertex_ai

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - [%(threadName)s] - %(message)s")
PREP_PREFIX = "prep/"
OUTPUT_PREFIX = "dashboards/"
PRICE_CHART_JSON_FOLDER = "price-chart-json/"
RECOMMENDATION_PREFIX = "recommendations/"
OPTIONS_ANALYSIS_PREFIX = "options-analysis/contracts/"
MAX_WORKERS = 8

# ... (LLM Prompting section remains the same) ...
_EXAMPLE_JSON_FOR_LLM = """
{
  "seo": {
    "title": "Apple (AAPL) AI-Powered Dashboard: Signals & Charts | ProfitScout",
    "metaDescription": "Explore the AI-powered dashboard for Apple (AAPL), featuring data-driven signals on trend, momentum, and volatility. Get the latest charts and insights from ProfitScout.",
    "keywords": ["Apple stock dashboard", "AAPL stock analysis", "AAPL charts", "AAPL momentum", "ProfitScout AAPL"]
  },
  "teaser": {
    "signal": "BULLISH",
    "summary": "AAPL is demonstrating strong bullish momentum, trading above its 50-day moving average with high volume and neutral RSI, suggesting sustained investor interest.",
    "metrics": {
      "Trend Strength": "Above 50D MA",
      "RSI": "58.4 (Neutral)",
      "Volume": "+75%"
    }
  }
}
"""

_PROMPT_TEMPLATE = r"""
You are an expert financial copywriter and SEO analyst for "ProfitScout". Your task is to generate a JSON object with SEO metadata and a teaser summary for a stock dashboard, based ONLY on the KPI data provided.

### Signal Policy
Use the `signal` from the `trendStrength` KPI to set the teaser signal. Capitalize it.
- If `signal` is "bullish" -> "BULLISH"
- If `signal` is "bearish" -> "BEARISH"

### Instructions
1.  **SEO Title** (60–70 chars):
    * Format: "{company_name} ({ticker}) AI-Powered Dashboard: Signals & Charts | ProfitScout"

2.  **SEO Meta Description** (150–160 chars):
    * Write a compelling summary mentioning the company, ticker, "AI-powered dashboard", "data-driven signals", "trend", "momentum", "volatility", and "ProfitScout".

3.  **SEO Keywords**:
    * Provide a list of 5 relevant keywords, including "{company_name} stock dashboard", "{ticker} stock analysis", "{ticker} charts", and "ProfitScout {ticker}".

4.  **Teaser Section**:
    * `signal`: Use the signal from the policy above.
    * `summary`: A sharp 1–2 sentence outlook that synthesizes the provided KPIs into a cohesive narrative.
    * `metrics`: A dictionary of **exactly 3** key-value pairs derived from the `kpis` data. Use "Trend Strength", "RSI", and "Volume" as the keys. Format the values concisely (e.g., "58.4 (Neutral)", "+75%").

5.  **Format**:
    * Output **ONLY** the JSON object that matches the example structure exactly.

### Input Data
- **Ticker**: {ticker}
- **Company Name**: {company_name}
- **Key Performance Indicators (KPIs)**:
{kpis_str}

### Example Output (Return ONLY this JSON structure)
{example_json}
"""

def _slug(s: str) -> str:
    """Creates a safe filename string, consistent with the enrichment service."""
    return re.sub(r"[^a-zA-Z0-9._-]+", "-", s)[:200]

def _delete_old_dashboard_files(ticker: str):
    prefix = f"{OUTPUT_PREFIX}{ticker}_dashboard_"
    blobs_to_delete = gcs.list_blobs(config.GCS_BUCKET_NAME, prefix)
    for blob_name in blobs_to_delete:
        try:
            gcs.delete_blob(config.GCS_BUCKET_NAME, blob_name)
        except Exception as e:
            logging.error(f"[{ticker}] Failed to delete old dashboard file {blob_name}: {e}")

def _get_company_metadata(ticker: str) -> Dict[str, Any]:
    client = bigquery.Client(project=config.SOURCE_PROJECT_ID)
    query = f"SELECT company_name, sector, industry FROM `{config.BUNDLER_STOCK_METADATA_TABLE_ID}` WHERE ticker = @ticker ORDER BY quarter_end_date DESC LIMIT 1"
    job_config = bigquery.QueryJobConfig(query_parameters=[bigquery.ScalarQueryParameter("ticker", "STRING", ticker)])
    df = client.query(query, job_config=job_config).to_dataframe()
    if not df.empty:
        return df.iloc[0].to_dict()
    return {"company_name": ticker, "sector": None, "industry": None}

def _get_stock_analysis(ticker: str) -> Optional[str]:
    """Fetches the latest stock-level analysis markdown file."""
    latest_blob = gcs.get_latest_blob_for_ticker(config.GCS_BUCKET_NAME, RECOMMENDATION_PREFIX, ticker)
    return latest_blob.download_as_text() if latest_blob else None

def _get_contract_analysis(ticker: str, contract_symbol: str) -> Optional[Dict[str, Any]]:
    """Fetches the detailed analysis for a single options contract."""
    blob_name = f"{OPTIONS_ANALYSIS_PREFIX}{_slug(ticker)}/{_slug(contract_symbol)}.json"
    try:
        content = gcs.read_blob(config.GCS_BUCKET_NAME, blob_name)
        return json.loads(content) if content else None
    except (json.JSONDecodeError, Exception):
        logging.warning(f"[{ticker}] Could not find or parse analysis for {contract_symbol}")
        return None

def _get_options_chain_table(ticker: str) -> List[Dict[str, Any]]:
    """Queries top 10 candidates and enriches them with their specific AI analysis."""
    client = bigquery.Client(project=config.SOURCE_PROJECT_ID)
    query = f"""
        SELECT *
        FROM `{config.SOURCE_PROJECT_ID}.{config.BIGQUERY_DATASET}.options_candidates`
        WHERE ticker = @ticker AND selection_run_ts >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 24 HOUR)
        ORDER BY options_score DESC, rn
        LIMIT 10
    """
    job_config = bigquery.QueryJobConfig(query_parameters=[bigquery.ScalarQueryParameter("ticker", "STRING", ticker)])
    df = client.query(query, job_config=job_config).to_dataframe()
    if df.empty:
        return []

    # Clean data for JSON serialization
    for col in df.select_dtypes(include=['dbdate', 'dbtimestamp', 'datetimetz']).columns:
        df[col] = df[col].astype(str)
    
    records = df.to_dict('records')
    
    # Enrich each record with its detailed analysis
    for record in records:
        contract_symbol = record.get("contract_symbol")
        if contract_symbol:
            record['aiAnalysis'] = _get_contract_analysis(ticker, contract_symbol)
            
    return records

def _get_price_chart_data(ticker: str) -> Optional[Dict[str, Any]]:
    latest_blob = gcs.get_latest_blob_for_ticker(config.GCS_BUCKET_NAME, PRICE_CHART_JSON_FOLDER, ticker)
    if latest_blob:
        try:
            return json.loads(latest_blob.download_as_text())
        except (json.JSONDecodeError, Exception) as e:
            logging.error(f"[{ticker}] Failed to read or parse price chart JSON: {e}")
    return None

def process_ticker(prep_blob_name: str) -> Optional[str]:
    match = re.search(r'prep/([A-Z\.]+)_(\d{4}-\d{2}-\d{2})\.json$', prep_blob_name)
    if not match: return None
    
    ticker, run_date_str = match.groups()
    logging.info(f"[{ticker}] Starting dashboard generation for {run_date_str}...")

    try:
        prep_json_str = gcs.read_blob(config.GCS_BUCKET_NAME, prep_blob_name)
        if not prep_json_str: return None
        prep_data = json.loads(prep_json_str)

        metadata = _get_company_metadata(ticker)
        company_name = metadata.get("company_name", ticker)
        
        prompt = _PROMPT_TEMPLATE.format(
            ticker=ticker,
            company_name=company_name,
            kpis_str=json.dumps(prep_data.get("kpis"), indent=2),
            example_json=_EXAMPLE_JSON_FOR_LLM
        )
        
        llm_response_str = vertex_ai.generate(prompt)
        if llm_response_str.strip().startswith("```json"):
            llm_response_str = re.search(r'\{.*\}', llm_response_str, re.DOTALL).group(0)
        llm_data = json.loads(llm_response_str)

        # Assemble all data components
        stock_analysis = _get_stock_analysis(ticker)
        price_chart_data = _get_price_chart_data(ticker)
        enriched_chain_table = _get_options_chain_table(ticker)

        final_dashboard = {
            "ticker": ticker,
            "runDate": run_date_str,
            "titleInfo": {"companyName": company_name, "ticker": ticker, "asOfDate": run_date_str},
            "kpis": prep_data.get("kpis"),
            "priceChartData": price_chart_data,
            "stockLevelAnalysis": stock_analysis, # For default view
            "optionsTable": {
                "title": "Top AI-Curated Options",
                "chains": enriched_chain_table # Enriched with analysis
            },
            "seo": llm_data.get("seo"),
            "teaser": llm_data.get("teaser"),
        }
        
        _delete_old_dashboard_files(ticker)
        output_blob_name = f"{OUTPUT_PREFIX}{ticker}_dashboard_{run_date_str}.json"
        gcs.write_text(config.GCS_BUCKET_NAME, output_blob_name, json.dumps(final_dashboard, indent=2))
        logging.info(f"[{ticker}] Successfully generated and uploaded dashboard JSON to {output_blob_name}")
        return output_blob_name

    except Exception as e:
        logging.error(f"[{ticker}] Failed during dashboard generation: {e}", exc_info=True)
        return None

def run_pipeline():
    logging.info("--- Starting Dashboard Generation Pipeline ---")
    
    all_prep_files = gcs.list_blobs(config.GCS_BUCKET_NAME, prefix=PREP_PREFIX)
    all_dashboard_files = set(gcs.list_blobs(config.GCS_BUCKET_NAME, prefix=OUTPUT_PREFIX))
    
    work_items = []
    for prep_path in all_prep_files:
        try:
            ticker, run_date = re.search(r'prep/([A-Z\.]+)_(\d{4}-\d{2}-\d{2})\.json$', prep_path).groups()
            expected_dashboard_path = f"{OUTPUT_PREFIX}{ticker}_dashboard_{run_date}.json"
            if expected_dashboard_path not in all_dashboard_files:
                work_items.append(prep_path)
        except (AttributeError, IndexError):
            continue

    if not work_items:
        logging.info("All prep files have a corresponding dashboard JSON. Nothing to do.")
        return

    logging.info(f"Found {len(work_items)} new prep files to process into dashboards.")
    processed_count = 0
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_item = {executor.submit(process_ticker, item): item for item in work_items}
        for future in as_completed(future_to_item):
            try:
                if future.result():
                    processed_count += 1
            except Exception as exc:
                logging.error(f"Item {future_to_item[future]} generated an unhandled exception: {exc}", exc_info=True)
    
    logging.info(f"--- Dashboard Generation Pipeline Finished. Processed {processed_count} of {len(work_items)} new dashboards. ---")

if __name__ == "__main__":
    run_pipeline()