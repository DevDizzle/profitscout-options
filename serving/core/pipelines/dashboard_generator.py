# serving/core/pipelines/dashboard_generator.py
import logging
import pandas as pd
import json
import re
import os
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
MAX_WORKERS = 8

# --- LLM Prompting ---

_EXAMPLE_JSON_FOR_LLM = """
{
  "seo": {
    "title": "Apple (AAPL) AI-Powered Dashboard: Signals & Charts | ProfitScout",
    "metaDescription": "Explore the AI-powered dashboard for Apple (AAPL), featuring data-driven buy/sell signals, key performance indicators, revenue trends, and momentum charts. Get the latest insights from ProfitScout.",
    "keywords": ["Apple stock dashboard", "AAPL stock analysis", "AAPL charts", "AAPL AI recommendation", "ProfitScout AAPL"]
  },
  "teaser": {
    "signal": "BUY",
    "summary": "AAPL is showing strong positive momentum, with its AI score indicating a favorable outlook, supported by outperforming 30-day price trends versus its industry.",
    "metrics": {
      "AI Score": "+7.5",
      "30-Day Change vs Industry": "Outperform",
      "Revenue QoQ": "Strong"
    }
  },
  "altText": {
    "90dayPrice": "90-day price chart for AAPL showing a clear uptrend, with the price consistently above its 50-day moving average, indicating bullish momentum.",
    "revenueTrend": "Quarterly revenue trend chart for AAPL, displaying steady year-over-year growth for the past four quarters.",
    "rsiMacd": "RSI and MACD chart for AAPL. The RSI is in neutral territory while the MACD line has recently crossed above its signal line, a positive indicator."
  }
}
"""

_PROMPT_TEMPLATE = r"""
You are an expert financial copywriter and SEO analyst for a fintech company named "ProfitScout". Your task is to generate a specific JSON object containing SEO metadata, a teaser summary, and chart alt text for a stock dashboard page, based ONLY on the data provided.

### Signal Policy
Use the `recommendation` from the AI Score KPI to set the teaser signal:
- If `recommendation` is "BUY" -> "BUY"
- If `recommendation` is "HOLD" -> "HOLD"
- If `recommendation` is "SELL" -> "SELL"

### Instructions
1.  **SEO Title** (60–70 chars):
    * Format: "{company_name} ({ticker}) AI-Powered Dashboard: Signals & Charts | ProfitScout"

2.  **SEO Meta Description** (150–160 chars):
    * Write a compelling summary mentioning the company, ticker, "AI-powered dashboard", "data-driven signals", "KPIs", and "ProfitScout".

3.  **SEO Keywords**:
    * Provide a list of 5 relevant keywords, including "{company_name} stock dashboard", "{ticker} stock analysis", "{ticker} charts", and "ProfitScout {ticker}".

4.  **Teaser Section**:
    * `signal`: Use the signal from the policy above.
    * `summary`: A sharp 1–2 sentence outlook combining the AI signal with insights from the KPIs provided.
    * `metrics`: A dictionary of **exactly 3** key-value pairs derived from the `kpis` data. Prioritize the AI Score, 30-Day Change vs Industry, and Revenue QoQ.

5.  **Chart Alt Text**:
    * For each chart key (`90dayPrice`, `revenueTrend`, `rsiMacd`), write a descriptive 20-40 word alt text. The text should describe what the chart shows and mention the ticker.

6.  **Format**:
    * Output **ONLY** the JSON object that matches the example structure exactly. Do not include any text, warnings, or markdown code blocks before or after the JSON.

### Input Data
- **Ticker**: {ticker}
- **Company Name**: {company_name}
- **Current Year**: {year}
- **Key Performance Indicators (KPIs)**:
{kpis_str}
- **Business Profile Snippet**:
{profile_snippet}

### Example Output (Return ONLY this JSON structure)
{example_json}
"""


# --- Helper Functions ---

def _get_additional_data(ticker: str, business_profile_uri: str) -> Dict[str, Any]:
    """Fetches business profile from GCS URI and company metadata from BigQuery."""
    data = {"profile": None, "company_name": None, "sector": None, "industry": None}
    
    # 1. Fetch Business Profile from GCS using the provided URI
    if business_profile_uri:
        profile_json_str = gcs.read_blob(config.GCS_BUCKET_NAME, business_profile_uri.replace(f"gs://{config.GCS_BUCKET_NAME}/", ""))
        if profile_json_str:
            try:
                profile_data = json.loads(profile_json_str)
                full_text = profile_data.get("business_summary", "")
                data["profile"] = {
                    "fullText": full_text,
                    "truncated": (full_text[:200] + '...') if len(full_text) > 200 else full_text,
                    "tags": [] # Placeholder for future tag extraction
                }
            except json.JSONDecodeError:
                logging.warning(f"[{ticker}] Failed to parse business profile JSON from {business_profile_uri}.")

    # 2. Fetch Company Metadata from BigQuery
    client = bigquery.Client(project=config.SOURCE_PROJECT_ID)
    query = f"""
        SELECT company_name, sector, industry
        FROM `{config.BUNDLER_STOCK_METADATA_TABLE_ID}`
        WHERE ticker = @ticker
        ORDER BY quarter_end_date DESC
        LIMIT 1
    """
    job_config = bigquery.QueryJobConfig(query_parameters=[bigquery.ScalarQueryParameter("ticker", "STRING", ticker)])
    df = client.query(query, job_config=job_config).to_dataframe()
    if not df.empty:
        data["company_name"] = df["company_name"].iloc[0]
        data["sector"] = df["sector"].iloc[0]
        data["industry"] = df["industry"].iloc[0]

    return data


def _infer_related_stocks(ticker: str, industry: str) -> List[str]:
    """Queries BQ for up to 3 other tickers in the same industry."""
    if not industry:
        return []
    client = bigquery.Client(project=config.SOURCE_PROJECT_ID)
    query = f"""
        SELECT ticker
        FROM `{config.BUNDLER_STOCK_METADATA_TABLE_ID}`
        WHERE industry = @industry AND ticker != @ticker
        GROUP BY ticker
        ORDER BY MAX(market_cap) DESC -- Order by market cap to get relevant peers
        LIMIT 3
    """
    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("industry", "STRING", industry),
            bigquery.ScalarQueryParameter("ticker", "STRING", ticker),
        ]
    )
    df = client.query(query, job_config=job_config).to_dataframe()
    return df["ticker"].tolist()


# --- Main Worker and Pipeline ---

def process_ticker(prep_blob_name: str) -> Optional[str]:
    """
    Processes one prep JSON to generate and upload a final dashboard JSON.
    """
    match = re.search(r'prep/([A-Z\.]+)_(\d{4}-\d{2}-\d{2})\.json$', prep_blob_name)
    if not match: return None
    
    ticker, run_date_str = match.groups()
    run_date = date.fromisoformat(run_date_str)
    
    logging.info(f"[{ticker}] Starting dashboard generation for {run_date_str}...")

    try:
        # 1. Load Prep Data from GCS
        prep_json_str = gcs.read_blob(config.GCS_BUCKET_NAME, prep_blob_name)
        if not prep_json_str:
            logging.warning(f"[{ticker}] Prep JSON not found or empty at {prep_blob_name}.")
            return None
        prep_data = json.loads(prep_json_str)

        # 2. Fetch Additional Data (Profile, Company Name, etc.)
        additional_data = _get_additional_data(ticker, prep_data.get("businessProfileUri"))
        company_name = additional_data.get("company_name", ticker)
        
        # 3. Infer Related Stocks
        related_stocks = _infer_related_stocks(ticker, additional_data.get("industry"))
        
        # 4. Generate Content with LLM
        prompt = _PROMPT_TEMPLATE.format(
            ticker=ticker,
            company_name=company_name,
            year=run_date.year,
            kpis_str=json.dumps(prep_data.get("kpis"), indent=2),
            profile_snippet=additional_data.get("profile", {}).get("truncated", ""),
            example_json=_EXAMPLE_JSON_FOR_LLM
        )
        
        llm_response_str = vertex_ai.generate(prompt)
        # Strip markdown and parse JSON; fallback to empty dict
        if llm_response_str.strip().startswith("```json"):
            llm_response_str = re.search(r'\{.*\}', llm_response_str, re.DOTALL).group(0)
        llm_data = json.loads(llm_response_str)

        # 5. Assemble Final Dashboard JSON
        final_dashboard = {
            "ticker": ticker, "runDate": run_date_str,
            "titleInfo": {
                "companyName": company_name, "ticker": ticker,
                "asOfDate": run_date_str,
            },
            "kpis": prep_data.get("kpis"),
            "profile": additional_data.get("profile"),
            "charts": {
                key: {"uri": uri, "alt": llm_data.get("altText", {}).get(key, f"Chart for {key} for {ticker}")}
                for key, uri in prep_data.get("chartUris", {}).items()
            },
            "seo": llm_data.get("seo"),
            "teaser": llm_data.get("teaser"),
            "relatedStocks": related_stocks,
            "calendar": {}, # Placeholder, to be handled by backend
            "options": {}, # Placeholder, to be handled by backend
        }
        
        # 6. Upload Final JSON to GCS
        output_blob_name = f"{OUTPUT_PREFIX}{ticker}_dashboard_{run_date_str}.json"
        gcs.write_text(config.GCS_BUCKET_NAME, output_blob_name, json.dumps(final_dashboard, indent=2), "application/json")
        logging.info(f"[{ticker}] Successfully generated and uploaded dashboard JSON to {output_blob_name}")
        return output_blob_name

    except Exception as e:
        logging.error(f"[{ticker}] Failed during dashboard generation: {e}", exc_info=True)
        return None

def run_pipeline():
    """Finds new prep files and generates corresponding dashboard JSONs."""
    logging.info("--- Starting Dashboard Generation Pipeline ---")
    
    all_prep_files = gcs.list_blobs(config.GCS_BUCKET_NAME, prefix=PREP_PREFIX)
    all_dashboard_files = set(gcs.list_blobs(config.GCS_BUCKET_NAME, prefix=OUTPUT_PREFIX))
    
    work_items = []
    for prep_path in all_prep_files:
        # Construct expected dashboard path from prep path
        expected_dashboard_path = prep_path.replace(PREP_PREFIX, OUTPUT_PREFIX).replace(".json", "_dashboard.json")
        if expected_dashboard_path not in all_dashboard_files:
            work_items.append(prep_path)

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