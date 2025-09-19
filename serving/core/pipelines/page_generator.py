# serving/core/pipelines/page_generator.py
import logging
import pandas as pd
import json
import re
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from google.cloud import bigquery
from datetime import date
from typing import Dict, Optional

from .. import config, gcs
from ..clients import vertex_ai

INPUT_PREFIX = config.RECOMMENDATION_PREFIX
OUTPUT_PREFIX = config.PAGE_JSON_PREFIX

# --- Updated Example (momentum-led tone preserved) ---
_EXAMPLE_JSON_FOR_LLM = """
{
  "seo": {
    "title": "Is Expedia Group (EXPE) Breaking Out Right Now? | ProfitScout",
    "metaDescription": "AI-powered analysis signals a momentum-led BUY for Expedia Group (EXPE), with strong technicals, favorable news flow, and supportive earnings tone. Explore the full stock analysis.",
    "keywords": ["Expedia Group stock", "EXPE stock analysis", "Is EXPE a buy", "AI stock signals 2025", "EXPE technicals"]
  },
  "teaser": {
    "signal": "BUY",
    "summary": "EXPE shows a momentum breakout supported by positive headlines and stable fundamentals.",
    "metrics": {
      "Price Trend": "Uptrend with higher highs",
      "Volume Confirmation": "Above-average volume on advances",
      "Guidance Tone": "Improving quarter over quarter"
    }
  },
  "relatedStocks": ["BKNG", "ABNB", "TRIP"]
}
"""

# --- Updated Prompt (aligns thresholds + momentum emphasis) ---
_PROMPT_TEMPLATE = r"""
You are an expert financial copywriter and SEO analyst. Your task is to generate a specific JSON object with compelling SEO metadata, a teaser summary, and related stocks based on the provided analysis.

### Signal Policy (align with momentum-led framework)
Use the `weighted_score` to set the final recommendation signal:
- `weighted_score` > 0.62 → "BUY"
- `weighted_score` between 0.44 and 0.62 → "HOLD"
- `weighted_score` < 0.44 → "SELL"

### Momentum Emphasis
- Treat this analysis as **momentum-led** (weights tilt toward Technicals + News).
- Lead the teaser summary with momentum context (trend, breakouts, volume, breadth).
- If momentum and fundamentals conflict, reflect that tension concisely in the summary.

### Instructions
1) **SEO Title** (60–70 chars)
   - Frame as a question or bold, insightful statement.
   - Must include full company name "{{company_name}}" and ticker "({{ticker}})".
   - End with "| ProfitScout".

2) **Teaser Section**
   - `signal`: Use the signal from the policy above.
   - `summary`: A sharp 1–2 sentence momentum-led outlook using the aggregated text.
   - `metrics`: **Exactly 3** high-signal items. Prefer at least **one momentum indicator** (e.g., trend/breakout/volume/breadth/RSI/MA cross), plus 1–2 from earnings tone, guidance, or fundamentals.

3) **Related Stocks**
   - Infer **2–3** direct public competitor tickers from the "About" section in the aggregated text.

4) **Format**
   - Output **ONLY** the JSON object that matches the example structure exactly.

### Input Data
- **Ticker**: {ticker}
- **Company Name**: {company_name}
- **Current Year**: {year}
- **Weighted Score**: {weighted_score}
- **Full Aggregated Analysis (Text)**:
{aggregated_text}

### Example Output (Return ONLY this JSON structure)
{example_json}
"""

def _split_aggregated_text(aggregated_text: str) -> Dict[str, str]:
    """Splits aggregated_text into a dictionary of its component sections."""
    sections = re.split(r'\n\n---\n\n', aggregated_text.strip())
    section_dict = {}
    for section in sections:
        match = re.match(r'## (.*?) Analysis\n\n(.*)', section, re.DOTALL)
        if match:
            key = match.group(1).lower().replace(' ', '')
            text = match.group(2).strip()
            # --- THIS IS THE MODIFIED SECTION ---
            key_map = {
                "news": "newsSummary",
                "technicals": "technicals",
                "mda": "mdAndA",
                "transcript": "earningsCall",
                "financials": "financials",
                "fundamentals": "fundamentals" # <-- ADDED
            }
            final_key = key_map.get(key, key)
            section_dict[final_key] = text
        elif section.startswith("## About"):
            section_dict["about"] = re.sub(r'## About\n\n', '', section).strip()
    return section_dict

def _get_data_from_bq(ticker: str, run_date: str) -> Optional[Dict]:
    """
    Fetches aggregated_text, weighted_score, and company_name for a specific ticker and date.
    """
    try:
        client = bigquery.Client(project=config.SOURCE_PROJECT_ID)
        query = f"""
            SELECT
                t1.aggregated_text,
                t1.weighted_score,
                t2.company_name
            FROM `{config.SCORES_TABLE_ID}` AS t1
            LEFT JOIN `{config.BUNDLER_STOCK_METADATA_TABLE_ID}` AS t2
                ON t1.ticker = t2.ticker
            WHERE t1.ticker = @ticker AND t1.run_date = @run_date
            ORDER BY t2.quarter_end_date DESC
            LIMIT 1
        """
        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("ticker", "STRING", ticker),
                bigquery.ScalarQueryParameter("run_date", "DATE", run_date),
            ]
        )
        df = client.query(query, job_config=job_config).to_dataframe()
        return df.to_dict('records')[0] if not df.empty else None
    except Exception as e:
        logging.error(f"[{ticker}] Failed to fetch BQ data for {run_date}: {e}", exc_info=True)
        return None

def _delete_old_page_files(ticker: str):
    """Deletes all previous page JSON files for a given ticker."""
    prefix = f"{OUTPUT_PREFIX}{ticker}_page_"
    blobs_to_delete = gcs.list_blobs(config.GCS_BUCKET_NAME, prefix)
    for blob_name in blobs_to_delete:
        try:
            gcs.delete_blob(config.GCS_BUCKET_NAME, blob_name)
        except Exception as e:
            logging.error(f"[{ticker}] Failed to delete old page file {blob_name}: {e}")


def process_blob(blob_name: str) -> Optional[str]:
    """
    Processes one recommendation blob to generate a page JSON.
    """
    dated_format_regex = re.compile(r'([A-Z\.]+)_recommendation_(\d{4}-\d{2}-\d{2})\.md$')
    file_name = os.path.basename(blob_name)
    match = dated_format_regex.match(file_name)
    if not match:
        return None

    ticker, run_date_str = match.groups()

    bq_data = _get_data_from_bq(ticker, run_date_str)
    if not bq_data or not bq_data.get("company_name"):
        logging.error(f"[{ticker}] Could not find BQ data or company name for {run_date_str}.")
        return None

    aggregated_text = bq_data.get("aggregated_text")
    weighted_score = bq_data.get("weighted_score")
    company_name = bq_data.get("company_name")

    full_analysis_sections = _split_aggregated_text(aggregated_text)

    bullish_score = round((weighted_score - 0.5) * 20, 2) if weighted_score is not None else 0.0

    final_json = {
        "symbol": ticker,
        "date": run_date_str,
        "bullishScore": bullish_score,
        "fullAnalysis": full_analysis_sections
    }

    prompt = _PROMPT_TEMPLATE.format(
        ticker=ticker,
        company_name=company_name,
        year=date.today().year,
        weighted_score=round(weighted_score, 4),
        aggregated_text=aggregated_text,
        example_json=_EXAMPLE_JSON_FOR_LLM,
    )

    json_blob_path = f"{OUTPUT_PREFIX}{ticker}_page_{run_date_str}.json"
    logging.info(f"[{ticker}] Generating SEO/Teaser JSON for {run_date_str}.")

    try:
        llm_response_str = vertex_ai.generate(prompt)

        if llm_response_str.strip().startswith("```json"):
            llm_response_str = re.search(r'\{.*\}', llm_response_str, re.DOTALL).group(0)

        llm_generated_data = json.loads(llm_response_str)
        final_json.update(llm_generated_data)

        # This is the key: delete old files before writing the new one.
        _delete_old_page_files(ticker)
        gcs.write_text(config.GCS_BUCKET_NAME, json_blob_path, json.dumps(final_json, indent=2), "application/json")
        logging.info(f"[{ticker}] Successfully uploaded complete JSON file to {json_blob_path}")
        return json_blob_path

    except (json.JSONDecodeError, ValueError, AttributeError) as e:
        logging.error(f"[{ticker}] Failed to generate/parse LLM JSON. Error: {e}. Response: '{llm_response_str}'")
        return None
    except Exception as e:
        logging.error(f"[{ticker}] An unexpected error occurred: {e}", exc_info=True)
        return None


def run_pipeline():
    """
    Finds all available recommendations and generates a fresh page JSON for each,
    deleting any old page files in the process.
    """
    logging.info("--- Starting Page Generation Pipeline ---")

    # --- THIS IS THE FIX ---
    # We no longer need to check for existing pages. We simply process all
    # recommendations that are present. The deletion logic is handled inside process_blob.
    work_items = gcs.list_blobs(config.GCS_BUCKET_NAME, prefix=INPUT_PREFIX)

    if not work_items:
        logging.info("No recommendation files found to process.")
        return

    logging.info(f"Found {len(work_items)} recommendations to process into pages.")
    with ThreadPoolExecutor(max_workers=config.MAX_WORKERS_RECOMMENDER) as executor:
        futures = [executor.submit(process_blob, item) for item in work_items]
        count = sum(1 for future in as_completed(futures) if future.result())

    logging.info(f"--- Page Generation Pipeline Finished. Processed {count} new pages. ---")