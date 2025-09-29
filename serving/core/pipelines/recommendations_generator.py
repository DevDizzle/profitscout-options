# serving/core/pipelines/recommendations_generator.py
import logging
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from google.cloud import bigquery
from .. import config, gcs
from ..clients import vertex_ai
from datetime import date, datetime
import re
import json

# --- Example Output ---
_EXAMPLE_OUTPUT = """
# Coty Inc. (COTY) 📉

**Neutral / Mixed outlook, as the stock's severe technical breakdown and negative news flow are creating significant short-term headwinds.**

**Quick take:** Coty is caught between signs of improving operational efficiency and a cascade of negative market data, from analyst downgrades to a deteriorating financial picture.

### Key Drivers & Headwinds
- **Waning Analyst Confidence:** A recent wave of analyst downgrades, including a significant Buy-to-Hold shift from Berenberg, signals that institutional experts are losing faith in the stock's near-term prospects. This consensus shift often precedes further price declines.
- **Confirmed Technical Breakdown:** The stock is trading firmly below its 50 and 200-day moving averages with a negative MACD. This confirms that sellers are in control and that any rallies are likely to be met with resistance.
- **Deteriorating Core Financials:** A reported 6% year-over-year revenue decrease for the recent quarter and a forward guidance for a mid-single-digit decline show that the company's core business is currently contracting, not growing.
- **A Glimmer of Fundamental Strength:** Despite top-line weakness, the company's Return on Equity (ROE) is improving. This is a crucial positive signal that management is making more efficient use of capital, which could form the basis for a future turnaround.

### The Bottom Line
While there are underlying fundamental strengths, such as strong brand equity and improving ROE, they are currently eclipsed by the overwhelming weight of negative evidence. The combination of poor technicals, negative sentiment, and declining revenue suggests the path of least resistance is lower in the near term.
"""

# --- Prompt Template ---
_PROMPT_TEMPLATE = r"""
You are an intelligence analyst creating a concise brief for a decision-maker. Your task is to read a dossier of raw intelligence (`Aggregated Analysis Text`) and extract the most critical, data-supported insights. Your goal is to help a reader quickly grasp the most important factors affecting the stock *right now*. Avoid generic statements.

### Analysis Rules (Strict)
1.  **Identify the Core Conflict**: Read all sections to find the central tension. Is it strong fundamentals vs. weak technicals? A turnaround story? A growth story with valuation concerns? This conflict is your core narrative.
2.  **Extract Specific, High-Value Data**: Do not just summarize. Pull specific, quantifiable data points (percentages, price levels, analyst names, financial metrics) from the text.
3.  **Explain the "So What?"**: For every data point you present, you MUST briefly explain its impact or why it matters to an investor.

### Formatting Rules (Strict)
- **H1 line**: `# {{company_name}} ({{ticker}}) [Emoji]`
- **Bold Outlook Line**: A single bolded line combining the `outlook_signal` and `momentum_context`.
- **Quick take**: A single sentence that summarizes the core conflict you identified.
- **Key Drivers & Headwinds**: Use "### Key Drivers & Headwinds". Under this, create 3-5 bullet points. Each bullet point MUST follow this "Theme -> Data -> Impact" structure:
    - Start with a bolded theme, like `**Waning Analyst Confidence:**`.
    - Follow with the specific, quantifiable data point from the text.
    - End with a single sentence explaining the impact of that data (the "so what?").
- **The Bottom Line**: Use "### The Bottom Line". This is a concluding paragraph synthesizing the drivers and headwinds to provide a final, forward-looking perspective.

### Input Data
- **Outlook Signal**: {outlook_signal}
- **Momentum Context**: {momentum_context}
- **Aggregated Analysis Text**:
{aggregated_text}

### Example Output
{example_output}
"""
def _get_signal_and_context(score: float, momentum_pct: float | None) -> tuple[str, str]:
    """
    Determines the 5-tier outlook signal and the momentum context.
    """
    # --- THIS IS THE FIX ---
    # The scoring ranges have been adjusted to prevent any gaps.
    if score >= 0.75:
        outlook = "Strongly Bullish"
    elif 0.60 <= score < 0.75:
        outlook = "Moderately Bullish"
    elif 0.40 <= score < 0.60:
        outlook = "Neutral / Mixed"
    elif 0.25 <= score < 0.40:
        outlook = "Moderately Bearish"
    else: # score < 0.25
        outlook = "Strongly Bearish"


    context = ""
    if momentum_pct is not None:
        is_bullish_outlook = "Bullish" in outlook
        is_bearish_outlook = "Bearish" in outlook

        if is_bullish_outlook and momentum_pct > 0:
            context = "with confirming positive momentum."
        elif is_bullish_outlook and momentum_pct < 0:
            context = "encountering short-term weakness."
        elif is_bearish_outlook and momentum_pct < 0:
            context = "with confirming negative momentum."
        elif is_bearish_outlook and momentum_pct > 0:
            context = "encountering a short-term rally."

    return outlook, context


def _get_daily_work_list() -> list[dict]:
    """
    MODIFIED: Builds the work list from the GCS tickerlist.txt and enriches it
    with the latest available data from BigQuery for each ticker.
    This version now handles potential duplicate entries by selecting the one with the highest weighted_score.
    """
    logging.info("Fetching work list from GCS and enriching from BigQuery...")
    tickers = gcs.get_tickers()
    if not tickers:
        logging.critical("Ticker list from GCS is empty. No work to do.")
        return []
        
    client = bigquery.Client(project=config.SOURCE_PROJECT_ID)
    
    # The query now de-duplicates the data by selecting the record with the highest
    # `weighted_score` for each ticker on the most recent run date.
    query = f"""
        WITH GCS_Tickers AS (
            SELECT ticker FROM UNNEST(@tickers) AS ticker
        ),
        RankedScores AS (
            SELECT
                t1.ticker,
                t2.company_name,
                t1.weighted_score,
                t1.aggregated_text,
                ROW_NUMBER() OVER(PARTITION BY t1.ticker ORDER BY t1.run_date DESC, t1.weighted_score DESC) as rn
            FROM `{config.SCORES_TABLE_ID}` AS t1
            JOIN `{config.BUNDLER_STOCK_METADATA_TABLE_ID}` AS t2 ON t1.ticker = t2.ticker
            WHERE t1.weighted_score IS NOT NULL AND t2.company_name IS NOT NULL
        ),
        LatestScores AS (
            SELECT * FROM RankedScores WHERE rn = 1
        ),
        LatestMomentum AS (
            SELECT
                ticker,
                close_30d_delta_pct
            FROM (
                SELECT
                    ticker,
                    close_30d_delta_pct,
                    ROW_NUMBER() OVER(PARTITION BY ticker ORDER BY date DESC) as rn
                FROM `{config.SOURCE_PROJECT_ID}.{config.BIGQUERY_DATASET}.options_analysis_input`
                WHERE close_30d_delta_pct IS NOT NULL
            )
            WHERE rn = 1
        )
        SELECT
            g.ticker,
            s.company_name,
            s.weighted_score,
            s.aggregated_text,
            m.close_30d_delta_pct
        FROM GCS_Tickers g
        LEFT JOIN LatestScores s ON g.ticker = s.ticker
        LEFT JOIN LatestMomentum m ON g.ticker = m.ticker
    """
    
    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ArrayQueryParameter("tickers", "STRING", tickers),
        ]
    )

    try:
        df = client.query(query, job_config=job_config).to_dataframe()
        df.dropna(subset=['company_name', 'weighted_score', 'aggregated_text'], inplace=True)
        if df.empty:
            logging.warning("No tickers with sufficient data found after enriching from BigQuery.")
            return []
        logging.info(f"Successfully created work list for {len(df)} tickers.")
        return df.to_dict('records')
    except Exception as e:
        logging.critical(f"Failed to build and enrich work list: {e}", exc_info=True)
        return []


def _delete_old_recommendation_files(ticker: str):
    """Deletes all old recommendation files for a given ticker."""
    prefix = f"{config.RECOMMENDATION_PREFIX}{ticker}_recommendation_"
    blobs_to_delete = gcs.list_blobs(config.GCS_BUCKET_NAME, prefix)
    for blob_name in blobs_to_delete:
        try:
            gcs.delete_blob(config.GCS_BUCKET_NAME, blob_name)
        except Exception as e:
            logging.error(f"[{ticker}] Failed to delete old file {blob_name}: {e}")

def _process_ticker(ticker_data: dict):
    """
    Generates recommendation markdown and its companion JSON metadata file.
    """
    ticker = ticker_data["ticker"]
    today_str = date.today().strftime('%Y-%m-%d')
    
    base_blob_path = f"{config.RECOMMENDATION_PREFIX}{ticker}_recommendation_{today_str}"
    md_blob_path = f"{base_blob_path}.md"
    json_blob_path = f"{base_blob_path}.json"
    
    try:
        momentum_pct = ticker_data.get("close_30d_delta_pct")
        if pd.isna(momentum_pct):
            momentum_pct = None

        outlook_signal, momentum_context = _get_signal_and_context(
            ticker_data["weighted_score"],
            momentum_pct
        )

        prompt = _PROMPT_TEMPLATE.format(
            ticker=ticker,
            company_name=ticker_data["company_name"],
            outlook_signal=outlook_signal,
            momentum_context=momentum_context,
            aggregated_text=ticker_data["aggregated_text"],
            example_output=_EXAMPLE_OUTPUT
        )
        
        recommendation_text = vertex_ai.generate(prompt)

        if not recommendation_text:
            logging.error(f"[{ticker}] LLM returned no text. Aborting.")
            return None
        
        metadata = {
            "ticker": ticker,
            "run_date": today_str,
            "outlook_signal": outlook_signal,
            "momentum_context": momentum_context,
            "weighted_score": ticker_data["weighted_score"],
            "recommendation_md_path": f"gs://{config.GCS_BUCKET_NAME}/{md_blob_path}"
        }
        
        _delete_old_recommendation_files(ticker)
        
        gcs.write_text(config.GCS_BUCKET_NAME, md_blob_path, recommendation_text, "text/markdown")
        gcs.write_text(config.GCS_BUCKET_NAME, json_blob_path, json.dumps(metadata, indent=2), "application/json")
        
        logging.info(f"[{ticker}] Successfully generated and wrote recommendation files to {md_blob_path} and {json_blob_path}")
        return md_blob_path
        
    except Exception as e:
        logging.error(f"[{ticker}] Unhandled exception in processing: {e}", exc_info=True)
        return None

def run_pipeline():
    logging.info("--- Starting Advanced Recommendation Generation Pipeline ---")
    
    work_list = _get_daily_work_list()
    if not work_list:
        return
    
    processed_count = 0
    with ThreadPoolExecutor(max_workers=config.MAX_WORKERS_RECOMMENDER) as executor:
        future_to_ticker = {
            executor.submit(_process_ticker, item): item['ticker']
            for item in work_list
        }
        for future in as_completed(future_to_ticker):
            if future.result():
                processed_count += 1
                
    logging.info(f"--- Recommendation Pipeline Finished. Processed {processed_count} of {len(work_list)} tickers. ---")