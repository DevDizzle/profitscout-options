# serving/core/pipelines/recommendations_generator.py
import logging
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from google.cloud import bigquery
from .. import config, gcs
from ..clients import vertex_ai
from datetime import date, datetime
import re

# --- Updated: New Example reflecting the richer analysis ---
_EXAMPLE_OUTPUT = """
# Oracle (ORCL) ☁️

**Strongly Bullish outlook encountering short-term weakness.**

**Quick take:** While current price action is weak, Oracle's strong fundamentals, driven by booming AI demand and a massive order backlog, create a compelling long-term picture.

### Profile
Oracle provides comprehensive enterprise IT solutions, including cloud applications (OCA) and infrastructure (OCI), serving a wide range of global clients.

### Key Highlights
- ⚖️ **Conflicting Signals:** The powerful long-term fundamental story is not yet reflected in the current bearish momentum.
- 📈 Management reports booming demand for AI infrastructure, with Remaining Performance Obligations (RPO) now at $455B.
- 📈 Recent quarter revenue grew 12%, driven by strong cloud performance.
- 🚩 Price is trading below its key 21-day and 50-day moving averages, confirming the short-term downtrend.
- 🚩 Rising debt (now over $105B) and negative free cash flow are key watch items.

Overall: A Strongly Bullish outlook is warranted based on fundamentals, but traders should be aware of the negative short-term price momentum.

💡 Help shape the future: share your feedback to guide our next update.
"""

# --- Updated: New Prompt accepting the richer signal context ---
_PROMPT_TEMPLATE = r"""
You are a confident but approachable financial analyst writing AI-powered stock recommendations.
Your tone should be clear, professional, and concise. Your goal is to give users clarity, not noise.

### Analysis Rules (Very Important)
- **Lead with the Primary Outlook**: The `outlook_signal` is your main thesis (e.g., "Strongly Bullish").
- **Integrate the Momentum Context**: The `momentum_context` tells you if the short-term price trend agrees or disagrees with the main outlook. You MUST reflect this in your analysis. If they conflict, highlight the tension (e.g., "Strong fundamentals are clashing with weak price action.").
- **Momentum-First Narrative**: Start your "Quick take" and "Key Highlights" with the momentum story (price trends, chart patterns) before discussing fundamentals (earnings, financials).

### Formatting Rules (Strict)
- Use this layout and spacing exactly.
- **H1 line**: `# {{company_name}} ({{ticker}}) [Emoji]`
- **Bold Outlook Line**: A single bolded line combining the `outlook_signal` and `momentum_context`.
- **Quick Take**: 1-2 sentences summarizing the overall outlook, leading with the momentum vs. fundamental picture.
- **Profile & Key Highlights**: Use "### Profile" and "### Key Highlights" headers.
- **Bullets**: Use Markdown dashes (`- `). Start each bullet with one emoji (📈 bullish, 🚩 bearish, ⚖️ mixed/conflicting). Each bullet must be a concise, data-driven insight from the input text. Cite specific numbers and trends.
- **Overall**: One sentence that reconciles the momentum view with the fundamental context.
- **Hook**: Close with a single call-to-action encouraging feedback.

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
    # 1. Determine the primary outlook signal based on score
    if score > 0.75:
        outlook = "Strongly Bullish"
    elif 0.60 <= score <= 0.74:
        outlook = "Moderately Bullish"
    elif 0.40 <= score <= 0.59:
        outlook = "Neutral / Mixed"
    elif 0.25 <= score <= 0.39:
        outlook = "Moderately Bearish"
    else: # score < 0.25
        outlook = "Strongly Bearish"

    # 2. Determine the momentum context
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
    Builds the list of tickers to process, now including 30-day momentum.
    """
    client = bigquery.Client(project=config.SOURCE_PROJECT_ID)
    today_iso = date.today().isoformat()
    
    # NEW: Join with options_analysis_input to get the momentum data
    query = f"""
        WITH LatestScores AS (
            SELECT 
                ticker,
                company_name,
                weighted_score,
                aggregated_text,
                run_date
            FROM (
                SELECT 
                    t1.ticker,
                    t2.company_name,
                    t1.weighted_score,
                    t1.aggregated_text,
                    t1.run_date,
                    ROW_NUMBER() OVER(PARTITION BY t1.ticker ORDER BY t1.run_date DESC) as rn_score,
                    ROW_NUMBER() OVER(PARTITION BY t1.ticker ORDER BY t2.quarter_end_date DESC) as rn_meta
                FROM `{config.SCORES_TABLE_ID}` AS t1
                LEFT JOIN `{config.BUNDLER_STOCK_METADATA_TABLE_ID}` AS t2 
                    ON t1.ticker = t2.ticker
                WHERE t1.run_date = '{today_iso}' AND t1.weighted_score IS NOT NULL
            )
            WHERE rn_score = 1 AND rn_meta = 1
        ),
        Momentum AS (
            SELECT
                ticker,
                close_30d_delta_pct
            FROM (
                SELECT 
                    ticker,
                    close_30d_delta_pct,
                    ROW_NUMBER() OVER(PARTITION BY ticker ORDER BY date DESC) as rn
                FROM `{config.SOURCE_PROJECT_ID}.{config.BIGQUERY_DATASET}.options_analysis_input`
            )
            WHERE rn = 1
        )
        SELECT
            s.ticker,
            s.company_name,
            s.weighted_score,
            s.aggregated_text,
            m.close_30d_delta_pct
        FROM LatestScores s
        LEFT JOIN Momentum m ON s.ticker = m.ticker
    """

    try:
        df = client.query(query).to_dataframe()
        if df.empty:
            logging.warning("No tickers found in daily work list. Exiting.")
            return []
        return df.to_dict('records')
    except Exception as e:
        logging.critical(f"Failed to build daily work list: {e}", exc_info=True)
        return []


def _delete_all_recommendations_for_ticker(ticker: str):
    """Deletes all old recommendation files for a given ticker to ensure only the latest exists."""
    prefix = f"{config.RECOMMENDATION_PREFIX}{ticker}_recommendation_"
    blobs_to_delete = gcs.list_blobs(config.GCS_BUCKET_NAME, prefix)
    for blob_name in blobs_to_delete:
        try:
            gcs.delete_blob(config.GCS_BUCKET_NAME, blob_name)
        except Exception as e:
            logging.error(f"[{ticker}] Failed to delete old file {blob_name}: {e}")

def _process_ticker(ticker_data: dict):
    """
    Main worker function for a single ticker using the new logic.
    """
    ticker = ticker_data["ticker"]
    today_str = date.today().strftime('%Y-%m-%d')
    md_blob_path = f"{config.RECOMMENDATION_PREFIX}{ticker}_recommendation_{today_str}.md"
    
    try:
        # Generate the new, richer signals
        outlook_signal, momentum_context = _get_signal_and_context(
            ticker_data["weighted_score"],
            ticker_data.get("close_30d_delta_pct")
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
        
        # Clean slate: remove old files before writing the new one
        _delete_all_recommendations_for_ticker(ticker)
        gcs.write_text(config.GCS_BUCKET_NAME, md_blob_path, recommendation_text, "text/markdown")
        
        logging.info(f"[{ticker}] Successfully generated and wrote recommendation to {md_blob_path}")
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