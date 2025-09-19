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
    Builds the list of tickers to process.
    MODIFIED: This version fetches the LATEST score for ALL tickers,
    not just those from today, to allow for a full rebuild.
    """
    client = bigquery.Client(project=config.SOURCE_PROJECT_ID)
    
    # --- THIS IS THE "REBUILD ALL" QUERY ---
    # It finds the most recent score for every ticker, ignoring the date.
    query = f"""
        WITH
        -- Step 1: Get the most recent score for every ticker
        LatestScores AS (
            SELECT
                ticker,
                company_name,
                weighted_score,
                aggregated_text
            FROM (
                SELECT
                    t1.ticker,
                    t2.company_name,
                    t1.weighted_score,
                    t1.aggregated_text,
                    ROW_NUMBER() OVER(PARTITION BY t1.ticker ORDER BY t1.run_date DESC) as rn
                FROM `{config.SCORES_TABLE_ID}` AS t1
                JOIN `{config.BUNDLER_STOCK_METADATA_TABLE_ID}` AS t2 ON t1.ticker = t2.ticker
                WHERE t1.weighted_score IS NOT NULL AND t2.company_name IS NOT NULL
            )
            WHERE rn = 1
        ),
        -- Step 2: Get the most recent momentum data for every ticker
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
        -- Step 3: Join them together
        SELECT
            s.ticker,
            s.company_name,
            s.weighted_score,
            s.aggregated_text,
            m.close_30d_delta_pct
        FROM LatestScores s
        LEFT JOIN LatestMomentum m ON s.ticker = m.ticker
    """

    try:
        df = client.query(query).to_dataframe()
        if df.empty:
            logging.warning("No tickers found in work list. Exiting.")
            return []
        return df.to_dict('records')
    except Exception as e:
        logging.critical(f"Failed to build work list: {e}", exc_info=True)
        return []


def _delete_old_recommendation_files(ticker: str):
    """Deletes all old recommendation files (.md and .json) for a given ticker."""
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