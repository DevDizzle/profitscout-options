# serving/core/pipelines/recommendation_generator.py
import logging
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from google.cloud import bigquery
from .. import config, gcs
from ..clients import vertex_ai
from datetime import date, datetime
import re

# --- Templates and other top-level definitions remain the same ---
_EXAMPLE_OUTPUT = """
# American Airlines (AAL) ✈️

**BUY** – Momentum-led breakout with supportive news flow.

**Quick take:** Momentum strength and positive headlines favor upside near term; fundamentals look stable.

### Profile
American Airlines is a major network air carrier serving passengers and cargo.
It operates an extensive domestic and international network with diversified revenue streams.

### Key Highlights
- ⚡ Momentum breakout on strong volume; trend and breadth improving.
- 📈 Price strength aligns with supportive headlines and analyst chatter.
- ⚖️ Management confident, but input costs and fuel risks persist.
- 📈 Transcript tone improved; guidance modestly higher quarter over quarter.
- ⚖️ Financials stabilizing; leverage remains a watch item.
- 📈 Fundamentals show gradual margin recovery and operating efficiency.

Overall: Momentum-led setup supported by steady fundamentals — appropriate for a BUY.

💡 Help shape the future: share your feedback to guide our next update.
"""

_PROMPT_TEMPLATE = r"""
You are a confident but approachable financial analyst writing AI-powered stock recommendations.
Tone = clear, professional, and concise.
Think: "Smarter Investing Starts Here" — give users clarity, not noise.

### Section Mapping
Map aggregated text sections into these labels (for your internal reasoning; do not print these labels):
- **Profile** → "About"
- **News Buzz** → "News Analysis"
- **Tech Signals** → "Technical Analysis"
- **Mgmt Chat** → "MD&A Analysis"
- **Earnings Scoop** → "Transcript Analysis"
- **Financials** → "Financials Analysis"
- **Fundamentals** → "Key Metrics Analysis" + "Ratios Analysis"

### Momentum Emphasis (important)
- Treat this as a **momentum-led** framework (weights tilt toward Technicals + News).
- Lead the narrative with near-term momentum; fundamentals/financials provide context.
- If momentum and fundamentals conflict, include a single ⚖️ bullet noting the tension.

### Formatting Rules (strict)
- Use this layout and spacing exactly:
  1) H1 line: "# {{company_name}} ({{ticker}}) [Emoji]"
  2) Bold recommendation line: "**BUY/HOLD/SELL** – short one-liner."
  3) A blank line, then "**Quick take:**" + 1–2 sentences.
  4) A blank line, then "### Profile" + 1–2 sentences.
  5) A blank line, then "### Key Highlights" followed by bullets.
  6) A blank line, then one sentence "Overall: …"
  7) A blank line, then a single call-to-action line (no header).
- Exactly one blank line between every block. No extra blank lines at start or end.
- **Key Highlights** must be a Markdown bulleted list using dashes (`- `), not numbered lists.
- **Bullet order (momentum-first):**
  - First bullet: ⚡ momentum summary (from Technicals/News).
  - Second bullet: 📈 or 🚩 momentum confirmation (breakouts/trend/breadth/volume).
  - Remaining bullets: strongest items from Transcript, MD&A, Financials, Fundamentals.
- Each bullet: start with one emoji (📈 bullish, 🚩 bearish, ⚖️ mixed, ⚡ momentum), then a concise statement (≤ 12 words).
- Do not include bold subsection labels inside bullets (no "**News Buzz:**", etc.).
- No additional headings beyond "### Profile" and "### Key Highlights".
- Total length ≲ 250 words.

### Content Instructions
1. **Recommendation**: Strictly "BUY" (> 0.62), "HOLD" (0.44–0.62), or "SELL" (< 0.44). Add a short, confident one-liner with **momentum context**. Do not show the raw score.
2. **Quick Take**: 1–2 sentences summarizing the overall outlook, leading with momentum.
3. **Profile**: Summarize the "About" text in **1–2 sentences**.
4. **Highlights**: Convert mapped analyses into concise emoji bullets (respect the momentum-first ordering).
5. **Wrap-Up**: End with one sentence ("Overall: ...") explaining why the call fits a momentum-led view now.
6. **Engagement Hook**: Close with a single call-to-action that encourages feedback.

### Input Data
- **Weighted Score**: {weighted_score}
- **Aggregated Analysis Text**:
{aggregated_text}

### Example Output
{example_output}
"""

def _get_all_last_gen_dates() -> dict[str, date]:
    """
    Efficiently gets the last generation date for all tickers by listing blobs once.
    """
    prefix = config.RECOMMENDATION_PREFIX
    blobs = gcs.list_blobs(config.GCS_BUCKET_NAME, prefix)
    date_regex = re.compile(r'([A-Z\.]+)_recommendation_(\d{4}-\d{2}-\d{2})\.md$')
    last_dates = {}

    for blob_name in blobs:
        match = date_regex.search(blob_name)
        if match:
            ticker, date_str = match.groups()
            try:
                blob_date = datetime.strptime(date_str, '%Y-%m-%d').date()
                if ticker not in last_dates or blob_date > last_dates[ticker]:
                    last_dates[ticker] = blob_date
            except ValueError:
                continue
    return last_dates

def _get_daily_work_list() -> list[dict]:
    """
    Builds the list of tickers that need processing for the day.
    """
    client = bigquery.Client(project=config.SOURCE_PROJECT_ID)
    today_iso = date.today().isoformat()
    
    today_scores_query = f"""
        WITH LatestMetadata AS (
            SELECT 
                ticker,
                company_name,
                ROW_NUMBER() OVER(PARTITION BY ticker ORDER BY quarter_end_date DESC) as rn
            FROM `{config.BUNDLER_STOCK_METADATA_TABLE_ID}`
        )
        SELECT 
            t1.ticker, 
            t1.weighted_score, 
            t1.aggregated_text,
            t2.company_name
        FROM `{config.SCORES_TABLE_ID}` AS t1
        LEFT JOIN LatestMetadata AS t2 ON t1.ticker = t2.ticker AND t2.rn = 1
        WHERE t1.run_date = '{today_iso}' AND t1.weighted_score IS NOT NULL
    """
    try:
        today_df = client.query(today_scores_query).to_dataframe()
        if today_df.empty: return []
    except Exception as e:
        logging.critical(f"Failed to fetch today's scores: {e}", exc_info=True)
        return []

    prev_scores_query = f"""
        WITH RankedScores AS (
            SELECT ticker, weighted_score, ROW_NUMBER() OVER(PARTITION BY ticker ORDER BY run_date DESC) as rn
            FROM `{config.SCORES_TABLE_ID}`
            WHERE run_date < '{today_iso}' AND weighted_score IS NOT NULL
        )
        SELECT ticker, weighted_score AS prev_weighted_score FROM RankedScores WHERE rn = 1
    """
    try:
        prev_df = client.query(prev_scores_query).to_dataframe()
    except Exception:
        prev_df = pd.DataFrame(columns=['ticker', 'prev_weighted_score'])

    merged_df = pd.merge(today_df, prev_df, on="ticker", how="left")
    
    merged_df['score_diff'] = (merged_df['weighted_score'] - merged_df['prev_weighted_score']).abs()
    merged_df['needs_new_text'] = (merged_df['score_diff'] >= 0.02) | (merged_df['prev_weighted_score'].isna())
    
    last_gen_dates_map = _get_all_last_gen_dates()
    last_gen_df = pd.DataFrame(list(last_gen_dates_map.items()), columns=['ticker', 'last_gen_date'])
    merged_df = pd.merge(merged_df, last_gen_df, on='ticker', how='left')

    today_date = date.today()
    merged_df['needs_new_text'] = merged_df['needs_new_text'] | (
        merged_df['last_gen_date'].apply(lambda x: (today_date - x).days >= 7 if pd.notnull(x) else True)
    )
    
    logging.info(f"Found {len(today_df)} tickers. Flagged {merged_df['needs_new_text'].sum()} for new text generation.")
    return merged_df.to_dict('records')


def _get_latest_recommendation_text_from_gcs(ticker: str) -> str | None:
    """Retrieves the text from the most recent recommendation file for a ticker."""
    prefix = f"{config.RECOMMENDATION_PREFIX}{ticker}_recommendation_"
    blobs = gcs.list_blobs(config.GCS_BUCKET_NAME, prefix)
    if not blobs: return None
    latest_blob_name = sorted(blobs, reverse=True)[0]
    try:
        content = gcs.read_blob(config.GCS_BUCKET_NAME, latest_blob_name)
        # Return content without the chart section
        return content.split("\n\n### 90-Day Performance")[0].strip() if content else None
    except Exception as e:
        logging.error(f"[{ticker}] Failed to read latest recommendation {latest_blob_name}: {e}")
    return None

def _delete_all_recommendations_for_ticker(ticker: str):
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
    Main worker function for a single ticker.
    """
    ticker = ticker_data["ticker"]
    company_name = ticker_data["company_name"]
    needs_new_text = ticker_data["needs_new_text"]
    today_str = date.today().strftime('%Y-%m-%d')
    md_blob_path = f"{config.RECOMMENDATION_PREFIX}{ticker}_recommendation_{today_str}.md"
    
    try:
        recommendation_text = ""
        if needs_new_text:
            prompt = _PROMPT_TEMPLATE.format(
                ticker=ticker,
                company_name=company_name,
                weighted_score=ticker_data["weighted_score"],
                aggregated_text=ticker_data["aggregated_text"],
                example_output=_EXAMPLE_OUTPUT
            )
            recommendation_text = vertex_ai.generate(prompt)
        else:
            recommendation_text = _get_latest_recommendation_text_from_gcs(ticker)

        if not recommendation_text:
            logging.error(f"[{ticker}] Could not get or generate text. Aborting.")
            return None
        
        _delete_all_recommendations_for_ticker(ticker)
        gcs.write_text(config.GCS_BUCKET_NAME, md_blob_path, recommendation_text, "text/markdown")
        
        logging.info(f"[{ticker}] Successfully generated and wrote recommendation to {md_blob_path}")
        return md_blob_path
        
    except Exception as e:
        logging.error(f"[{ticker}] Unhandled exception in processing: {e}", exc_info=True)
        return None

def run_pipeline():
    logging.info("--- Starting Optimized Recommendation Generation Pipeline ---")
    
    work_list = _get_daily_work_list()
    if not work_list:
        logging.warning("No tickers in daily work list. Exiting.")
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