# enrichment/core/pipelines/options_analyzer.py
import logging
import re
import json
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
from google.cloud import bigquery
from .. import config
from ..clients import vertex_ai

# --- Configuration ---
MIN_SCORE = 0.50
MAX_WORKERS = 16
OUTPUT_TABLE_ID = f"{config.PROJECT_ID}.{config.BIGQUERY_DATASET}.options_analysis_signals"


# --- New LLM Prompting ---
_EXAMPLE_OUTPUT = """{
  "setup_quality": "Strong",
  "summary": "The contract's low implied volatility and high liquidity create a favorable risk/reward profile that aligns well with the stock's bullish technical momentum."
}"""

_PROMPT_TEMPLATE = r"""
You are an expert options analyst. Your task is to analyze a single options contract based on the provided JSON data and produce a quality rating and a summary.

### Analysis Guidelines:
- **Directional Alignment**: The `signal` field ("BUY" for bullish, "SELL" for bearish) provides the primary directional thesis. Your analysis should align with this.
- **Volatility (`iv_signal`)**: A "low" `iv_signal` is favorable for buying options, as the premium is cheaper. A "high" signal makes it more expensive and requires a stronger directional move to be profitable.
- **Liquidity**: High `open_interest` and `volume` are good. A `spread_pct` (calculated as (ask-bid)/mid_price) greater than 10% is a sign of poor liquidity and is a negative factor.
- **Greeks**: A `delta` between 0.35 and 0.60 is often ideal for balancing leverage and risk.

### Output Instructions:
- Your response MUST be a JSON object.
- It must contain exactly two keys: `setup_quality` and `summary`.
- `setup_quality`: Your rating of the trade setup. Must be one of "Strong", "Fair", or "Weak".
- `summary`: A single-sentence justification for your rating, framed as an observation of market conditions, not as a recommendation.

### Example Output (for format only):
{example_output}

### Provided data:
{contract_data}
"""


def _load_df_to_bq(df: pd.DataFrame, table_id: str, project_id: str):
    """
    Truncates and loads a pandas DataFrame into a BigQuery table.
    """
    if df.empty:
        logging.warning("DataFrame is empty. Skipping BigQuery load.")
        return

    client = bigquery.Client(project=project_id)
    job_config = bigquery.LoadJobConfig(
        write_disposition="WRITE_TRUNCATE",
    )

    try:
        job = client.load_table_from_dataframe(df, table_id, job_config=job_config)
        job.result()
        logging.info(f"Loaded {job.output_rows} rows into BigQuery table: {table_id}")
    except Exception as e:
        logging.error(f"Failed to load DataFrame to {table_id}: {e}", exc_info=True)
        raise

def _fetch_candidates_all() -> pd.DataFrame:
    """
    Fetches all candidates from the last 24 hours that meet the minimum score,
    and joins them with same-day features from options_analysis_input.
    """
    client = bigquery.Client(project=config.PROJECT_ID)
    project = config.PROJECT_ID
    dataset = config.BIGQUERY_DATASET

    query = f"""
    WITH candidates AS (
      SELECT *
      FROM `{project}.{dataset}.options_candidates`
      WHERE selection_run_ts >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 24 HOUR)
        AND options_score >= {MIN_SCORE}
    ),
    analysis AS (
      SELECT *
      FROM `{project}.{dataset}.options_analysis_input`
      WHERE date >= DATE_SUB(CURRENT_DATE(), INTERVAL 2 DAY)
    )
    SELECT
      c.*,
      a.iv_signal
    FROM candidates c
    JOIN analysis a ON c.ticker = a.ticker AND c.fetch_date = a.date
    ORDER BY c.ticker, c.options_score DESC
    """
    df = client.query(query).to_dataframe()
    # Stringify dates for JSON safety in the prompt
    for col in ("selection_run_ts", "expiration_date", "fetch_date"):
        if col in df.columns:
            df[col] = df[col].astype(str)
    return df

def _row_to_llm_payload(row: pd.Series) -> str:
    """Builds the single-contract JSON payload for the LLM."""
    # Add spread_pct calculation for the prompt
    bid = row.get("bid", 0)
    ask = row.get("ask", 0)
    mid_px = (bid + ask) / 2.0 if bid > 0 and ask > 0 else row.get("last_price", 0)
    spread_pct = ((ask - bid) / mid_px) * 100 if mid_px > 0 else None


    payload = {
        "signal": row.get("signal"),
        "iv_signal": row.get("iv_signal"),
        "option_type": str(row["option_type"]).lower(),
        "delta": row.get("delta"),
        "open_interest": row.get("open_interest"),
        "volume": row.get("volume"),
        "spread_pct": spread_pct,
    }
    return json.dumps(payload, indent=2)


def _process_contract(row: pd.Series):
    """Processes a single contract row and returns a dictionary for the BQ table."""
    ticker = row["ticker"]
    csym = row.get("contract_symbol")

    prompt = _PROMPT_TEMPLATE.format(
        example_output=_EXAMPLE_OUTPUT,
        contract_data=_row_to_llm_payload(row),
    )

    try:
        resp = vertex_ai.generate(prompt)
        # Clean up potential markdown formatting from the LLM response
        if resp.strip().startswith("```json"):
            resp = re.search(r'\{.*\}', resp, re.DOTALL).group(0)
        obj = json.loads(resp)

        quality = obj.get("setup_quality")
        if quality not in ("Strong", "Fair", "Weak"):
            raise ValueError(f"Invalid 'setup_quality' received: {quality}")

        # --- THIS IS THE FIX ---
        # We now directly use the text ("Strong", "Fair", "Weak") instead of mapping to emojis.
        return {
            "ticker": ticker,
            "run_date": row.get("fetch_date"),
            "expiration_date": row.get("expiration_date"),
            "strike_price": row.get("strike"),
            "implied_volatility": row.get("implied_volatility"),
            "iv_signal": row.get("iv_signal"),
            "setup_quality_signal": quality, # Use the direct text value
            "summary": obj.get("summary"),
            "contract_symbol": csym,
            "option_type": str(row["option_type"]).lower()
        }
    except Exception as e:
        logging.error(f"[{ticker}] Contract {csym} failed: {e}", exc_info=True)
        return None

def run_pipeline():
    """
    Runs the contract-level decisioning pipeline and loads results to BigQuery.
    """
    logging.info("--- Starting Options Analysis Signal Generation ---")
    df = _fetch_candidates_all()
    if df.empty:
        logging.warning("No candidate contracts found. Exiting.")
        return

    results = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = {ex.submit(_process_contract, row): row.get("contract_symbol") for _, row in df.iterrows()}
        for fut in as_completed(futures):
            try:
                result = fut.result()
                if result:
                    results.append(result)
            except Exception as e:
                logging.error(f"Future for {futures[fut]} failed: {e}", exc_info=True)


    if not results:
        logging.warning("No results were generated after processing. Exiting.")
        return

    # Convert results to DataFrame and load to BigQuery
    output_df = pd.DataFrame(results)
    logging.info(f"Generated {len(output_df)} signals. Loading to BigQuery...")
    _load_df_to_bq(output_df, OUTPUT_TABLE_ID, config.PROJECT_ID)

    logging.info(f"--- Finished. Wrote {len(output_df)} signals to {OUTPUT_TABLE_ID}. ---")