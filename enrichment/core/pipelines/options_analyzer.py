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


# --- Final, Data-Rich LLM Prompting ---
_EXAMPLE_OUTPUT = """{
  "setup_quality": "Strong",
  "summary": "The stock's powerful bullish trend, confirmed by its price being over 10% above the 50D SMA and a positive 90-day performance of 22%, aligns perfectly with this call option. The reasonable implied volatility of 31% and high open interest of over 3,500 contracts suggest a well-priced and liquid entry."
}"""

_PROMPT_TEMPLATE = r"""
You are an expert options analyst with access to a comprehensive data payload for a single options contract. Your task is to synthesize this data to produce a professional quality rating and summary.

### Analysis Framework:
1.  **Primary Thesis (Trend Alignment)**: Your entire analysis MUST be framed by the `trend_signal`.
    - A "Bullish" signal strongly favors CALL options. A "Bearish" signal strongly favors PUT options.
    - This alignment is the single most important factor. A misalignment (e.g., Bullish trend for a PUT) should almost always result in a "Weak" rating.

2.  **Secondary Thesis (Momentum & Conviction)**: Use the numerical momentum indicators to gauge the strength of the primary trend.
    - `close_30d_delta_pct`: Is this strongly positive or negative?
    - `rsi`: Is the RSI confirming the trend or is it in an extreme (overbought/oversold) state that might suggest a pause?

3.  **Valuation & Risk (Volatility & Greeks)**: Assess if the option is fairly priced and what risks are present.
    - `implied_volatility_pct`: Is this high or low? Compare it to the `historical_volatility_30d_pct`. A lower IV is generally better for buyers.
    - `theta`: How much value will the option lose per day? This is especially important for low `dte` contracts.

4.  **Execution Feasibility (Liquidity)**: Determine if the option can be traded easily.
    - `liquidity_signal`: A "Poor" signal, driven by a high `spread_pct` or low `open_interest`, is a major red flag and should downgrade the quality.

### Output Instructions:
- Your response MUST be a JSON object with `setup_quality` and `summary`.
- `setup_quality`: Your final rating: "Strong", "Fair", or "Weak".
- `summary`: A single, dense sentence that justifies your rating, citing at least three specific data points (e.g., "30-day momentum," "implied volatility," "open interest") from the input to support your conclusion.

### Example Output (for format only):
{example_output}

### Comprehensive data for the contract:
{contract_data}
"""


def _load_df_to_bq(df: pd.DataFrame, table_id: str, project_id: str):
    """Truncates and loads a pandas DataFrame into a BigQuery table."""
    if df.empty:
        logging.warning("DataFrame is empty. Skipping BigQuery load.")
        return
    client = bigquery.Client(project=project_id)
    job_config = bigquery.LoadJobConfig(write_disposition="WRITE_TRUNCATE")
    try:
        job = client.load_table_from_dataframe(df, table_id, job_config=job_config)
        job.result()
        logging.info(f"Loaded {job.output_rows} rows into BigQuery table: {table_id}")
    except Exception as e:
        logging.error(f"Failed to load DataFrame to {table_id}: {e}", exc_info=True)
        raise


def _fetch_candidates_all() -> pd.DataFrame:
    """Fetches all candidates and enriches them with a comprehensive feature set."""
    client = bigquery.Client(project=config.PROJECT_ID)
    project = config.PROJECT_ID
    dataset = config.BIGQUERY_DATASET
    query = f"""
    WITH LatestRun AS (
        SELECT MAX(selection_run_ts) AS max_ts
        FROM `{project}.{dataset}.options_candidates`
    ),
    candidates AS (
        SELECT *
        FROM `{project}.{dataset}.options_candidates`
        WHERE selection_run_ts = (SELECT max_ts FROM LatestRun)
            AND options_score >= {MIN_SCORE}
    ),
    latest_analysis AS (
        SELECT *
        FROM (
            SELECT *, ROW_NUMBER() OVER(PARTITION BY ticker ORDER BY date DESC) as rn
            FROM `{project}.{dataset}.options_analysis_input`
        )
        WHERE rn = 1
    )
    SELECT
        -- From Candidates (10 fields)
        c.contract_symbol, c.option_type, c.expiration_date, c.strike, c.bid, c.ask,
        c.volume, c.open_interest, c.implied_volatility, c.delta, c.theta,
        c.fetch_date, c.ticker,

        -- From Analysis Input (5+ fields)
        a.adj_close, a.latest_sma50, a.latest_sma200, a.latest_rsi,
        a.close_30d_delta_pct, a.hv_30 as historical_volatility_30d, a.iv_signal

    FROM candidates c
    JOIN latest_analysis a ON c.ticker = a.ticker
    ORDER BY c.ticker, c.options_score DESC
    """
    df = client.query(query).to_dataframe()
    for col in ("expiration_date", "fetch_date"):
        if col in df.columns:
            df[col] = df[col].astype(str)
    return df


def _row_to_llm_payload(row: pd.Series) -> str:
    """Builds the final hybrid payload of signals and numerical values."""
    # --- Liquidity Calculations ---
    bid = row.get("bid", 0)
    ask = row.get("ask", 0)
    mid_px = (bid + ask) / 2.0 if bid > 0 and ask > 0 else 0
    spread_pct = round(((ask - bid) / mid_px) * 100, 2) if mid_px > 0 else 100.0

    # --- Signal Generation ---
    liquidity_signal = "Good"
    if row.get("open_interest", 0) < 500 or spread_pct > 20:
        liquidity_signal = "Poor"
    elif row.get("open_interest", 0) < 1000 or spread_pct > 15:
        liquidity_signal = "Fair"

    trend_signal = "Neutral"
    if pd.notna(row.get("adj_close")) and pd.notna(row.get("latest_sma50")):
        if row["adj_close"] > row["latest_sma50"]:
            trend_signal = "Bullish"
        else:
            trend_signal = "Bearish"

    rsi_signal = "Neutral"
    if pd.notna(row.get("latest_rsi")):
        if row["latest_rsi"] > 70:
            rsi_signal = "Overbought"
        elif row["latest_rsi"] < 30:
            rsi_signal = "Oversold"

    # --- Final Payload Assembly ---
    payload = {
        # --- High-Level Signals ---
        "trend_signal": trend_signal,
        "rsi_signal": rsi_signal,
        "iv_signal": row.get("iv_signal"),
        "liquidity_signal": liquidity_signal,

        # --- Key Numerical Values ---
        "underlying_price": round(row.get("adj_close", 0), 2),
        "strike_price": round(row.get("strike", 0), 2),
        "dte": (pd.to_datetime(row["expiration_date"]).date() - pd.to_datetime(row["fetch_date"]).date()).days,
        "close_30d_delta_pct": round(row.get("close_30d_delta_pct", 0), 2),
        "implied_volatility_pct": round(row.get("implied_volatility", 0) * 100, 2),
        "historical_volatility_30d_pct": round(row.get("historical_volatility_30d", 0) * 100, 2),
        "delta": round(row.get("delta", 0), 3),
        "theta": round(row.get("theta", 0), 3),
        "open_interest": row.get("open_interest"),
    }
    return json.dumps({k: v for k, v in payload.items() if pd.notna(v)}, indent=2)


def _process_contract(row: pd.Series):
    """Processes a single contract row using the final payload."""
    ticker = row["ticker"]
    csym = row.get("contract_symbol")
    prompt = _PROMPT_TEMPLATE.format(
        example_output=_EXAMPLE_OUTPUT,
        contract_data=_row_to_llm_payload(row),
    )
    try:
        resp = vertex_ai.generate(prompt)
        if resp.strip().startswith("```json"):
            resp = re.search(r'\{.*\}', resp, re.DOTALL).group(0)
        obj = json.loads(resp)
        quality = obj.get("setup_quality")
        if quality not in ("Strong", "Fair", "Weak"):
            raise ValueError(f"Invalid 'setup_quality' received: {quality}")
        return {
            "ticker": ticker,
            "run_date": row.get("fetch_date"),
            "expiration_date": row.get("expiration_date"),
            "strike_price": row.get("strike"),
            "implied_volatility": row.get("implied_volatility"),
            "iv_signal": row.get("iv_signal"),
            "stock_price_trend_signal": trend_signal, # Pass the calculated trend signal
            "setup_quality_signal": quality,
            "summary": obj.get("summary"),
            "contract_symbol": csym,
            "option_type": str(row["option_type"]).lower()
        }
    except Exception as e:
        logging.error(f"[{ticker}] Contract {csym} failed: {e}", exc_info=True)
        return None


def run_pipeline():
    """Runs the contract-level decisioning pipeline."""
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
    output_df = pd.DataFrame(results)
    logging.info(f"Generated {len(output_df)} signals. Loading to BigQuery...")
    _load_df_to_bq(output_df, OUTPUT_TABLE_ID, config.PROJECT_ID)
    logging.info(f"--- Finished. Wrote {len(output_df)} signals to {OUTPUT_TABLE_ID}. ---")