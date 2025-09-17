import logging
import re
import json
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
from google.cloud import bigquery
from .. import config, gcs
from ..clients import vertex_ai

OUTPUT_PREFIX = "options-analysis/contracts/"
MIN_SCORE = 0.50
MAX_WORKERS = 16

# --- Example output: explicit contract-level decision ---
_EXAMPLE_OUTPUT = """{
  "decision": "BUY",
  "strategy": "PUT",
  "contract": {
    "ticker": "MIDD",
    "option_type": "put",
    "expiration_date": "2025-10-18",
    "strike": 150.0,
    "contract_symbol": "MIDD251018P00150000"
  },
  "analysis": "This contract is a strong candidate because its bearish direction aligns with the overall SELL signal. The implied volatility is elevated relative to its historical volatility and industry peers, suggesting the premium is rich, which is favorable for selling puts or buying puts. The technicals are also supportive, with the MACD trending down and the price below its key moving averages.",
  "reasoning": "The decision to BUY is based on the combination of a clear bearish signal, favorable volatility dynamics, and supportive technical indicators. The contract has good liquidity and the strike price is at a reasonable out-of-the-money level."
}"""

# --- Prompt: single-contract verdict (BUY / NO_BUY) ---
_PROMPT_TEMPLATE = r"""
You are an options analyst. Your task is to decide whether to BUY or NO_BUY a single options contract based on the provided JSON data. The JSON contains the contract's details and enriched data for the underlying ticker.

### Analysis Guidelines:
- **Directional Alignment**: If the `signal` is "BUY", you should favor CALL options. If the `signal` is "SELL", you should favor PUT options.
- **Liquidity**: High open interest and volume are good. A `spread_pct` greater than 10% is a red flag.
- **Greeks**: A `delta` between 0.35 and 0.60 is ideal for leverage. High `theta` indicates rapid time decay. `Vega` shows sensitivity to volatility.
- **Volatility**: Compare `iv_avg` to `hv_30` and `iv_industry_avg`. A low `iv_avg` suggests the option is cheap (good for buying), while a high `iv_avg` suggests it's expensive.
- **Technicals**: Use RSI, MACD, and SMAs to confirm the trend. The 30-day and 90-day deltas indicate momentum.

### Output Instructions:
- Your response must be a JSON object with the following structure:
  - `decision`: "BUY" or "NO_BUY".
  - `strategy`: "CALL" or "PUT".
  - `contract`: A dictionary with `ticker`, `option_type`, `expiration_date`, `strike`, and `contract_symbol`.
  - `analysis`: A 100–150 word justification for your decision.
  - `reasoning`: A 50–100 word explanation of why this contract is or isn't a good trade.

### Example Output (for format only):
{example_output}

### Provided data:
{contract_data}
"""

def _slug(s: str) -> str:
    """Creates a safe filename for GCS."""
    return re.sub(r"[^a-zA-Z0-9._-]+", "-", s)[:200]

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
      WHERE date >= DATE_SUB(CURRENT_DATE(), INTERVAL 2 DAY) -- Look back 2 days for a match
    )
    SELECT
      c.*,
      a.open AS underlying_open,
      a.high AS underlying_high,
      a.low AS underlying_low,
      a.adj_close AS underlying_adj_close,
      a.volume AS underlying_volume,
      a.iv_avg,
      a.hv_30,
      a.iv_industry_avg,
      a.iv_signal,
      a.latest_rsi,
      a.latest_macd,
      a.latest_sma50,
      a.latest_sma200,
      a.close_30d_delta_pct,
      a.rsi_30d_delta,
      a.macd_30d_delta,
      a.close_90d_delta_pct,
      a.rsi_90d_delta,
      a.macd_90d_delta
    FROM candidates c
    JOIN analysis a ON c.ticker = a.ticker AND c.fetch_date = a.date
    ORDER BY c.ticker, c.options_score DESC
    """
    df = client.query(query).to_dataframe()
    # Stringify dates for JSON safety
    for col in ("selection_run_ts", "expiration_date", "fetch_date"):
        if col in df.columns:
            df[col] = df[col].astype(str)
    return df

def _row_to_llm_payload(row: pd.Series) -> str:
    """Builds the single-contract JSON payload for the LLM."""
    contract = {
        "ticker": row["ticker"],
        "option_type": str(row["option_type"]).lower(),
        "expiration_date": row["expiration_date"],
        "strike": float(row["strike"]) if pd.notna(row["strike"]) else 0.0,
        "contract_symbol": row.get("contract_symbol", ""),
        "last_price": row.get("last_price"),
        "bid": row.get("bid"),
        "ask": row.get("ask"),
        "volume": row.get("volume"),
        "open_interest": row.get("open_interest"),
        "implied_volatility": row.get("implied_volatility"),
        "delta": row.get("delta"),
        "theta": row.get("theta"),
        "vega": row.get("vega"),
        "gamma": row.get("gamma"),
        "options_score": row.get("options_score"),
    }
    enriched = {
        "signal": row.get("signal"),
        "underlying_open": row.get("underlying_open"),
        "underlying_high": row.get("underlying_high"),
        "underlying_low": row.get("underlying_low"),
        "underlying_adj_close": row.get("underlying_adj_close"),
        "underlying_volume": row.get("underlying_volume"),
        "iv_avg": row.get("iv_avg"),
        "hv_30": row.get("hv_30"),
        "iv_industry_avg": row.get("iv_industry_avg"),
        "iv_signal": row.get("iv_signal"),
        "latest_rsi": row.get("latest_rsi"),
        "latest_macd": row.get("latest_macd"),
        "latest_sma50": row.get("latest_sma50"),
        "latest_sma200": row.get("latest_sma200"),
        "close_30d_delta_pct": row.get("close_30d_delta_pct"),
        "rsi_30d_delta": row.get("rsi_30d_delta"),
        "macd_30d_delta": row.get("macd_30d_delta"),
        "close_90d_delta_pct": row.get("close_90d_delta_pct"),
        "rsi_90d_delta": row.get("rsi_90d_delta"),
        "macd_90d_delta": row.get("macd_90d_delta"),
    }
    payload = {
        "contract": contract,
        "enriched_data": enriched,
    }
    return json.dumps(payload, indent=2)

def _process_contract(row: pd.Series):
    """Processes a single contract row and generates a BUY/NO_BUY JSON."""
    ticker = row["ticker"]
    csym = row.get("contract_symbol") or f"{ticker}_{row['expiration_date']}_{row['strike']}"
    blob_name = f"{OUTPUT_PREFIX}{_slug(ticker)}/{_slug(csym)}.json"

    prompt = _PROMPT_TEMPLATE.format(
        example_output=_EXAMPLE_OUTPUT,
        contract_data=_row_to_llm_payload(row),
    )

    try:
        resp = vertex_ai.generate(prompt)
        obj = json.loads(resp)

        if obj.get("decision") not in ("BUY", "NO_BUY"):
            raise ValueError("Response missing or invalid 'decision'")
        if "contract" not in obj:
            raise ValueError("Response missing 'contract' block")

        sc = obj["contract"]
        sc.setdefault("ticker", ticker)
        sc.setdefault("option_type", str(row["option_type"]).lower())
        sc.setdefault("expiration_date", row["expiration_date"])
        sc.setdefault("strike", float(row["strike"]) if pd.notna(row["strike"]) else 0.0)
        sc.setdefault("contract_symbol", csym)

        gcs.write_text(config.GCS_BUCKET_NAME, blob_name, json.dumps(obj, indent=2), "application/json")
        logging.info(f"[{ticker}] Wrote {obj['decision']} to gs://{config.GCS_BUCKET_NAME}/{blob_name}")
        return blob_name
    except Exception as e:
        logging.error(f"[{ticker}] Contract {csym} failed: {e}", exc_info=True)
        return None

def run_pipeline():
    """
    Runs the contract-level decisioning pipeline for all candidates.
    """
    logging.info("--- Starting Contract Decision Pipeline ---")
    df = _fetch_candidates_all()
    if df.empty:
        logging.warning("No candidate contracts found. Exiting.")
        return

    processed = 0
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = [ex.submit(_process_contract, row) for _, row in df.iterrows()]
        for fut in as_completed(futures):
            if fut.result():
                processed += 1

    logging.info(f"--- Finished. Decisions written for {processed}/{len(df)} contracts. ---")