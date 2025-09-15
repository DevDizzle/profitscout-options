import logging
import pandas as pd
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from google.cloud import bigquery
from .. import config, gcs
from ..clients import vertex_ai

OUTPUT_PREFIX = "options-analysis/"

# --- Updated example generated, now selecting one best contract per ticker with reasoning ---
_EXAMPLE_OUTPUT = """{
  "selected_contract": {
    "ticker": "MIDD",
    "expiration_date": "2025-10-18",
    "strike": 150.0
  },
  "analysis": "For MIDD's SELL signal, the selected put at 150 strike (exp 2025-10-18) stands out with 5.2% OTM, 28 DTE for balanced decay, and superior liquidity (OI 2,500, vol 1,200) vs. alternatives. Theta -0.03 minimizes time risk, vega 0.14 captures IV spikes (current 42%, 65th percentile > HV_30 32%), and delta -0.45 offers leverage on downside. Aligns with 30d close delta -7.5% and RSI 32 (30d delta -10, oversold). MACD negative crossover reinforces bearish bias over 90d trends.",
  "reasoning": "Chose this over the 145 strike (too deep OTM, low delta 0.3, poor liquidity OI 800) and 155 strike (higher theta risk -0.05, IV misalignment) because it balances leverage, liquidity, and technical fit—exceptional for 25% ROI on 4% drop vs. peers' 15%."
}"""

_PROMPT_TEMPLATE = r"""
You are a sharp financial analyst specializing in equity options. Your task is to evaluate multiple viable option contracts for a ticker, incorporating enriched price and technical data, then select the single best one with reasoning.
Use **only** the JSON data provided (array of contracts + shared enriched fields).

### Key Interpretation Guidelines
1.  **Compare & Select**: Evaluate all contracts on alignment with signal (BUY/SELL = CALL/PUT), moneyness/leverage, liquidity (OI/volume/spread), risks (theta/vega), IV context (avg/percentile/HV/industry/signal), and technicals (RSI/MACD/SMAs + 30d/90d deltas for momentum).
2.  **Best Criteria**: Prioritize: High options_score, good liquidity, optimal DTE (15-45 for balance), delta ~0.4-0.6 for leverage, low theta risk, IV favoring direction (low percentile for buys, high for sells). Use 30d deltas for near-term timing, 90d for trend confirmation.
3.  **Value Add**: Estimate ROI scenarios (e.g., on 5% stock move), suggest strategy tweaks (e.g., pair with stock), highlight edges from technicals (e.g., SMA crossover + price delta).

### Step-by-Step Reasoning
1.  Assess each contract's strengths/weaknesses relative to others.
2.  Select one best overall, explaining why it outperforms alternatives (e.g., better IV alignment + liquidity).
3.  Provide dense analysis for the selected, plus separate reasoning comparing to others.

### Output — return exactly this JSON, nothing else
{
  "selected_contract": {
    "ticker": "<ticker string>",
    "expiration_date": "<YYYY-MM-DD from selected>",
    "strike": <float from selected>
  },
  "analysis": "<One dense paragraph (100-150 words) on the selected trade idea, integrating liquidity, risks, IV, technicals, and signal alignment. Include ROI estimates.>",
  "reasoning": "<Concise paragraph (50-100 words) explaining why this over others, citing key diffs in score, Greeks, technicals, etc.>"
}

### Example Output (for format only; do not copy wording)
{example_output}

### Provided data:
{contract_data}
"""

def _get_options_work_list() -> dict[str, pd.DataFrame]:
    """
    Fetches all of today's options candidates from BigQuery, joined with enriched max_date data from price_data.
    Returns a dict of ticker: sub-DF for grouped processing.
    """
    client = bigquery.Client(project=config.PROJECT_ID)
    query = f"""
        WITH latest_price AS (
            SELECT ticker, MAX(date) AS max_date
            FROM `{config.PROJECT_ID}.{config.BIGQUERY_DATASET}.price_data`
            GROUP BY ticker
        )
        SELECT
            c.*,
            s.company_name,
            p.open AS underlying_open,
            p.high AS underlying_high,
            p.low AS underlying_low,
            p.adj_close AS underlying_adj_close,
            p.volume AS underlying_volume,
            p.iv_avg,
            p.hv_30,
            p.iv_percentile,
            p.iv_industry_avg,
            p.iv_signal,
            p.latest_rsi,
            p.latest_macd,
            p.latest_sma50,
            p.latest_sma200,
            p.close_30d_delta_pct,
            p.rsi_30d_delta,
            p.macd_30d_delta,
            p.close_90d_delta_pct,
            p.rsi_90d_delta,
            p.macd_90d_delta
        FROM `{config.CAND_TABLE}` AS c
        JOIN (
            SELECT ticker, company_name, ROW_NUMBER() OVER(PARTITION BY ticker ORDER BY quarter_end_date DESC) as rn
            FROM `{config.PROJECT_ID}.{config.BIGQUERY_DATASET}.stock_metadata`
        ) AS s ON c.ticker = s.ticker AND s.rn = 1
        JOIN latest_price lp ON c.ticker = lp.ticker
        JOIN `{config.PROJECT_ID}.{config.BIGQUERY_DATASET}.price_data` p 
            ON lp.ticker = p.ticker AND lp.max_date = p.date
        WHERE c.selection_run_ts >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 24 HOUR)
          AND c.options_score >= 0.5
        ORDER BY c.ticker, c.options_score DESC
    """
    try:
        df = client.query(query).to_dataframe()
        # Convert dates/timestamps to strings for JSON
        if 'selection_run_ts' in df.columns:
            df['selection_run_ts'] = df['selection_run_ts'].astype(str)
        if 'expiration_date' in df.columns:
            df['expiration_date'] = df['expiration_date'].astype(str)
        if 'fetch_date' in df.columns:
            df['fetch_date'] = df['fetch_date'].astype(str)
        if 'max_date' in df.columns:
            df['max_date'] = df['max_date'].astype(str)
        
        # Group by ticker
        grouped = {ticker: sub_df for ticker, sub_df in df.groupby('ticker')}
        return grouped
    except Exception as e:
        logging.critical(f"Failed to fetch options work list: {e}", exc_info=True)
        return {}


def _process_ticker(ticker: str, contracts_df: pd.DataFrame):
    """
    Main worker for a ticker: Sends all contracts + enriched data to LLM, which selects one best with reasoning/analysis.
    Generates one JSON per ticker.
    """
    analysis_blob_path = f"{OUTPUT_PREFIX}{ticker}_best_option_analysis.json"
    logging.info(f"[{ticker}] Generating best option analysis from {len(contracts_df)} candidates.")

    try:
        # Prepare data: Array of contracts + shared enriched fields (take first row's enriched since same for ticker)
        shared_enriched = contracts_df.iloc[0][['underlying_open', 'underlying_high', 'underlying_low', 'underlying_adj_close', 'underlying_volume',
                                                'iv_avg', 'hv_30', 'iv_percentile', 'iv_industry_avg', 'iv_signal',
                                                'latest_rsi', 'latest_macd', 'latest_sma50', 'latest_sma200',
                                                'close_30d_delta_pct', 'rsi_30d_delta', 'macd_30d_delta',
                                                'close_90d_delta_pct', 'rsi_90d_delta', 'macd_90d_delta']].to_dict()
        
        contracts_list = contracts_df.drop(columns=shared_enriched.keys()).to_dict('records')  # Drop duplicates
        
        data_dict = {
            "ticker": ticker,
            "company_name": contracts_df['company_name'].iloc[0],
            "contracts": contracts_list,
            "enriched_data": shared_enriched
        }
        
        contract_json_str = json.dumps(data_dict, indent=2)

        prompt = _PROMPT_TEMPLATE.format(
            example_output=_EXAMPLE_OUTPUT,
            contract_data=contract_json_str
        )
        
        analysis_json_str = vertex_ai.generate(prompt)

        # Validate JSON
        json.loads(analysis_json_str)
        
        gcs.write_text(config.GCS_BUCKET_NAME, analysis_blob_path, analysis_json_str, "application/json")
        
        logging.info(f"[{ticker}] Successfully generated and wrote best analysis to {analysis_blob_path}")
        return analysis_blob_path

    except Exception as e:
        logging.error(f"[{ticker}] Failed during best option analysis generation: {e}", exc_info=True)
        return None

def run_pipeline():
    """Main pipeline for generating per-ticker best-option analysis."""
    logging.info("--- Starting Options Analysis Pipeline ---")
    
    grouped_work = _get_options_work_list()
    if not grouped_work:
        logging.warning("No options candidates found to analyze. Exiting.")
        return

    processed_count = 0
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(_process_ticker, ticker, df): ticker for ticker, df in grouped_work.items()}
        for future in as_completed(futures):
            if future.result():
                processed_count += 1
                
    logging.info(f"--- Options Analysis Pipeline Finished. Processed {processed_count} of {len(grouped_work)} tickers. ---")