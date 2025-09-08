import logging
import pandas as pd
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from google.cloud import bigquery
from .. import config, gcs
from ..clients import vertex_ai

OUTPUT_PREFIX = "options-analysis/"

# --- Example generated from the data you provided ---
_EXAMPLE_OUTPUT = """{
  "score": 0.90,
  "analysis": "This Middleby Corp (MIDD) put option presents a compelling bearish opportunity. The contract is approximately 5.0% out-of-the-money with a near-term expiration, offering significant leverage if the underlying stock continues its downward trend. Liquidity is exceptionally strong, with over 3,000 contracts in open interest, ensuring ease of trading. The option's theta decay is manageable, and its delta indicates a clear, but not overly aggressive, directional bet. Given the high options score and SELL signal on the underlying, this contract is well-positioned to capitalize on a price decline."
}"""

_PROMPT_TEMPLATE = r"""
You are a sharp financial analyst specializing in equity options. Your task is to provide a concise, data-driven analysis for a specific options contract.
Use **only** the JSON data provided.

### Key Interpretation Guidelines
1.  **Signal Alignment**: Does the option type (CALL/PUT) align with the underlying stock's signal (BUY/SELL)?
2.  **Moneyness & Leverage**: How far out-of-the-money is the contract? Does this offer good leverage?
3.  **Liquidity**: Is the open interest high enough for easy trading?
4.  **Risk Factors**: How significant is the theta (time decay)? Is implied volatility high or low?
5.  **Overall Score**: Synthesize the data points and the `options_score` into a final assessment.

### Step-by-Step Reasoning
1.  Evaluate the contract's characteristics (DTE, moneyness, liquidity, greeks) in the context of the underlying `signal`.
2.  Convert the `options_score` (0.0 to 1.0) into a qualitative strength (e.g., weak, moderate, strong, exceptional).
3.  Summarize the key pros and cons into a dense paragraph.

### Output — return exactly this JSON, nothing else
{
  "score": <float between 0 and 1, same as options_score input>,
  "analysis": "<One dense paragraph (100-150 words) summarizing the trade idea, highlighting liquidity, risk, and alignment with the underlying signal.>"
}

### Example Output (for format only; do not copy wording)
{example_output}

### Provided data:
{contract_data}
"""

def _get_options_work_list() -> pd.DataFrame:
    """
    Fetches all of today's options candidates from BigQuery.
    """
    client = bigquery.Client(project=config.PROJECT_ID)
    # This query now fetches from the CAND_TABLE defined in the enrichment config
    query = f"""
        SELECT
            c.*,
            s.company_name
        FROM `{config.CAND_TABLE}` AS c
        JOIN (
            SELECT ticker, company_name, ROW_NUMBER() OVER(PARTITION BY ticker ORDER BY quarter_end_date DESC) as rn
            FROM `{config.PROJECT_ID}.{config.BIGQUERY_DATASET}.stock_metadata`
        ) AS s ON c.ticker = s.ticker AND s.rn = 1
        WHERE c.selection_run_ts >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 24 HOUR)
          AND c.options_score >= 0.5
        ORDER BY c.ticker, c.options_score DESC
    """
    try:
        df = client.query(query).to_dataframe()
        # Convert timestamp to string for JSON serialization
        if 'selection_run_ts' in df.columns:
            df['selection_run_ts'] = df['selection_run_ts'].astype(str)
        if 'expiration_date' in df.columns:
            df['expiration_date'] = df['expiration_date'].astype(str)
        if 'fetch_date' in df.columns:
            df['fetch_date'] = df['fetch_date'].astype(str)
        return df
    except Exception as e:
        logging.critical(f"Failed to fetch options work list: {e}", exc_info=True)
        return pd.DataFrame()


def _process_contract(contract_data: pd.Series):
    """
    Main worker function for a single option contract.
    Generates one JSON analysis file for the contract.
    """
    ticker = contract_data.get('ticker', 'UNKNOWN')
    contract_symbol = contract_data.get('contract_symbol', 'UNKNOWN')
    
    analysis_blob_path = f"{OUTPUT_PREFIX}{ticker}_{contract_symbol}_analysis.json"
    logging.info(f"[{ticker}] Generating options analysis for {contract_symbol}")

    try:
        # Convert Series to a JSON string for the prompt
        contract_json_str = contract_data.to_json(indent=2)

        prompt = _PROMPT_TEMPLATE.format(
            example_output=_EXAMPLE_OUTPUT,
            contract_data=contract_json_str
        )
        
        analysis_json_str = vertex_ai.generate(prompt)

        # Basic validation to ensure it's a JSON object
        json.loads(analysis_json_str)
        
        gcs.write_text(config.GCS_BUCKET_NAME, analysis_blob_path, analysis_json_str, "application/json")
        
        logging.info(f"[{ticker}] Successfully generated and wrote analysis to {analysis_blob_path}")
        return analysis_blob_path

    except Exception as e:
        logging.error(f"[{ticker}] Failed during options analysis generation for {contract_symbol}: {e}", exc_info=True)
        return None

def run_pipeline():
    """Main pipeline for generating single-contract options analysis."""
    logging.info("--- Starting Options Analysis Pipeline ---")
    
    work_df = _get_options_work_list()
    if work_df.empty:
        logging.warning("No options candidates found to analyze. Exiting.")
        return

    work_items = [row for _, row in work_df.iterrows()]
    
    processed_count = 0
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(_process_contract, item): item.get('contract_symbol') for item in work_items}
        for future in as_completed(futures):
            if future.result():
                processed_count += 1
                
    logging.info(f"--- Options Analysis Pipeline Finished. Processed {processed_count} of {len(work_items)} contracts. ---")