# enrichment/core/config.py
import os

# --- Global Project ---
PROJECT_ID = os.getenv("PROJECT_ID", "profitscout-lx6bb")
BIGQUERY_DATASET = os.getenv("BIGQUERY_DATASET", "profit_scout")

# --- BigQuery ---
SCORES_TABLE = f"{PROJECT_ID}.{BIGQUERY_DATASET}.analysis_scores"
CHAIN_TABLE  = f"{PROJECT_ID}.{BIGQUERY_DATASET}.options_chain"
CAND_TABLE   = f"{PROJECT_ID}.{BIGQUERY_DATASET}.options_candidates"