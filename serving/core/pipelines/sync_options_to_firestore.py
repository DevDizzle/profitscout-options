# serving/core/pipelines/sync_options_to_firestore.py
import logging
import pandas as pd
from google.cloud import firestore, bigquery
from .. import config
import numpy as np

# --------- Tunables ----------
BATCH_SIZE = 500
PRIMARY_KEY_FIELD = "contract_symbol" # Using the unique contract symbol for the document ID
OPTIONS_TABLE_ID = f"{config.SOURCE_PROJECT_ID}.{config.BIGQUERY_DATASET}.options_candidates"
FIRESTORE_COLLECTION_NAME = "options_candidates"

def _iter_batches(iterable, n):
    """Yield successive n-sized chunks from iterable."""
    batch = []
    for item in iterable:
        batch.append(item)
        if len(batch) >= n:
            yield batch
            batch = []
    if batch:
        yield batch

def _commit_ops(db, ops):
    """Commits a list of Firestore operations in batches."""
    batch = db.batch()
    count = 0
    for op in ops:
        if op["type"] == "set":
            batch.set(op["ref"], op["data"])
        elif op["type"] == "delete":
            batch.delete(op["ref"])
        count += 1
        if count >= BATCH_SIZE:
            batch.commit()
            batch = db.batch()
            count = 0
    if count:
        batch.commit()

def _delete_collection_in_batches(collection_ref):
    """Wipes all documents from a Firestore collection."""
    logging.info(f"Wiping Firestore collection: '{collection_ref.id}'...")
    deleted_count = 0
    while True:
        docs = list(collection_ref.limit(BATCH_SIZE).stream())
        if not docs:
            break
        ops = [{"type": "delete", "ref": d.reference} for d in docs]
        _commit_ops(firestore.Client(project=config.DESTINATION_PROJECT_ID), ops)
        deleted_count += len(ops)
        logging.info(f"Deleted {deleted_count} docs...")
    logging.info(f"Wipe complete for collection '{collection_ref.id}'.")

def _load_bq_df(bq: bigquery.Client, query: str) -> pd.DataFrame:
    """Loads data from a BigQuery query into a pandas DataFrame and cleans it."""
    df = bq.query(query).to_dataframe()
    if not df.empty:
        # Convert date/time columns to string
        for col in df.columns:
            dtype_str = str(df[col].dtype)
            if dtype_str.startswith("datetime64") or "datetimetz" in dtype_str or dtype_str == "dbdate":
                df[col] = df[col].astype(str)
        
        df = df.replace({pd.NA: np.nan})
        df = df.where(pd.notna(df), None)
    return df

def run_pipeline(full_reset: bool = False):
    """
    Syncs the entire options_candidates table from BigQuery to a Firestore collection.
    """
    db = firestore.Client(project=config.DESTINATION_PROJECT_ID)
    # Query from the source project
    bq = bigquery.Client(project=config.SOURCE_PROJECT_ID)
    
    collection_ref = db.collection(FIRESTORE_COLLECTION_NAME)
    logging.info(f"--- Options Firestore Sync Pipeline ---")
    logging.info(f"Target collection: {collection_ref.id}")
    logging.info(f"Full reset? {'YES' if full_reset else 'NO'}")

    try:
        # Query includes joining with stock_metadata to get company_name
        options_query = f"""
            SELECT
                t1.*,
                t2.company_name
            FROM `{OPTIONS_TABLE_ID}` AS t1
            LEFT JOIN (
                SELECT ticker, company_name, ROW_NUMBER() OVER(PARTITION BY ticker ORDER BY quarter_end_date DESC) as rn
                FROM `{config.SOURCE_PROJECT_ID}.{config.BIGQUERY_DATASET}.stock_metadata`
            ) AS t2 ON t1.ticker = t2.ticker AND t2.rn = 1
        """
        options_df = _load_bq_df(bq, options_query)
    except Exception as e:
        logging.critical(f"Failed to query options candidates from BigQuery: {e}", exc_info=True)
        raise

    if full_reset:
        _delete_collection_in_batches(collection_ref)

    if options_df.empty:
        logging.warning("No options candidates found in BigQuery. Collection will be empty or unchanged.")
        return

    if PRIMARY_KEY_FIELD not in options_df.columns:
        raise ValueError(f"Expected primary key column '{PRIMARY_KEY_FIELD}' in options_candidates table")

    upsert_ops = []
    for _, row in options_df.iterrows():
        key = str(row[PRIMARY_KEY_FIELD])
        doc_ref = collection_ref.document(key)
        upsert_ops.append({"type": "set", "ref": doc_ref, "data": row.to_dict()})
    
    logging.info(f"Upserting {len(upsert_ops)} documents to '{collection_ref.id}'...")
    for chunk in _iter_batches(upsert_ops, BATCH_SIZE):
        _commit_ops(db, chunk)
    
    current_keys = set(str(x) for x in options_df[PRIMARY_KEY_FIELD].tolist())
    existing_keys = [doc.id for doc in collection_ref.stream()]
    to_delete = [k for k in existing_keys if k not in current_keys]

    if to_delete:
        logging.info(f"Deleting {len(to_delete)} stale documents from '{collection_ref.id}'...")
        delete_ops = [{"type": "delete", "ref": collection_ref.document(k)} for k in to_delete]
        for chunk in _iter_batches(delete_ops, BATCH_SIZE):
            _commit_ops(db, chunk)
            
    logging.info(f"Sync complete for '{collection_ref.id}'.")
    logging.info("--- Options Firestore Sync Pipeline Finished ---")