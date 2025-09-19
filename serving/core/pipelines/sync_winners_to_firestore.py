# serving/core/pipelines/sync_winners_to_firestore.py
import logging
import pandas as pd
from google.cloud import firestore, bigquery
from .. import config
import numpy as np

# --- Configuration ---
BATCH_SIZE = 500
FIRESTORE_COLLECTION_NAME = "winners_dashboard"
WINNERS_TABLE_ID = f"{config.DESTINATION_PROJECT_ID}.{config.BIGQUERY_DATASET}.winners_dashboard"

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
        batch = collection_ref.firestore.batch()
        for doc in docs:
            batch.delete(doc.reference)
        batch.commit()
        deleted_count += len(docs)
        logging.info(f"Deleted {deleted_count} docs...")
    logging.info(f"Wipe complete for collection '{collection_ref.id}'.")

def _load_bq_df(bq: bigquery.Client) -> pd.DataFrame:
    """Loads the winners dashboard data and prepares it for Firestore."""
    query = f"SELECT * FROM `{WINNERS_TABLE_ID}`"
    df = bq.query(query).to_dataframe()
    if not df.empty:
        # Convert date/time columns to string for Firestore
        for col in df.columns:
            dtype_str = str(df[col].dtype)
            if dtype_str.startswith("datetime64") or "datetimetz" in dtype_str or "dbdate" in dtype_str:
                df[col] = df[col].astype(str)
        
        df = df.replace({pd.NA: np.nan})
        df = df.where(pd.notna(df), None)
    return df

def run_pipeline():
    """
    Syncs the winners_dashboard table from BigQuery to a Firestore collection.
    This pipeline performs a full wipe-and-reload each time to ensure freshness.
    """
    db = firestore.Client(project=config.DESTINATION_PROJECT_ID)
    bq = bigquery.Client(project=config.DESTINATION_PROJECT_ID)
    
    collection_ref = db.collection(FIRESTORE_COLLECTION_NAME)
    logging.info(f"--- Winners Dashboard Firestore Sync Pipeline ---")
    logging.info(f"Target collection: {collection_ref.id}")

    # For a daily dashboard, wiping each time is the cleanest approach.
    _delete_collection_in_batches(collection_ref)

    try:
        winners_df = _load_bq_df(bq)
    except Exception as e:
        logging.critical(f"Failed to query winners dashboard table from BigQuery: {e}", exc_info=True)
        raise

    if winners_df.empty:
        logging.warning("No winners found in BigQuery. Firestore collection will be empty.")
        logging.info("--- Winners Dashboard Firestore Sync Pipeline Finished ---")
        return

    upsert_ops = []
    for _, row in winners_df.iterrows():
        # Use the ticker as the unique document ID in Firestore
        doc_ref = collection_ref.document(row["ticker"])
        upsert_ops.append({"type": "set", "ref": doc_ref, "data": row.to_dict()})
    
    logging.info(f"Upserting {len(upsert_ops)} winner documents to '{collection_ref.id}'...")
    for chunk in _iter_batches(upsert_ops, BATCH_SIZE):
        _commit_ops(db, chunk)
    
    logging.info(f"Sync complete. Wrote {len(winners_df)} documents.")
    logging.info("--- Winners Dashboard Firestore Sync Pipeline Finished ---")