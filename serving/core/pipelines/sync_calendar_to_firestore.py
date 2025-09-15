# serving/core/pipelines/sync_calendar_to_firestore.py
import logging
import pandas as pd
from google.cloud import firestore, bigquery
from .. import config
import numpy as np
import uuid
import re

# --------- Tunables ----------
BATCH_SIZE = 500
CALENDAR_TABLE_ID = f"{config.SOURCE_PROJECT_ID}.{config.BIGQUERY_DATASET}.calendar_events"
FIRESTORE_COLLECTION_NAME = "calendar_events"

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
        ops_to_delete = [{"type": "delete", "ref": d.reference} for d in docs]
        _commit_ops(firestore.Client(project=config.DESTINATION_PROJECT_ID), ops_to_delete)
        deleted_count += len(ops_to_delete)
        logging.info(f"Deleted {deleted_count} docs...")
    logging.info(f"Wipe complete for collection '{collection_ref.id}'.")

def _load_bq_df(bq: bigquery.Client, query: str) -> pd.DataFrame:
    """Loads data from a BigQuery query into a pandas DataFrame and cleans it."""
    df = bq.query(query).to_dataframe()
    if not df.empty:
        for col in df.columns:
            dtype_str = str(df[col].dtype)
            if dtype_str.startswith("datetime64") or "datetimetz" in dtype_str or dtype_str == "dbdate":
                df[col] = df[col].astype(str)
        
        df = df.replace({pd.NA: np.nan})
        df = df.where(pd.notna(df), None)
    return df

def run_pipeline(full_reset: bool = False):
    """
    Syncs the entire calendar_events table from BigQuery to a Firestore collection.
    """
    db = firestore.Client(project=config.DESTINATION_PROJECT_ID)
    bq = bigquery.Client(project=config.SOURCE_PROJECT_ID)
    
    collection_ref = db.collection(FIRESTORE_COLLECTION_NAME)
    logging.info(f"--- Calendar Events Firestore Sync Pipeline ---")
    logging.info(f"Target collection: {collection_ref.id}")
    logging.info(f"Full reset? {'YES' if full_reset else 'NO'}")

    try:
        # Fetch all future events
        calendar_query = f"SELECT * FROM `{CALENDAR_TABLE_ID}` WHERE event_date >= CURRENT_DATE()"
        calendar_df = _load_bq_df(bq, calendar_query)
    except Exception as e:
        logging.critical(f"Failed to query calendar events from BigQuery: {e}", exc_info=True)
        raise

    # Always wipe the collection to ensure no stale events remain
    _delete_collection_in_batches(collection_ref)

    if calendar_df.empty:
        logging.warning("No upcoming calendar events found in BigQuery. Collection will be empty.")
        return

    # Use ticker as the primary key, fallback to event_type
    calendar_df["collection_id"] = np.where(pd.notna(calendar_df["ticker"]), calendar_df["ticker"], calendar_df["event_type"])
    
    # Sanitize the collection_id to remove characters that are invalid in Firestore document IDs
    calendar_df["collection_id"] = calendar_df["collection_id"].str.replace(r'[/]', '_', regex=True)


    upsert_ops = []
    for collection_id, group in calendar_df.groupby("collection_id"):
        parent_doc_ref = collection_ref.document(collection_id)
        
        # Create a subcollection for the events
        events_collection_ref = parent_doc_ref.collection("events")
        
        for _, row in group.iterrows():
            # Generate a unique ID for each event document in the subcollection
            event_doc_id = str(uuid.uuid4())
            doc_ref = events_collection_ref.document(event_doc_id)
            
            # The data for this document is just the event row
            event_data = row.to_dict()
            upsert_ops.append({"type": "set", "ref": doc_ref, "data": event_data})
    
    logging.info(f"Upserting {len(upsert_ops)} documents to '{collection_ref.id}'...")
    for chunk in _iter_batches(upsert_ops, BATCH_SIZE):
        _commit_ops(db, chunk)
            
    logging.info(f"Sync complete for '{collection_ref.id}'.")
    logging.info("--- Calendar Events Firestore Sync Pipeline Finished ---")