# serving/core/pipelines/sync_calendar_to_firestore.py
import logging
import pandas as pd
from google.cloud import firestore, bigquery
from .. import config
import numpy as np
import re

# --------- Tunables ----------
BATCH_SIZE = 500
CALENDAR_TABLE_ID = f"{config.SOURCE_PROJECT_ID}.{config.BIGQUERY_DATASET}.calendar_events"
FIRESTORE_COLLECTION_NAME = "calendar_events"

# Firestore document ID cannot contain '/' and should be non-empty & reasonably short.
_ID_SANITIZE_RE = re.compile(r"[^\w\-\.:@]+")  # allow [A-Za-z0-9_] plus - . : @

def _sanitize_id(s: str, fallback: str = "UNKNOWN") -> str:
    if not s:
        return fallback
    # Replace disallowed chars with '_', then trim length if needed.
    cleaned = _ID_SANITIZE_RE.sub("_", s)
    cleaned = cleaned.strip("._-") or fallback
    # Firestore hard limit is ~1500 bytes; we keep it modest.
    return cleaned[:200]

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

def _commit_ops(db: firestore.Client, ops):
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

def _delete_collection_in_batches(db: firestore.Client, collection_ref):
    """Wipes all documents from a Firestore collection (including subcollections)."""
    logging.info(f"Wiping Firestore collection: '{collection_ref.id}'...")
    deleted_count = 0
    while True:
        docs = list(collection_ref.limit(BATCH_SIZE).stream())
        if not docs:
            break
        ops_to_delete = [{"type": "delete", "ref": d.reference} for d in docs]
        _commit_ops(db, ops_to_delete)
        deleted_count += len(ops_to_delete)
        logging.info(f"Deleted {deleted_count} docs...")
    logging.info(f"Wipe complete for collection '{collection_ref.id}'.")

def _load_bq_df(bq: bigquery.Client, query: str) -> pd.DataFrame:
    """Loads data from a BigQuery query into a pandas DataFrame and cleans it."""
    df = bq.query(query).to_dataframe()
    if not df.empty:
        # Convert datetimes/dates to strings for Firestore serialization
        for col in df.columns:
            dtype_str = str(df[col].dtype)
            if dtype_str.startswith("datetime64") or "datetimetz" in dtype_str or dtype_str == "dbdate":
                df[col] = df[col].astype(str)
        df = df.replace({pd.NA: np.nan})
        df = df.where(pd.notna(df), None)
    return df

def run_pipeline(full_reset: bool = False):
    """
    Syncs the rolling 90-day forward calendar from BigQuery to Firestore.

    BigQuery minimal schema expected:
      event_id (STRING, PK)
      entity (STRING, nullable)
      event_type (STRING)
      event_name (STRING)
      event_date (DATE)
      event_time (TIMESTAMP, nullable)
      source (STRING)
      last_seen (TIMESTAMP)
    """
    db = firestore.Client(project=config.DESTINATION_PROJECT_ID)
    bq = bigquery.Client(project=config.SOURCE_PROJECT_ID)

    collection_ref = db.collection(FIRESTORE_COLLECTION_NAME)
    logging.info(f"--- Calendar Events Firestore Sync Pipeline ---")
    logging.info(f"Target collection: {collection_ref.id}")
    logging.info(f"Full reset? {'YES' if full_reset else 'NO'}")

    try:
        # Only future events (rolling forward view)
        calendar_query = f"""
        SELECT event_id, entity, event_type, event_name, event_date, event_time, source, last_seen
        FROM `{CALENDAR_TABLE_ID}`
        WHERE event_date >= CURRENT_DATE()
        """
        calendar_df = _load_bq_df(bq, calendar_query)
    except Exception as e:
        logging.critical(f"Failed to query calendar events from BigQuery: {e}", exc_info=True)
        raise

    # For simplicity and to avoid stale docs, wipe each run (rolling 90-day window).
    # If you later move to incremental syncs, you can remove this wipe—doc IDs are stable.
    _delete_collection_in_batches(db, collection_ref)

    if calendar_df.empty:
        logging.warning("No upcoming calendar events found in BigQuery. Collection will be empty.")
        logging.info("--- Calendar Events Firestore Sync Pipeline Finished ---")
        return

    # Compute collection_id:
    # - Prefer 'entity' (ticker); for economic events where entity is NULL, use event_type ("Economic").
    # - Sanitize to be a safe Firestore document ID.
    collection_ids = []
    for _, row in calendar_df.iterrows():
        raw_id = row.get("entity") or row.get("event_type") or "UNKNOWN"
        collection_ids.append(_sanitize_id(str(raw_id), fallback="UNKNOWN"))
    calendar_df["collection_id"] = collection_ids

    # Prepare batched upserts
    upsert_ops = []
    # Group by collection (e.g., AAPL or Economic)
    for collection_id, group in calendar_df.groupby("collection_id"):
        parent_doc_ref = collection_ref.document(collection_id)
        events_collection_ref = parent_doc_ref.collection("events")

        for _, row in group.iterrows():
            # Use stable event_id as the subdocument ID so reruns overwrite cleanly
            event_doc_id = _sanitize_id(str(row["event_id"]), fallback=None)
            if not event_doc_id:
                # Shouldn't happen, but skip if event_id missing/unusable
                continue

            doc_ref = events_collection_ref.document(event_doc_id)

            # Convert the row to a serializable dict
            event_data = {
                "event_id": row.get("event_id"),
                "entity": row.get("entity"),
                "event_type": row.get("event_type"),
                "event_name": row.get("event_name"),
                "event_date": row.get("event_date"),   # already string from _load_bq_df
                "event_time": row.get("event_time"),   # string or None
                "source": row.get("source"),
                "last_seen": row.get("last_seen"),
                # Helpful for the app:
                "collection_id": collection_id,
            }

            upsert_ops.append({"type": "set", "ref": doc_ref, "data": event_data})

    logging.info(f"Upserting {len(upsert_ops)} documents to '{collection_ref.id}'...")
    for chunk in _iter_batches(upsert_ops, BATCH_SIZE):
        _commit_ops(db, chunk)

    logging.info(f"Sync complete for '{collection_ref.id}'.")
    logging.info("--- Calendar Events Firestore Sync Pipeline Finished ---")
