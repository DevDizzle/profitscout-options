# serving/core/pipelines/sync_winners_to_firestore.py
import logging
import pandas as pd
from google.cloud import firestore, bigquery
from .. import config
import numpy as np

# --- Configuration ---
BATCH_SIZE = 500
FIRESTORE_COLLECTION_NAME = "winners_dashboard"
# Updated to use the specific table ID you provided.
WINNERS_TABLE_ID = "profitscout-lx6bb.profit_scout.winners_dashboard"

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

def _delete_collection_in_batches(collection_ref: firestore.CollectionReference):
    """Wipes all documents from a Firestore collection in batches."""
    logging.info(f"Wiping Firestore collection: '{collection_ref.id}'...")
    deleted_count = 0
    while True:
        # Get a batch of documents to delete.
        docs = list(collection_ref.limit(BATCH_SIZE).stream())
        if not docs:
            # No documents left, so we're done.
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
    logging.info(f"Querying BigQuery table: {WINNERS_TABLE_ID}")
    query = f"SELECT * FROM `{WINNERS_TABLE_ID}`"
    df = bq.query(query).to_dataframe()

    if not df.empty:
        # Convert date/time columns to string for Firestore compatibility.
        for col in df.columns:
            dtype_str = str(df[col].dtype)
            if "datetime" in dtype_str or "dbdate" in dtype_str:
                df[col] = df[col].astype(str)

        # Replace pandas missing values (NA) with None, which Firestore handles correctly.
        df = df.replace({pd.NA: np.nan})
        df = df.where(pd.notna(df), None)
    return df

def run_pipeline():
    """
    Syncs the winners_dashboard table from BigQuery to a Firestore collection.

    This pipeline performs a full wipe-and-reload each time to ensure the
    Firestore collection is an exact mirror of the BigQuery table.
    """
    # Initialize clients for their specific roles.
    # Firestore client points to the DESTINATION project where data will be written.
    db = firestore.Client(project=config.DESTINATION_PROJECT_ID)
    # BigQuery client points to the SOURCE project where data is read from.
    bq = bigquery.Client(project=config.SOURCE_PROJECT_ID)

    collection_ref = db.collection(FIRESTORE_COLLECTION_NAME)
    logging.info("--- Winners Dashboard Firestore Sync Pipeline ---")
    logging.info(f"Source BQ Table: {WINNERS_TABLE_ID}")
    logging.info(f"Destination Firestore Collection: {collection_ref.path}")

    # For a daily dashboard, wiping each time is the cleanest approach.
    _delete_collection_in_batches(collection_ref)

    try:
        winners_df = _load_bq_df(bq)
    except Exception as e:
        logging.critical(f"Failed to query winners table from BigQuery: {e}", exc_info=True)
        raise

    if winners_df.empty:
        logging.warning("No winners found in BigQuery. Firestore collection will be empty.")
        logging.info("--- Winners Dashboard Firestore Sync Pipeline Finished ---")
        return

    logging.info(f"Upserting {len(winners_df)} documents to '{collection_ref.id}'...")

    # --- Refactored Write Logic ---
    # Iterate through the dataframe in chunks and commit each chunk as a batch.
    # This is more memory-efficient than creating a giant list of operations first.
    total_written = 0
    for batch_rows in _iter_batches(winners_df.iterrows(), BATCH_SIZE):
        batch = db.batch()
        for _, row in batch_rows:
            # Ensure the ticker is a string, as Firestore document IDs must be strings.
            doc_id = str(row["ticker"])
            doc_ref = collection_ref.document(doc_id)
            batch.set(doc_ref, row.to_dict())

        batch.commit()
        total_written += len(batch_rows)
        logging.info(f"Wrote {total_written}/{len(winners_df)} documents...")

    logging.info(f"Sync complete. Total documents written: {total_written}.")
    logging.info("--- Winners Dashboard Firestore Sync Pipeline Finished ---")