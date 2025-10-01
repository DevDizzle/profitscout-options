# serving/core/pipelines/sync_to_firestore.py
import logging
import re
from urllib.parse import urlparse
import pandas as pd
from google.cloud import firestore, bigquery, storage
from .. import config
import numpy as np

bq_client = bigquery.Client(project=config.SOURCE_PROJECT_ID) 

# --------- Tunables ----------
BATCH_SIZE = 500
PRIMARY_KEY_FIELD = "ticker"           # Firestore doc id
URI_FIELDS = ["uri", "image_uri", "pdf_uri"]  # Columns to validate (if using GCS)
VALIDATE_GCS_LINKS = False             # Set True to check GCS object existence
# ------------------------------

_GCS_URI_RE = re.compile(r"^gs://([^/]+)/(.+)$")

def _iter_batches(iterable, n):
    batch = []
    for item in iterable:
        batch.append(item)
        if len(batch) >= n:
            yield batch
            batch = []
    if batch:
        yield batch

def _is_gcs_uri(s: str) -> bool:
    if not s or not isinstance(s, str):
        return False
    if s.startswith("gs://"):
        return True
    try:
        u = urlparse(s)
        return ("storage.googleapis.com" in (u.netloc or "")) and len(u.path.split("/")) >= 3
    except Exception:
        return False

def _gcs_blob_from_any(storage_client: storage.Client, uri: str):
    if uri.startswith("gs://"):
        m = _GCS_URI_RE.match(uri)
        if not m:
            return None
        bucket_name, blob_name = m.group(1), m.group(2)
    else:
        u = urlparse(uri)
        parts = [p for p in (u.path or "").split("/") if p]
        if len(parts) < 2:
            return None
        bucket_name = parts[0]
        blob_name = "/".join(parts[1:])
    bucket = storage_client.bucket(bucket_name)
    return bucket_name, blob_name, bucket.blob(blob_name)

def _validate_gcs_links(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or not VALIDATE_GCS_LINKS:
        return df
    present_uri_cols = [c for c in URI_FIELDS if c in df.columns]
    if not present_uri_cols:
        return df

    storage_client = storage.Client(project=config.DESTINATION_PROJECT_ID)

    def row_available(row) -> bool:
        try:
            found_any = False
            for col in present_uri_cols:
                val = row.get(col)
                if not val:
                    return False
                if _is_gcs_uri(val):
                    found_any = True
                    tup = _gcs_blob_from_any(storage_client, val)
                    if not tup:
                        return False
                    _, _, blob = tup
                    if not blob.exists():
                        return False
            return True if found_any else True
        except Exception:
            return False

    df = df.copy()
    df["is_available"] = df.apply(row_available, axis=1)
    return df

def _commit_ops(db, ops):
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
    logging.info(f"Wiping Firestore collection: '{collection_ref.id}'...")
    while True:
        docs = list(collection_ref.limit(BATCH_SIZE).stream())
        if not docs:
            break
        ops = [{"type": "delete", "ref": d.reference} for d in docs]
        _commit_ops(firestore.Client(project=config.DESTINATION_PROJECT_ID), ops)
        logging.info(f"Deleted {len(ops)} docs...")
    logging.info("Wipe complete.")

def _load_bq_df(bq: bigquery.Client) -> pd.DataFrame:
    """
    Loads data from BigQuery and cleans it for Firestore.
    """
    query = f"""
      SELECT *
      FROM `{config.SYNC_FIRESTORE_TABLE_ID}`
      WHERE weighted_score IS NOT NULL
    """
    df = bq.query(query).to_dataframe()

    if not df.empty:
        logging.info("--- Raw DataFrame from BigQuery ---")
        # Use an in-memory string stream to capture the output of df.info()
        from io import StringIO
        buffer = StringIO()
        df.info(buf=buffer)
        logging.info(buffer.getvalue())
        logging.info(df.head())

        # Convert date-like columns to strings
        for col in df.columns:
            dtype_str = str(df[col].dtype)
            if dtype_str.startswith("datetime64") or "datetimetz" in dtype_str or dtype_str == "dbdate":
                df[col] = df[col].astype(str)

        # Explicitly handle numeric types, coercing errors to NaN
        for col in ['weighted_score', 'price', 'thirty_day_change_pct']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # Convert all pandas NA types and numpy NaN to None for Firestore
        df = df.replace({pd.NA: np.nan}).where(pd.notna(df), None)

        logging.info("--- Cleaned DataFrame for Firestore ---")
        buffer = StringIO()
        df.info(buf=buffer)
        logging.info(buffer.getvalue())
        logging.info(df.head())

    return df


def run_pipeline():
    """
    Wipes the entire Firestore collection and repopulates it from the
    BigQuery source table.
    """
    db = firestore.Client(project=config.DESTINATION_PROJECT_ID)
    bq = bigquery.Client(project=config.DESTINATION_PROJECT_ID)
    collection_ref = db.collection(config.FIRESTORE_COLLECTION)

    logging.info("--- Firestore Sync Pipeline (Full Reset Mode) ---")
    logging.info(f"Target collection: {config.FIRESTORE_COLLECTION}")

    try:
        df = _load_bq_df(bq)
    except Exception as e:
        logging.critical(f"Failed to query BigQuery: {e}", exc_info=True)
        raise

    # 1. Always delete the entire collection first
    _delete_collection_in_batches(collection_ref)

    if df.empty:
        logging.info("BigQuery returned 0 rows. Collection is now empty.")
        return

    if PRIMARY_KEY_FIELD not in df.columns:
        raise ValueError(f"Expected primary key column '{PRIMARY_KEY_FIELD}'")

    # 2. Prepare all documents for insertion
    upsert_ops = []
    for _, row in df.iterrows():
        key = str(row[PRIMARY_KEY_FIELD])
        doc_ref = collection_ref.document(key)
        # Convert row to dict here to handle any remaining numpy types
        upsert_ops.append({"type": "set", "ref": doc_ref, "data": row.to_dict()})

    # 3. Commit the new documents in batches
    logging.info(f"Populating collection with {len(upsert_ops)} documents...")
    for chunk in _iter_batches(upsert_ops, BATCH_SIZE):
        _commit_ops(db, chunk)

    logging.info(f"✅ Sync complete. Wrote {len(upsert_ops)} documents.")
    return