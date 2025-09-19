# serving/core/bq.py
import logging
import pandas as pd
from google.cloud import bigquery
import time

def load_df_to_bq(df: pd.DataFrame, table_id: str, project_id: str, write_disposition: str = "WRITE_TRUNCATE"):
    """
    Loads a pandas DataFrame into a BigQuery table using simple APPEND or TRUNCATE.
    """
    if df.empty:
        logging.warning("DataFrame is empty. Skipping BigQuery load.")
        return
    
    client = bigquery.Client(project=project_id)
    
    # Conditionally add schema_update_options ONLY for WRITE_APPEND
    job_config = bigquery.LoadJobConfig(
        write_disposition=write_disposition,
    )
    if write_disposition == "WRITE_APPEND":
        job_config.schema_update_options = [
            bigquery.SchemaUpdateOption.ALLOW_FIELD_ADDITION
        ]
    
    try:
        job = client.load_table_from_dataframe(df, table_id, job_config=job_config)
        job.result()
        logging.info(f"Loaded {job.output_rows} rows into BigQuery table: {table_id} using {write_disposition}")
    except Exception as e:
        logging.error(f"Failed to load DataFrame to {table_id}: {e}", exc_info=True)
        raise

def upsert_df_to_bq(df: pd.DataFrame, table_id: str, project_id: str):
    """
    Upserts a DataFrame into a BigQuery table using a MERGE statement.
    """
    if df.empty:
        logging.warning("DataFrame is empty. Skipping BigQuery MERGE operation.")
        return

    client = bigquery.Client(project=project_id)
    
    dataset_id = table_id.split('.')[-2]
    final_table_name = table_id.split('.')[-1]
    
    temp_table_name = f"{final_table_name}_temp_{int(time.time())}"
    temp_table_id = f"{project_id}.{dataset_id}.{temp_table_name}"

    job_config = bigquery.LoadJobConfig(write_disposition="WRITE_TRUNCATE")
    try:
        load_job = client.load_table_from_dataframe(df, temp_table_id, job_config=job_config)
        load_job.result()
    except Exception as e:
        logging.error(f"Failed to load DataFrame to temp table {temp_table_id}: {e}", exc_info=True)
        raise

    cols_to_insert = ", ".join([f"`{col}`" for col in df.columns])
    cols_to_update = ", ".join([f"T.`{col}` = S.`{col}`" for col in df.columns if col != 'ticker'])
    
    merge_sql = f"""
    MERGE `{table_id}` T
    USING `{temp_table_id}` S ON T.ticker = S.ticker
    WHEN MATCHED THEN
      UPDATE SET {cols_to_update}
    WHEN NOT MATCHED THEN
      INSERT ({cols_to_insert}) VALUES ({cols_to_insert})
    """

    try:
        logging.info(f"Executing MERGE to upsert data into {table_id}...")
        merge_job = client.query(merge_sql)
        merge_job.result()
        logging.info(f"MERGE complete. {merge_job.num_dml_affected_rows} rows affected in {table_id}.")
    except Exception as e:
        logging.error(f"Failed to execute MERGE on {table_id}: {e}", exc_info=True)
        raise
    finally:
        client.delete_table(temp_table_id, not_found_ok=True)