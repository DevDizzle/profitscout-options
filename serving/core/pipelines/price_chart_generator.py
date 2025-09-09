# serving/core/pipelines/price_chart_generator.py
import logging
import pandas as pd
import numpy as np
import io
import os
import json
from datetime import date
from concurrent.futures import ThreadPoolExecutor, as_completed
from queue import Queue
import threading

from google.cloud import bigquery

# Matplotlib: use non-interactive backend and avoid global state in threads
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from .. import config, gcs

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
PRICE_CHART_FOLDER = "90-Day-Chart/"
TICKER_LIST_PATH = "tickerlist.txt"

# Matplotlib is not thread-safe; keep a lock (harmless redundancy with single plotter)
_PLOT_LOCK = threading.Lock()

# Single-threaded plot queue: only the plotter thread may call Matplotlib
PLOT_QUEUE: "Queue[tuple[callable, tuple, dict]]" = Queue()


def _plotter():
    """Dedicated plotter thread: serializes all Matplotlib access."""
    while True:
        task = PLOT_QUEUE.get()
        if task is None:  # shutdown sentinel
            PLOT_QUEUE.task_done()
            break
        func, args, kwargs = task
        try:
            func(*args, **kwargs)
        except Exception as e:
            logging.exception(f"Plot task failed: {e}")
        finally:
            PLOT_QUEUE.task_done()


def _enqueue_plot(func, *args, **kwargs):
    """Enqueue a plotting task to be executed by the plotter thread."""
    PLOT_QUEUE.put((func, args, kwargs))


def _get_all_price_histories(tickers: list[str]) -> dict[str, pd.DataFrame]:
    """Fetch price history for all tickers in a single BigQuery call and split by ticker."""
    if not tickers:
        return {}
    client = bigquery.Client(project=config.SOURCE_PROJECT_ID)
    fixed_start_date = "2020-01-01"
    query = """
        SELECT ticker, date, adj_close
        FROM `{}`
        WHERE ticker IN UNNEST(@tickers) AND date >= @start_date
        ORDER BY ticker, date ASC
    """.format(config.PRICE_DATA_TABLE_ID)
    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ArrayQueryParameter("tickers", "STRING", tickers),
            bigquery.ScalarQueryParameter("start_date", "DATE", fixed_start_date),
        ]
    )
    full_df = client.query(query, job_config=job_config).to_dataframe()
    return {t: grp.copy() for t, grp in full_df.groupby("ticker")}


def _generate_price_chart(ticker: str, price_df: pd.DataFrame) -> str | None:
    """Generates a 90-day price chart, saves locally, uploads to GCS, returns GCS URI."""
    if price_df is None or price_df.empty:
        logging.warning(f"[{ticker}] No price data available for price chart.")
        return None

    df = price_df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "adj_close"]).sort_values("date")

    if len(df) >= 50:
        df["sma_50"] = df["adj_close"].rolling(window=50).mean()
    if len(df) >= 200:
        df["sma_200"] = df["adj_close"].rolling(window=200).mean()

    plot_df = df.tail(90)
    if plot_df.empty:
        return None

    today_str = date.today().strftime('%Y-%m-%d')
    # --- MODIFICATION 1: Change file extension ---
    local_file_path = f"{ticker}_price_chart_{today_str}.webp"

    with _PLOT_LOCK:
        fig, ax = plt.subplots(figsize=(10, 5), dpi=160)
        try:
            bg, price_c, sma50_c, sma200_c = "#20222D", "#9CFF0A", "#00BFFF", "#FF6347"
            fig.patch.set_facecolor(bg)
            ax.set_facecolor(bg)

            ax.plot(plot_df["date"], plot_df["adj_close"], label="Price", color=price_c, linewidth=2.2)
            if "sma_50" in plot_df.columns:
                ax.plot(plot_df["date"], plot_df["sma_50"], label="50-Day SMA", color=sma50_c, linestyle="--", linewidth=1.6)
            if "sma_200" in plot_df.columns:
                ax.plot(plot_df["date"], plot_df["sma_200"], label="200-Day SMA", color=sma200_c, linestyle="--", linewidth=1.6)

            ax.set_title(f"{ticker} — 90-Day Price", color="white", fontsize=14, pad=12)
            ax.tick_params(axis="both", colors="#C8C9CC", labelsize=9)
            ax.grid(True, linestyle="--", alpha=0.15)
            leg = ax.legend(loc="upper left", frameon=False)
            for text in leg.get_texts():
                text.set_color("#EDEEEF")

            plt.tight_layout()
            # --- MODIFICATION 2: Change save format ---
            fig.savefig(local_file_path, format="webp", bbox_inches="tight")
        finally:
            plt.close(fig)

    try:
        # --- MODIFICATION 3: Change blob name extension ---
        blob_name = f"{PRICE_CHART_FOLDER}{ticker}_{today_str}.webp"
        # --- MODIFICATION 4: Add content_type parameter ---
        gcs_uri = gcs.upload_from_filename(config.GCS_BUCKET_NAME, local_file_path, blob_name, content_type="image/webp")
        if gcs_uri:
            logging.info(f"[{ticker}] Successfully uploaded price chart to {gcs_uri}")
        return gcs_uri
    finally:
        if os.path.exists(local_file_path):
            os.remove(local_file_path)


def process_ticker(ticker: str, price_histories: dict):
    """Worker: prepare data & enqueue plotting tasks. No Matplotlib calls here."""
    logging.info(f"--- Processing price chart for {ticker} ---")

    price_df = price_histories.get(ticker)
    if price_df is None or price_df.empty:
        logging.warning(f"[{ticker}] No price data available for price chart.")
    else:
        _enqueue_plot(_generate_price_chart, ticker, price_df.copy())

    return ticker


def run_pipeline():
    """Orchestrate price chart generation with concurrent data prep + single-thread plotting."""
    logging.info("--- Starting Price Chart Generation Script ---")

    tickers = gcs.get_tickers()
    if not tickers:
        logging.critical("No tickers loaded from GCS. Exiting.")
        return

    logging.info(f"Found {len(tickers)} tickers to process.")
    price_histories = _get_all_price_histories(tickers)

    # Start dedicated plotter thread
    plot_thread = threading.Thread(target=_plotter, name="PlotterThread", daemon=True)
    plot_thread.start()

    # Prepare per-ticker work concurrently; enqueue plot tasks to the plotter
    count = 0
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(process_ticker, t, price_histories): t for t in tickers}
        for fut in as_completed(futures):
            t = futures[fut]
            try:
                if fut.result():
                    count += 1
            except Exception as e:
                logging.exception(f"[{t}] Unhandled error in worker: {e}")

    # Drain the plotting queue and cleanly stop the plotter
    PLOT_QUEUE.join()
    PLOT_QUEUE.put(None)
    plot_thread.join()

    logging.info(f"--- Price Chart Generation Finished. Processed {count} of {len(tickers)} tickers. ---")


if __name__ == "__main__":
    run_pipeline()