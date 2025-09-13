# serving/core/pipelines/momentum_chart_generator.py
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

from google.cloud import storage

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from .. import config, gcs

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
MOMENTUM_CHART_FOLDER = "Momentum-Chart/"
TECHNICALS_FOLDER = "technicals/"
TICKER_LIST_PATH = "tickerlist.txt"

_PLOT_LOCK = threading.Lock()
PLOT_QUEUE: "Queue[tuple[callable, tuple, dict]]" = Queue()


def _plotter():
    """Dedicated plotter thread."""
    while True:
        task = PLOT_QUEUE.get()
        if task is None:
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
    """Enqueue a plotting task."""
    PLOT_QUEUE.put((func, args, kwargs))

def _delete_old_momentum_charts(ticker: str):
    """Deletes all previous momentum chart images for a given ticker."""
    prefix = f"{MOMENTUM_CHART_FOLDER}{ticker}_"
    blobs_to_delete = gcs.list_blobs(config.GCS_BUCKET_NAME, prefix)
    for blob_name in blobs_to_delete:
        try:
            gcs.delete_blob(config.GCS_BUCKET_NAME, blob_name)
        except Exception as e:
            logging.error(f"[{ticker}] Failed to delete old momentum chart {blob_name}: {e}")

def _generate_momentum_chart(ticker: str, technicals_data: dict) -> str | None:
    """Generates a 90-day momentum (RSI/MACD) chart and uploads it to GCS."""
    timeseries = technicals_data.get("technicals_timeseries", [])
    if not timeseries:
        return None

    df = pd.DataFrame(timeseries).copy()
    if "date" not in df.columns:
        return None

    df['date'] = pd.to_datetime(df['date'], errors="coerce")
    df = df.dropna(subset=['date']).sort_values('date').tail(90)
    if df.empty:
        return None

    today_str = date.today().strftime('%Y-%m-%d')
    local_file_path = f"{ticker}_momentum_chart_{today_str}.webp"

    with _PLOT_LOCK:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), dpi=160, sharex=True, gridspec_kw={'height_ratios': [2, 1]})
        try:
            bg_color, text_color = "#20222D", "#EDEEEF"
            fig.patch.set_facecolor(bg_color)
            ax1.set_facecolor(bg_color)
            ax2.set_facecolor(bg_color)

            if 'RSI_14' in df.columns:
                ax1.plot(df['date'], df['RSI_14'], color="#00BFFF", label='RSI (14)')
                ax1.axhline(70, linestyle='--', color='red', linewidth=0.7)
                ax1.axhline(30, linestyle='--', color='green', linewidth=0.7)
                ax1.set_ylim(0, 100)
                ax1.set_ylabel('RSI', color=text_color, fontsize=10)
                ax1.tick_params(axis='y', labelcolor=text_color)
                ax1.grid(True, linestyle="--", alpha=0.15)
                ax1.set_title(f"{ticker} — 90-Day Momentum Indicators", color=text_color, fontsize=14, pad=12)

            have_hist = 'MACDh_12_26_9' in df.columns
            have_macd = 'MACD_12_26_9' in df.columns
            have_sig  = 'MACDs_12_26_9' in df.columns

            if have_hist:
                bar_colors = ['green' if x > 0 else 'red' for x in df['MACDh_12_26_9']]
                ax2.bar(df['date'], df['MACDh_12_26_9'], label='Histogram', color=bar_colors)
            if have_macd:
                ax2.plot(df['date'], df['MACD_12_26_9'], color="#FF6347", label='MACD')
            if have_sig:
                ax2.plot(df['date'], df['MACDs_12_26_9'], color="#9CFF0A", label='Signal')

            ax2.set_ylabel('MACD', color=text_color, fontsize=10)
            ax2.tick_params(axis='x', labelcolor=text_color, rotation=45)
            ax2.tick_params(axis='y', labelcolor=text_color)
            ax2.grid(True, linestyle="--", alpha=0.15)
            fig.legend(loc='lower center', ncol=4, bbox_to_anchor=(0.5, -0.05), frameon=False, labelcolor=text_color)
            plt.tight_layout(rect=[0, 0, 1, 0.98])
            fig.subplots_adjust(bottom=0.2)
            fig.savefig(local_file_path, format="webp", bbox_inches="tight")
        finally:
            plt.close(fig)

    try:
        _delete_old_momentum_charts(ticker)
        blob_name = f"{MOMENTUM_CHART_FOLDER}{ticker}_{today_str}.webp"
        gcs_uri = gcs.upload_from_filename(config.GCS_BUCKET_NAME, local_file_path, blob_name, content_type="image/webp")
        if gcs_uri:
            logging.info(f"[{ticker}] Successfully uploaded momentum chart to {gcs_uri}")
        return gcs_uri
    finally:
        if os.path.exists(local_file_path):
            os.remove(local_file_path)

def process_ticker(ticker: str):
    """Worker: prepare data & enqueue plotting tasks."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(config.GCS_BUCKET_NAME)
    blob_name = f"{TECHNICALS_FOLDER}{ticker}_technicals.json"
    technicals_blob = bucket.get_blob(blob_name)

    if technicals_blob:
        try:
            technicals_data = json.loads(technicals_blob.download_as_text())
            _enqueue_plot(_generate_momentum_chart, ticker, technicals_data)
        except Exception as e:
            logging.error(f"[{ticker}] Failed to process technicals file: {e}")
    else:
        logging.warning(f"[{ticker}] No technicals file found at '{blob_name}'.")
    return ticker

def run_pipeline():
    """Orchestrate momentum chart generation."""
    logging.info("--- Starting Momentum Chart Generation Script ---")
    tickers = gcs.get_tickers()
    if not tickers:
        logging.critical("No tickers loaded. Exiting.")
        return

    plot_thread = threading.Thread(target=_plotter, name="PlotterThread", daemon=True)
    plot_thread.start()

    count = 0
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(process_ticker, t): t for t in tickers}
        for fut in as_completed(futures):
            t = futures[fut]
            try:
                if fut.result():
                    count += 1
            except Exception as e:
                logging.exception(f"[{t}] Unhandled error in worker: {e}")

    PLOT_QUEUE.join()
    PLOT_QUEUE.put(None)
    plot_thread.join()
    logging.info(f"--- Momentum Chart Generation Finished. Processed {count} of {len(tickers)} tickers. ---")