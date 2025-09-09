# serving/core/pipelines/revenue_chart_generator.py
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
REVENUE_CHART_FOLDER = "Revenue-Chart/"
FINANCIALS_FOLDER = "financial-statements/"
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


def _generate_revenue_chart(ticker: str, financials_data: dict) -> str | None:
    """Generates a YoY quarterly revenue chart and uploads it to GCS."""
    reports = financials_data.get("quarterly_reports", [])
    if len(reports) < 8:
        logging.warning(f"[{ticker}] Not enough quarterly data for YoY revenue chart ({len(reports)} quarters found, need 8).")
        return None

    rows = []
    for rpt in reports:
        is_data = rpt.get("income_statement", {})
        if "revenue" in is_data and "grossProfitRatio" in is_data and "date" in is_data:
            revenue_val = is_data.get("revenue")
            gross_profit_ratio = is_data.get("grossProfitRatio")
            
            rows.append({
                "date": is_data["date"],
                "period": f"{is_data.get('calendarYear','?')}-{is_data.get('period','?')}",
                "revenue": abs(revenue_val) if revenue_val is not None else 0,
                "grossMargin": abs(gross_profit_ratio * 100) if gross_profit_ratio is not None else 0
            })

    df = (pd.DataFrame(rows)
            .assign(date=lambda d: pd.to_datetime(d["date"], errors="coerce"))
            .dropna(subset=["date"])
            .sort_values("date")
            .tail(8)
            .reset_index(drop=True))

    if len(df) < 8:
        logging.warning(f"[{ticker}] After cleaning/sorting, still < 8 quarters for YoY.")
        return None

    labels = df["period"].tail(4).tolist()
    revenue_current = df["revenue"].tail(4).tolist()
    revenue_prior = df["revenue"].head(4).tolist()
    margin_current = df["grossMargin"].tail(4).tolist()

    x = np.arange(len(labels))
    width = 0.35

    today_str = date.today().strftime('%Y-%m-%d')
    # --- MODIFICATION 1: Change file extension ---
    local_file_path = f"{ticker}_revenue_chart_{today_str}.webp"

    with _PLOT_LOCK:
        fig, ax1 = plt.subplots(figsize=(10, 6), dpi=160)
        try:
            bg, text_c, bar_curr, bar_prev, line_c = "#20222D", "#EDEEEF", "#00BFFF", "#4D4D4D", "#9CFF0A"
            fig.patch.set_facecolor(bg)
            ax1.set_facecolor(bg)

            rects1 = ax1.bar(x - width/2, revenue_current, width, label='Current Year', color=bar_curr)
            rects2 = ax1.bar(x + width/2, revenue_prior, width, label='Prior Year', color=bar_prev)

            ax1.set_ylabel("Revenue (in Billions USD)", color=text_c, fontsize=10)
            ax1.tick_params(axis="y", labelcolor=text_c)
            ax1.get_yaxis().set_major_formatter(mticker.FuncFormatter(lambda val, p: f'${val/1e9:.1f}B'))

            ax2 = ax1.twinx()
            ax2.plot(x, margin_current, color=line_c, marker='o', linestyle='-', label="Gross Margin %")
            ax2.set_ylabel("Gross Margin (%)", color=line_c, fontsize=10)
            ax2.tick_params(axis="y", labelcolor=line_c)
            ax2.get_yaxis().set_major_formatter(mticker.FuncFormatter(lambda val, p: f'{val:.0f}%'))

            ax1.set_xticks(x)
            ax1.set_xticklabels(labels, rotation=45, ha="right", color=text_c, fontsize=9)
            ax1.set_title(f"{ticker} — Quarterly Revenue (Year-over-Year)", color=text_c, fontsize=14, pad=20)
            fig.legend(loc='upper center', bbox_to_anchor=(0.5, 0.95), ncol=3, frameon=False, labelcolor=text_c)

            plt.tight_layout(rect=[0, 0, 1, 0.9])
            # --- MODIFICATION 2: Change save format ---
            fig.savefig(local_file_path, format="webp", bbox_inches="tight")
        finally:
            plt.close(fig)

    try:
        # --- MODIFICATION 3: Change blob name extension ---
        blob_name = f"{REVENUE_CHART_FOLDER}{ticker}_{today_str}.webp"
        # --- MODIFICATION 4: Add content_type parameter ---
        gcs_uri = gcs.upload_from_filename(config.GCS_BUCKET_NAME, local_file_path, blob_name, content_type="image/webp")
        if gcs_uri:
            logging.info(f"[{ticker}] Successfully uploaded revenue chart to {gcs_uri}")
        return gcs_uri
    finally:
        if os.path.exists(local_file_path):
            os.remove(local_file_path)


def process_ticker(ticker: str):
    """Worker: prepare data & enqueue plotting tasks. No Matplotlib calls here."""
    logging.info(f"--- Processing revenue chart for {ticker} ---")

    latest_financials_blob = gcs.get_latest_blob_for_ticker(config.GCS_BUCKET_NAME, FINANCIALS_FOLDER, ticker)
    if latest_financials_blob:
        try:
            financials_data = json.loads(latest_financials_blob.download_as_text())
            _enqueue_plot(_generate_revenue_chart, ticker, financials_data)
        except Exception as e:
            logging.error(f"[{ticker}] Failed to process financials file: {e}")
    else:
        logging.warning(f"[{ticker}] No financial statement file found. Skipping revenue chart.")

    return ticker


def run_pipeline():
    """Orchestrate revenue chart generation with concurrent data prep + single-thread plotting."""
    logging.info("--- Starting Revenue Chart Generation Script ---")

    tickers = gcs.get_tickers()
    if not tickers:
        logging.critical("No tickers loaded from GCS. Exiting.")
        return

    logging.info(f"Found {len(tickers)} tickers to process.")

    # Start dedicated plotter thread
    plot_thread = threading.Thread(target=_plotter, name="PlotterThread", daemon=True)
    plot_thread.start()

    # Prepare per-ticker work concurrently; enqueue plot tasks to the plotter
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

    # Drain the plotting queue and cleanly stop the plotter
    PLOT_QUEUE.join()
    PLOT_QUEUE.put(None)
    plot_thread.join()

    logging.info(f"--- Revenue Chart Generation Finished. Processed {count} of {len(tickers)} tickers. ---")


if __name__ == "__main__":
    run_pipeline()