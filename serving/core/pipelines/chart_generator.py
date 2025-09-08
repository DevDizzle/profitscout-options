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
REVENUE_CHART_FOLDER = "Revenue-Chart/"
MOMENTUM_CHART_FOLDER = "Momentum-Chart/"
FINANCIALS_FOLDER = "financial-statements/"
TECHNICALS_FOLDER = "technicals/"
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
    local_file_path = f"{ticker}_price_chart_{today_str}.png"

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
            fig.savefig(local_file_path, format="png", bbox_inches="tight")
        finally:
            plt.close(fig)

    try:
        blob_name = f"{PRICE_CHART_FOLDER}{ticker}_{today_str}.png"
        gcs_uri = gcs.upload_from_filename(config.GCS_BUCKET_NAME, local_file_path, blob_name)
        if gcs_uri:
            logging.info(f"[{ticker}] Successfully uploaded price chart to {gcs_uri}")
        return gcs_uri
    finally:
        if os.path.exists(local_file_path):
            os.remove(local_file_path)


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
            rows.append({
                "date": is_data["date"],
                "period": f"{is_data.get('calendarYear','?')}-{is_data.get('period','?')}",
                "revenue": is_data["revenue"],
                "grossMargin": is_data["grossProfitRatio"] * 100
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
    local_file_path = f"{ticker}_revenue_chart_{today_str}.png"

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
            fig.savefig(local_file_path, format="png", bbox_inches="tight")
        finally:
            plt.close(fig)

    try:
        blob_name = f"{REVENUE_CHART_FOLDER}{ticker}_{today_str}.png"
        gcs_uri = gcs.upload_from_filename(config.GCS_BUCKET_NAME, local_file_path, blob_name)
        if gcs_uri:
            logging.info(f"[{ticker}] Successfully uploaded revenue chart to {gcs_uri}")
        return gcs_uri
    finally:
        if os.path.exists(local_file_path):
            os.remove(local_file_path)


def _generate_momentum_chart(ticker: str, technicals_data: dict) -> str | None:
    """Generates a 90-day momentum (RSI/MACD) chart and uploads it to GCS."""
    timeseries = technicals_data.get("technicals_timeseries", [])
    if not timeseries:
        logging.warning(f"[{ticker}] No technical timeseries data found.")
        return None

    df = pd.DataFrame(timeseries).copy()
    if "date" not in df.columns:
        logging.warning(f"[{ticker}] 'date' missing in technicals timeseries.")
        return None

    df['date'] = pd.to_datetime(df['date'], errors="coerce")
    df = df.dropna(subset=['date']).sort_values('date').tail(90)
    if df.empty:
        return None

    today_str = date.today().strftime('%Y-%m-%d')
    local_file_path = f"{ticker}_momentum_chart_{today_str}.png"

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
            else:
                ax1.text(0.5, 0.5, "RSI_14 not available", color=text_color, ha="center", va="center", transform=ax1.transAxes)

            # MACD trio
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
            fig.savefig(local_file_path, format="png", bbox_inches="tight")
        finally:
            plt.close(fig)

    try:
        blob_name = f"{MOMENTUM_CHART_FOLDER}{ticker}_{today_str}.png"
        gcs_uri = gcs.upload_from_filename(config.GCS_BUCKET_NAME, local_file_path, blob_name)
        if gcs_uri:
            logging.info(f"[{ticker}] Successfully uploaded momentum chart to {gcs_uri}")
        return gcs_uri
    finally:
        if os.path.exists(local_file_path):
            os.remove(local_file_path)


def process_ticker(ticker: str, price_histories: dict):
    """Worker: prepare data & enqueue plotting tasks. No Matplotlib calls here."""
    logging.info(f"--- Processing charts for {ticker} ---")

    # Price chart
    price_df = price_histories.get(ticker)
    if price_df is None or price_df.empty:
        logging.warning(f"[{ticker}] No price data available for price chart.")
    else:
        _enqueue_plot(_generate_price_chart, ticker, price_df.copy())

    # Revenue chart
    latest_financials_blob = gcs.get_latest_blob_for_ticker(config.GCS_BUCKET_NAME, FINANCIALS_FOLDER, ticker)
    if latest_financials_blob:
        try:
            financials_data = json.loads(latest_financials_blob.download_as_text())
            _enqueue_plot(_generate_revenue_chart, ticker, financials_data)
        except Exception as e:
            logging.error(f"[{ticker}] Failed to process financials file: {e}")
    else:
        logging.warning(f"[{ticker}] No financial statement file found. Skipping revenue chart.")

    # Momentum chart
    technicals_blob = gcs.get_latest_blob_for_ticker(config.GCS_BUCKET_NAME, TECHNICALS_FOLDER, ticker)
    if technicals_blob:
        try:
            technicals_data = json.loads(technicals_blob.download_as_text())
            _enqueue_plot(_generate_momentum_chart, ticker, technicals_data)
        except Exception as e:
            logging.error(f"[{ticker}] Failed to process technicals file: {e}")
    else:
        logging.warning(f"[{ticker}] No technicals file found. Skipping momentum chart.")

    return ticker


def main():
    """Orchestrate chart generation with concurrent data prep + single-thread plotting."""
    logging.info("--- Starting Chart Generation Script ---")

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

    logging.info(f"--- Chart Generation Finished. Processed {count} of {len(tickers)} tickers. ---")


if __name__ == "__main__":
    main()
