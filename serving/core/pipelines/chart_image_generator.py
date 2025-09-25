# serving/core/pipelines/chart_image_generator.py
import logging
import json
import pandas as pd
import mplfinance as mpf
from datetime import date
from concurrent.futures import ThreadPoolExecutor, as_completed
import tempfile
import os

from .. import config, gcs

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - [%(threadName)s] - %(message)s")
PRICE_CHART_JSON_FOLDER = "price-chart-json/"
OUTPUT_IMAGE_FOLDER = "price-chart-images/"
MAX_WORKERS = 8

def _delete_old_chart_images(ticker: str):
    """Deletes all previous chart image files for a given ticker."""
    prefix = f"{OUTPUT_IMAGE_FOLDER}{ticker}_"
    blobs_to_delete = gcs.list_blobs(config.GCS_BUCKET_NAME, prefix)
    for blob_name in blobs_to_delete:
        try:
            gcs.delete_blob(config.GCS_BUCKET_NAME, blob_name)
        except Exception as e:
            logging.error(f"[{ticker}] Failed to delete old chart image {blob_name}: {e}")

def _generate_chart_image(ticker: str, chart_json: dict) -> str | None:
    """
    Generates a static chart image from the provided JSON data and uploads it to GCS.
    """
    try:
        # --- 1. Prepare DataFrames from JSON ---
        df = pd.DataFrame(chart_json['candlestick'])
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date')

        volume_series = pd.DataFrame(chart_json['volume'])
        volume_series['date'] = pd.to_datetime(volume_series['date'])
        volume_series = volume_series.set_index('date')['value']
        
        df['volume'] = volume_series

        # Optimized: Avoid redundant DataFrame creation
        sma50_data = pd.DataFrame(chart_json['sma50'])
        sma50_df = sma50_data.set_index(pd.to_datetime(sma50_data['date'])).drop('date', axis=1)
        
        sma200_data = pd.DataFrame(chart_json['sma200'])
        sma200_df = sma200_data.set_index(pd.to_datetime(sma200_data['date'])).drop('date', axis=1)

        # --- 2. Define Chart Styling ---
        market_colors = mpf.make_marketcolors(
            up='#22c55e', down='#ef4444',
            wick={'up':'#22c55e', 'down':'#ef4444'},
            edge={'up':'#22c55e', 'down':'#ef4444'},
            volume={'up':'#22c55e', 'down':'#ef4444'}
        )
        
        # --- CORRECTED: Use a hex string with alpha transparency ---
        chart_style = mpf.make_mpf_style(
            base_mpf_style='nightclouds',
            marketcolors=market_colors,
            facecolor='#1C1C29',
            gridcolor='#FFFFFF1A',  # Hex string for transparent white
            edgecolor='#FFFFFF1A',  # Hex string for transparent white
            figcolor='#1C1C29',
            y_on_right=True,
            rc={
                "axes.labelcolor": "#A3A3A3",
                "xtick.color": "#A3A3A3",
                "ytick.color": "#A3A3A3",
                "text.color": "#A3A3A3"
            }
        )

        # --- 3. Create Additional Plots (Moving Averages) ---
        addplots = [
            mpf.make_addplot(sma50_df, color='#f97316', linestyle='dashed', width=1.2),
            mpf.make_addplot(sma200_df, color='#a855f7', linestyle='dashed', width=1.2)
        ]

        # --- 4. Generate and Save the Plot ---
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
            temp_filepath = temp_file.name
            mpf.plot(
                df,
                type='candle',
                style=chart_style,
                volume=True,
                addplot=addplots,
                figratio=(8, 4), 
                figscale=1.0, 
                panel_ratios=(3, 1),
                ylabel='',
                ylabel_lower='',
                savefig=temp_filepath
            )

        # --- 5. Upload to GCS ---
        today_str = date.today().strftime('%Y-%m-%d')
        destination_blob_name = f"{OUTPUT_IMAGE_FOLDER}{ticker}_{today_str}.png"
        
        _delete_old_chart_images(ticker)
        gcs_uri = gcs.upload_from_filename(
            config.GCS_BUCKET_NAME,
            temp_filepath,
            destination_blob_name,
            content_type="image/png"
        )
        
        os.remove(temp_filepath)

        if gcs_uri:
            logging.info(f"[{ticker}] Successfully generated and uploaded chart image to {gcs_uri}")
            return gcs_uri
        else:
            logging.error(f"[{ticker}] Failed to upload chart image.")
            return None

    except Exception as e:
        logging.error(f"[{ticker}] Failed to generate chart image: {e}", exc_info=True)
        return None

def process_ticker(ticker: str):
    """
    Worker function to fetch chart JSON and trigger image generation.
    """
    latest_blob = gcs.get_latest_blob_for_ticker(config.GCS_BUCKET_NAME, PRICE_CHART_JSON_FOLDER, ticker)
    if not latest_blob:
        logging.warning(f"[{ticker}] No price chart JSON found to generate an image.")
        return None
        
    try:
        chart_json_str = latest_blob.download_as_text()
        chart_json = json.loads(chart_json_str)
        return _generate_chart_image(ticker, chart_json)
    except (json.JSONDecodeError, Exception) as e:
        logging.error(f"[{ticker}] Failed to read or process price chart JSON '{latest_blob.name}': {e}")
        return None

def run_pipeline():
    """
    Orchestrates the static price chart image generation pipeline.
    """
    logging.info("--- Starting Chart Image Generation Pipeline ---")
    tickers = gcs.get_tickers()
    if not tickers:
        logging.critical("No tickers loaded. Exiting.")
        return

    processed_count = 0
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_ticker = {executor.submit(process_ticker, t): t for t in tickers}
        for future in as_completed(future_to_ticker):
            ticker = future_to_ticker[future]
            try:
                if future.result():
                    processed_count += 1
            except Exception as e:
                logging.exception(f"[{ticker}] Unhandled error in worker: {e}")
                
    logging.info(f"--- Chart Image Generation Finished. Processed {processed_count} of {len(tickers)} tickers. ---")