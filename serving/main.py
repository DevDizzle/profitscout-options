# serving/main.py
import functions_framework
import logging
from core.pipelines import (
    page_generator,
    price_chart_generator,
    revenue_chart_generator,
    momentum_chart_generator,
    data_bundler,
    recommendations_generator,
    sync_to_firestore,
    data_cruncher, 
    dashboard_generator, # <--- ADD THIS IMPORT
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# --- ADD THIS NEW FUNCTION ---
@functions_framework.http
def run_dashboard_generator(request):
    """
    Entry point for the dashboard JSON assembly pipeline.
    """
    dashboard_generator.run_pipeline()
    return "Dashboard generator pipeline finished.", 200
# -----------------------------

@functions_framework.http
def run_data_cruncher(request):
    """
    Entry point for the data cruncher (prep stage) pipeline.
    Fetches raw data, calculates KPIs, and outputs a JSON file per ticker.
    """
    data_cruncher.run_pipeline()
    return "Data cruncher pipeline finished.", 200

@functions_framework.http
def run_page_generator(request):
    """
    Finds recommendation files that are missing a page and processes them.
    """
    page_generator.run_pipeline()
    return "Page generator pipeline finished.", 200

@functions_framework.http
def run_price_chart_generator(request):
    """
    Entry point for the price chart image generator pipeline.
    """
    price_chart_generator.run_pipeline()
    return "Price chart generator pipeline finished.", 200

@functions_framework.http
def run_revenue_chart_generator(request):
    """
    Entry point for the revenue chart image generator pipeline.
    """
    revenue_chart_generator.run_pipeline()
    return "Revenue chart generator pipeline finished.", 200

@functions_framework.http
def run_momentum_chart_generator(request):
    """
    Entry point for the momentum chart image generator pipeline.
    """
    momentum_chart_generator.run_pipeline()
    return "Momentum chart generator pipeline finished.", 200

@functions_framework.http
def run_data_bundler(request):
    """
    Orchestrates the final assembly and loading of asset metadata.
    """
    data_bundler.run_pipeline()
    return "Data bundler pipeline finished.", 200

@functions_framework.http
def run_recommendations_generator(request):
    """
    Generates recommendation markdown files with text and charts.
    """
    recommendations_generator.run_pipeline()
    return "Recommendations generator pipeline finished.", 200

@functions_framework.http
def run_sync_to_firestore(request):
    """
    Syncs the final asset metadata from BigQuery to Firestore.
    """
    full_reset = False
    if request and request.is_json:
        data = request.get_json(silent=True)
        if data and data.get('full_reset') is True:
            full_reset = True
            logging.info("Full reset requested for Firestore sync.")

    sync_to_firestore.run_pipeline(full_reset=full_reset)
    return f"Sync to Firestore pipeline finished. Full reset: {full_reset}", 200