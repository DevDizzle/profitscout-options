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
    dashboard_generator,
    sync_options_to_firestore,
    sync_calendar_to_firestore,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

@functions_framework.http
def run_sync_calendar_to_firestore(request):
    """
    Syncs upcoming calendar events from BigQuery to Firestore.
    """
    # This pipeline does a full reset by default, so no parameter is needed.
    sync_calendar_to_firestore.run_pipeline()
    return "Sync calendar events to Firestore pipeline finished.", 200

@functions_framework.http
def run_sync_options_to_firestore(request):
    full_reset = False
    if request and request.is_json:
        data = request.get_json(silent=True)
        if data and data.get('full_reset') is True:
            full_reset = True
    sync_options_to_firestore.run_pipeline(full_reset=full_reset)
    return f"Sync options to Firestore pipeline finished. Full reset: {full_reset}", 200

@functions_framework.http
def run_dashboard_generator(request):
    dashboard_generator.run_pipeline()
    return "Dashboard generator pipeline finished.", 200

@functions_framework.http
def run_data_cruncher(request):
    data_cruncher.run_pipeline()
    return "Data cruncher pipeline finished.", 200

@functions_framework.http
def run_page_generator(request):
    page_generator.run_pipeline()
    return "Page generator pipeline finished.", 200

@functions_framework.http
def run_price_chart_generator(request):
    price_chart_generator.run_pipeline()
    return "Price chart generator pipeline finished.", 200

@functions_framework.http
def run_revenue_chart_generator(request):
    revenue_chart_generator.run_pipeline()
    return "Revenue chart generator pipeline finished.", 200

@functions_framework.http
def run_momentum_chart_generator(request):
    momentum_chart_generator.run_pipeline()
    return "Momentum chart generator pipeline finished.", 200

@functions_framework.http
def run_data_bundler(request):
    data_bundler.run_pipeline()
    return "Data bundler pipeline finished.", 200

@functions_framework.http
def run_recommendations_generator(request):
    recommendations_generator.run_pipeline()
    return "Recommendations generator pipeline finished.", 200

@functions_framework.http
def run_sync_to_firestore(request):
    full_reset = False
    if request and request.is_json:
        data = request.get_json(silent=True)
        if data and data.get('full_reset') is True:
            full_reset = True
    sync_to_firestore.run_pipeline(full_reset=full_reset)
    return f"Sync to Firestore pipeline finished. Full reset: {full_reset}", 200