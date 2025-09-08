# serving/main.py
import functions_framework
import logging
from core.pipelines import (
    page_generator,
    chart_generator,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

@functions_framework.http
def run_page_generator(request):
    """
    Finds recommendation files that are missing a page and processes them.
    """
    page_generator.run_pipeline()
    return "Page generator pipeline finished.", 200

@functions_framework.http
def run_chart_generator(request):
    """
    Entry point for the chart image generator pipeline.
    """
    # We call the main() function from the script to run the pipeline
    chart_generator.main() 
    return "Chart generator pipeline finished.", 200