# enrichment/main.py
import logging
import functions_framework
from core.pipelines import (
    options_candidate_selector as candidate_selector_pipeline,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

@functions_framework.http
def run_options_candidate_selector(request):
    """Programmatic options top-5 CALL/PUT selector."""
    candidate_selector_pipeline.run_pipeline()
    return "Options candidate selector pipeline finished.", 200