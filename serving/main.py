# serving/main.py
import functions_framework
import logging
from core.pipelines import (
    page_generator,
    options_recommendation_generator as options_reco_gen,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

@functions_framework.http
def run_options_recommendation_generator(request):
    """
    Entry point for the options recommendations markdown generator.
    Reads options_candidates and writes MD files to GCS (options-recommendations/).
    """
    options_reco_gen.run_pipeline()
    return "Options recommendation generator pipeline finished.", 200