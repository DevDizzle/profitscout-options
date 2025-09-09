#!/bin/bash
#
# Deploys all Cloud Functions for the ProfitScout serving layer.
#
# Before running:
# 1. Make sure you have authenticated with gcloud: `gcloud auth login`
# 2. Configure gcloud with the destination project ID: `gcloud config set project [YOUR_DESTINATION_PROJECT_ID]`
# 3. Update the variables in the "Configuration" section below.
# 4. Make the script executable: `chmod +x deploy_serving.sh`
#
# Usage:
# ./deploy_serving.sh

set -e # Exit immediately if a command exits with a non-zero status.
set -u # Treat unset variables as an error.
set -o pipefail # Return the exit status of the last command in a pipeline that failed.

# --- Configuration ---
# The GCP Project ID where the functions will be deployed.
export DESTINATION_PROJECT_ID="profitscout-fida8"

# The GCP Project ID where source data resides (BigQuery, GCS).
export SOURCE_PROJECT_ID="profitscout-lx6bb"

# The region for the Cloud Functions.
export REGION="us-central1"

# The service account email to run the functions.
# Ensure this SA has roles like Cloud Functions Invoker, BigQuery User,
# Storage Object Admin, Firestore User, and Vertex AI User.
export SERVICE_ACCOUNT_EMAIL="your-service-account@${DESTINATION_PROJECT_ID}.iam.gserviceaccount.com"

# --- Shared Environment Variables ---
# These are used by multiple functions.
export GCS_BUCKET_NAME="profit-scout-data"
export DESTINATION_GCS_BUCKET_NAME="profit-scout"
export BIGQUERY_DATASET="profit_scout"
export MODEL_NAME="gemini-1.5-flash"
export FIRESTORE_COLLECTION="tickers"

# --- Deployment Logic ---

# A helper function to reduce repetition.
deploy_function() {
  local NAME=$1
  local ENTRY_POINT=$2
  local MEMORY=$3
  local ENV_VARS=$4

  echo "------------------------------------------------------"
  echo "Deploying: ${NAME}..."
  echo "------------------------------------------------------"

  gcloud functions deploy "${NAME}" \
    --gen2 \
    --runtime=python311 \
    --project="${DESTINATION_PROJECT_ID}" \
    --region="${REGION}" \
    --source=./serving \
    --entry-point="${ENTRY_POINT}" \
    --trigger-http \
    --allow-unauthenticated \
    --timeout=540s \
    --memory="${MEMORY}" \
    --service-account="${SERVICE_ACCOUNT_EMAIL}" \
    --set-env-vars="^##^${ENV_VARS}" # Using '^##^' as a delimiter
}

# --- Function Definitions ---
# The order of deployment does not matter.

# 1. Data Bundler: Syncs GCS data and assembles final metadata.
VARS_BUNDLER="SOURCE_PROJECT_ID=${SOURCE_PROJECT_ID}##DESTINATION_PROJECT_ID=${DESTINATION_PROJECT_ID}##GCS_BUCKET_NAME=${GCS_BUCKET_NAME}##DESTINATION_GCS_BUCKET_NAME=${DESTINATION_GCS_BUCKET_NAME}##BIGQUERY_DATASET=${BIGQUERY_DATASET}"
deploy_function "run-data-bundler" "run_data_bundler" "1024Mi" "${VARS_BUNDLER}"

# 2. Recommendations Generator: Generates markdown with text and charts via Vertex AI.
VARS_REC_GEN="SOURCE_PROJECT_ID=${SOURCE_PROJECT_ID}##GCS_BUCKET_NAME=${GCS_BUCKET_NAME}##BIGQUERY_DATASET=${BIGQUERY_DATASET}##MODEL_NAME=${MODEL_NAME}##SERVICE_ACCOUNT_EMAIL=${SERVICE_ACCOUNT_EMAIL}"
deploy_function "run-recommendations-generator" "run_recommendations_generator" "1024Mi" "${VARS_REC_GEN}"

# 3. Chart Generators: Create and upload chart images. They need more memory for matplotlib.
VARS_CHART="SOURCE_PROJECT_ID=${SOURCE_PROJECT_ID}##GCS_BUCKET_NAME=${GCS_BUCKET_NAME}"
deploy_function "run-price-chart-generator" "run_price_chart_generator" "1024Mi" "${VARS_CHART}"
deploy_function "run-revenue-chart-generator" "run_revenue_chart_generator" "1024Mi" "${VARS_CHART}"
deploy_function "run-momentum-chart-generator" "run_momentum_chart_generator" "1024Mi" "${VARS_CHART}"

# 4. Page Generator: Creates the final JSON for the web page via Vertex AI.
VARS_PAGE_GEN="SOURCE_PROJECT_ID=${SOURCE_PROJECT_ID}##GCS_BUCKET_NAME=${GCS_BUCKET_NAME}##BIGQUERY_DATASET=${BIGQUERY_DATASET}##MODEL_NAME=${MODEL_NAME}"
deploy_function "run-page-generator" "run_page_generator" "512Mi" "${VARS_PAGE_GEN}"

# 5. Firestore Sync: Syncs final data from BigQuery to Firestore.
VARS_FIRESTORE="DESTINATION_PROJECT_ID=${DESTINATION_PROJECT_ID}##FIRESTORE_COLLECTION=${FIRESTORE_COLLECTION}"
deploy_function "run-sync-to-firestore" "run_sync_to_firestore" "512Mi" "${VARS_FIRESTORE}"


echo "------------------------------------------------------"
echo "✅ All serving layer functions deployed successfully!"
echo "------------------------------------------------------"