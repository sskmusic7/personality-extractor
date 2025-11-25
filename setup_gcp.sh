#!/bin/bash

# GCP Setup Script for Character Personality Extraction System
# This script helps you set up GCP credentials and configure the project

set -e

echo "=========================================="
echo "GCP Setup for Personality Extractor"
echo "=========================================="
echo ""

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Check if gcloud is installed
if ! command -v gcloud &> /dev/null; then
    echo -e "${RED}✗ gcloud CLI not found${NC}"
    echo "Please install Google Cloud SDK: https://cloud.google.com/sdk/docs/install"
    exit 1
fi

echo -e "${GREEN}✓ gcloud CLI found${NC}"

# Authenticate with GCP
echo ""
echo "Step 1: Authenticating with GCP..."
gcloud auth login

# Set project
PROJECT_ID="my first project"
echo ""
echo "Step 2: Setting GCP project to '$PROJECT_ID'..."
gcloud config set project "$PROJECT_ID"

# Enable required APIs
echo ""
echo "Step 3: Enabling required GCP APIs..."
echo "This may take a few minutes..."

gcloud services enable storage-component.googleapis.com
gcloud services enable aiplatform.googleapis.com
gcloud services enable storage-api.googleapis.com

echo -e "${GREEN}✓ APIs enabled${NC}"

# Create service account
SERVICE_ACCOUNT_NAME="personality-extractor-sa"
SERVICE_ACCOUNT_EMAIL="${SERVICE_ACCOUNT_NAME}@${PROJECT_ID}.iam.gserviceaccount.com"

echo ""
echo "Step 4: Creating service account..."

# Check if service account already exists
if gcloud iam service-accounts describe "$SERVICE_ACCOUNT_EMAIL" &> /dev/null; then
    echo -e "${YELLOW}Service account already exists, skipping creation${NC}"
else
    gcloud iam service-accounts create "$SERVICE_ACCOUNT_NAME" \
        --display-name="Personality Extractor Service Account" \
        --description="Service account for Character Personality Extraction System"
    echo -e "${GREEN}✓ Service account created${NC}"
fi

# Grant necessary roles
echo ""
echo "Step 5: Granting necessary permissions..."

gcloud projects add-iam-policy-binding "$PROJECT_ID" \
    --member="serviceAccount:${SERVICE_ACCOUNT_EMAIL}" \
    --role="roles/storage.admin"

gcloud projects add-iam-policy-binding "$PROJECT_ID" \
    --member="serviceAccount:${SERVICE_ACCOUNT_EMAIL}" \
    --role="roles/aiplatform.user"

echo -e "${GREEN}✓ Permissions granted${NC}"

# Create and download key
KEY_FILE="gcp-credentials.json"
echo ""
echo "Step 6: Creating and downloading service account key..."

# Delete old key if exists
if [ -f "$KEY_FILE" ]; then
    echo -e "${YELLOW}Old key file found, removing...${NC}"
    rm "$KEY_FILE"
fi

gcloud iam service-accounts keys create "$KEY_FILE" \
    --iam-account="$SERVICE_ACCOUNT_EMAIL"

echo -e "${GREEN}✓ Key file created: $KEY_FILE${NC}"

# Set environment variable
echo ""
echo "Step 7: Setting up environment variable..."
export GOOGLE_APPLICATION_CREDENTIALS="$(pwd)/$KEY_FILE"

# Add to .env file if it exists, or create one
if [ -f .env ]; then
    if grep -q "GOOGLE_APPLICATION_CREDENTIALS" .env; then
        sed -i '' "s|GOOGLE_APPLICATION_CREDENTIALS=.*|GOOGLE_APPLICATION_CREDENTIALS=$(pwd)/$KEY_FILE|" .env
    else
        echo "GOOGLE_APPLICATION_CREDENTIALS=$(pwd)/$KEY_FILE" >> .env
    fi
else
    echo "GOOGLE_APPLICATION_CREDENTIALS=$(pwd)/$KEY_FILE" > .env
    echo "GCP_PROJECT_ID=$PROJECT_ID" >> .env
    echo "GCP_REGION=us-central1" >> .env
fi

echo -e "${GREEN}✓ Environment variables configured${NC}"

# Summary
echo ""
echo "=========================================="
echo -e "${GREEN}Setup Complete!${NC}"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. The service account key is saved as: $KEY_FILE"
echo "2. Environment variable is set in .env file"
echo "3. Install Python dependencies: pip install -r requirements.txt"
echo "4. Run the application: python app.py"
echo ""
echo "IMPORTANT: Keep $KEY_FILE secure and never commit it to git!"
echo ""
echo "To use this in your current shell session, run:"
echo "  export GOOGLE_APPLICATION_CREDENTIALS=$(pwd)/$KEY_FILE"
echo ""







