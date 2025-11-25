#!/bin/bash
# GCP RAG Spanner Management Script
# Helps manage RAG Engine configurations and corpora to stop Spanner charges

set -e

PROJECT_ID="supparay-voice-rag"
REGION="us-central1"
CORPUS_ID="6917529027641081856"  # supparay-voice-corpus

echo "=========================================="
echo "GCP RAG Spanner Management Tool"
echo "=========================================="
echo ""
echo "Project: $PROJECT_ID"
echo "Region: $REGION"
echo ""

# Set project
gcloud config set project "$PROJECT_ID" --quiet

# Get access token
TOKEN=$(gcloud auth print-access-token)

echo "Choose an action:"
echo "1. List all RAG corpora"
echo "2. Check RAG Engine configuration"
echo "3. Delete corpus using managed Spanner (supparay-voice-corpus)"
echo "4. Delete RAG Engine config for a region (unprovision Spanner)"
echo "5. Show current billing status"
echo ""
read -p "Enter choice (1-5): " choice

case $choice in
  1)
    echo ""
    echo "=== Listing RAG Corpora ==="
    curl -s -H "Authorization: Bearer $TOKEN" \
      "https://${REGION}-aiplatform.googleapis.com/v1/projects/${PROJECT_ID}/locations/${REGION}/ragCorpora" \
      | python3 -m json.tool
    ;;
    
  2)
    echo ""
    echo "=== RAG Engine Configuration ==="
    curl -s -H "Authorization: Bearer $TOKEN" \
      "https://${REGION}-aiplatform.googleapis.com/v1/projects/${PROJECT_ID}/locations/${REGION}/ragEngineConfig" \
      | python3 -m json.tool
    ;;
    
  3)
    echo ""
    echo "⚠️  WARNING: This will permanently delete the corpus 'supparay-voice-corpus'"
    echo "This will stop Spanner charges for this corpus."
    read -p "Are you sure? (yes/no): " confirm
    if [ "$confirm" = "yes" ]; then
      echo ""
      echo "Deleting corpus..."
      response=$(curl -s -w "\n%{http_code}" -X DELETE \
        -H "Authorization: Bearer $TOKEN" \
        "https://${REGION}-aiplatform.googleapis.com/v1/projects/${PROJECT_ID}/locations/${REGION}/ragCorpora/${CORPUS_ID}")
      
      http_code=$(echo "$response" | tail -n1)
      body=$(echo "$response" | sed '$d')
      
      if [ "$http_code" = "200" ] || [ "$http_code" = "204" ]; then
        echo "✅ Corpus deleted successfully!"
        echo "Spanner charges should stop within 24-48 hours."
      else
        echo "❌ Error deleting corpus (HTTP $http_code):"
        echo "$body" | python3 -m json.tool 2>/dev/null || echo "$body"
      fi
    else
      echo "Cancelled."
    fi
    ;;
    
  4)
    echo ""
    echo "⚠️  WARNING: This will unprovision the RAG-managed Spanner instance"
    echo "You must delete all corpora first!"
    read -p "Have you deleted all corpora? (yes/no): " confirm
    if [ "$confirm" = "yes" ]; then
      echo ""
      echo "Unprovisioning RAG Engine Spanner..."
      response=$(curl -s -w "\n%{http_code}" -X PATCH \
        -H "Content-Type: application/json" \
        -H "Authorization: Bearer $TOKEN" \
        "https://${REGION}-aiplatform.googleapis.com/v1/projects/${PROJECT_ID}/locations/${REGION}/ragEngineConfig" \
        -d '{"ragManagedDbConfig": { "unprovisioned": {} }}')
      
      http_code=$(echo "$response" | tail -n1)
      body=$(echo "$response" | sed '$d')
      
      if [ "$http_code" = "200" ]; then
        echo "✅ RAG Engine Spanner unprovisioned successfully!"
        echo "$body" | python3 -m json.tool
      else
        echo "❌ Error unprovisioning (HTTP $http_code):"
        echo "$body" | python3 -m json.tool 2>/dev/null || echo "$body"
      fi
    else
      echo "Please delete all corpora first, then run this option again."
    fi
    ;;
    
  5)
    echo ""
    echo "=== Checking All Regions for RAG Engine Config ==="
    for region in us-central1 us-east1 us-west1 europe-west1 asia-east1; do
      echo ""
      echo "--- Region: $region ---"
      curl -s -H "Authorization: Bearer $TOKEN" \
        "https://${region}-aiplatform.googleapis.com/v1/projects/${PROJECT_ID}/locations/${region}/ragEngineConfig" \
        | python3 -m json.tool 2>/dev/null | grep -A 5 "ragManagedDbConfig" || echo "  No config or error"
    done
    ;;
    
  *)
    echo "Invalid choice."
    exit 1
    ;;
esac

echo ""
echo "Done!"





