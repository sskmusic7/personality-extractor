# GCP Authentication Setup

## Current Status

✅ **gcloud CLI**: Authenticated as `sskmusic7@gmail.com`
✅ **Project**: `eminent-century-464801-b0`
✅ **APIs Enabled**: Vertex AI, Cloud Storage

## What You Need To Do

The Python SDK needs **application-default credentials**. Run this command:

```bash
gcloud auth application-default login
```

This will:
1. Open a browser window
2. Ask you to sign in (use sskmusic7@gmail.com)
3. Ask for authorization
4. Save credentials locally

## After Authentication

Once you run the command above, the system will automatically:
- ✅ Connect to Vertex AI
- ✅ Use Gemini 2.5 Flash
- ✅ Upload to GCS buckets
- ✅ Create embeddings
- ✅ Extract patterns with LLM

## Verify It Works

After running `gcloud auth application-default login`, test with:

```bash
python3 -c "from gcp_vertex_rag import GCPVertexRAGManager; m = GCPVertexRAGManager(); print('✅ Ready!' if m.is_configured() else '❌ Not ready')"
```

## Alternative: Service Account (For Production)

If you prefer service account (better for production):

```bash
# Create service account
gcloud iam service-accounts create personality-extractor-sa

# Grant permissions
gcloud projects add-iam-policy-binding eminent-century-464801-b0 \
    --member="serviceAccount:personality-extractor-sa@eminent-century-464801-b0.iam.gserviceaccount.com" \
    --role="roles/storage.admin"

gcloud projects add-iam-policy-binding eminent-century-464801-b0 \
    --member="serviceAccount:personality-extractor-sa@eminent-century-464801-b0.iam.gserviceaccount.com" \
    --role="roles/aiplatform.user"

# Download key
gcloud iam service-accounts keys create gcp-credentials.json \
    --iam-account=personality-extractor-sa@eminent-century-464801-b0.iam.gserviceaccount.com

# Set environment variable
export GOOGLE_APPLICATION_CREDENTIALS="$(pwd)/gcp-credentials.json"
```





