# GCP RAG Spanner Audit Report
**Date:** January 17, 2025  
**Audit Scope:** All GCP projects for RAG Engine managed Spanner usage

## Executive Summary

**CRITICAL FINDING:** The project `supparay-voice-rag` has an **ACTIVE RAG corpus using managed Spanner database**, which is causing Spanner charges even though the Spanner API is disabled in your project.

## Projects with RAG Engine Configuration

### 1. ✅ **supparay-voice-rag** (PROJECT NUMBER: 731310706836)
**STATUS: ⚠️ ACTIVE USAGE - CHARGES LIKELY**

- **RAG Engine Config:** Basic tier (100 processing units) in 5 regions:
  - us-central1 ✅
  - us-east1 ✅
  - us-west1 ✅
  - europe-west1 ✅
  - asia-east1 ✅

- **RAG Corpora:**
  - **"supparay-voice-corpus"** (ID: 6917529027641081856)
    - Status: **ACTIVE**
    - Created: 2025-08-11
    - **Vector DB: `ragManagedDb` (MANAGED SPANNER)** ⚠️
    - **THIS IS CAUSING SPANNER CHARGES**
  
  - "Supparay MK 2" (ID: 7991637538768945152)
    - Status: INITIALIZED
    - Created: 2025-08-31
    - Vector DB: `vertexVectorSearch` (NOT using managed Spanner) ✅

**ACTION REQUIRED:** This project has an active corpus using managed Spanner. Even with Spanner API disabled, the managed Spanner instance continues to run and incur charges.

---

### 2. ✅ **blackwidow-467921** (PROJECT NUMBER: 282281157856)
**STATUS: ✅ NO ACTIVE USAGE - NO CHARGES**

- **RAG Engine Config:** Basic tier in 5 regions (all regions configured)
- **RAG Corpora:** None found
- **Risk:** Low - No corpora using managed Spanner

---

### 3. ✅ **ssk-newsletter-gcloud** (PROJECT NUMBER: 91400501677)
**STATUS: ✅ NO ACTIVE USAGE - NO CHARGES**

- **RAG Engine Config:** Basic tier in 5 regions (all regions configured)
- **RAG Corpora:** None found
- **Risk:** Low - No corpora using managed Spanner

---

### 4. ✅ **eminent-century-464801-b0** (PROJECT NUMBER: 287783957820)
**STATUS: ✅ NO ACTIVE USAGE - NO CHARGES**

- **RAG Engine Config:** Basic tier in 5 regions (all regions configured)
- **RAG Corpora:** None found
- **Risk:** Low - No corpora using managed Spanner

---

## Projects WITHOUT RAG Engine Configuration

These projects do NOT have RAG Engine configured and are NOT using managed Spanner:

- gen-lang-client-0742721898 (Forbes)
- gen-lang-client-0765427453 (OurPain2Power)
- gen-lang-client-0302604660 (aisha)
- gen-lang-client-0611799975 (Gemini API)
- suparray-project (Suparray Project - billing not enabled)
- website-builder-468300 (Website Builder)

---

## Why You're Still Being Charged

According to Google's email:
> "As long as there is one RagCorpus in a region that chooses to use the RAG-managed database (Spanner) for embedding indexing and vector search, you will be charged for using RAG-managed database (Spanner) for embedding indexing and vector search from that region."

**Root Cause:**
- The corpus "supparay-voice-corpus" in `supparay-voice-rag` (us-central1) is using `ragManagedDb`
- Even though you disabled the Spanner API, the managed Spanner instance is provisioned in Google's RAG-managed project (not yours)
- The instance continues to run and charge you as long as this corpus exists

---

## Recommended Actions

### Option 1: Delete the Corpus Using Managed Spanner (Recommended if not needed)

```bash
# Set project
gcloud config set project supparay-voice-rag

# Delete the corpus (replace CORPUS_ID with actual ID)
curl -X DELETE \
  -H "Authorization: Bearer $(gcloud auth print-access-token)" \
  "https://us-central1-aiplatform.googleapis.com/v1/projects/supparay-voice-rag/locations/us-central1/ragCorpora/6917529027641081856"
```

**Note:** This will permanently delete the corpus and all its data.

---

### Option 2: Migrate Corpus to Vertex Vector Search (Recommended if you need the data)

You would need to:
1. Export the corpus data
2. Create a new corpus using `vertexVectorSearch` instead of `ragManagedDb`
3. Re-import the data
4. Delete the old corpus

---

### Option 3: Delete RAG Engine Configuration Entirely (If not using RAG at all)

```bash
# Set project
gcloud config set project supparay-voice-rag

# Delete RAG Engine config for us-central1
curl -X PATCH \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $(gcloud auth print-access-token)" \
  "https://us-central1-aiplatform.googleapis.com/v1/projects/supparay-voice-rag/locations/us-central1/ragEngineConfig" \
  -d '{"ragManagedDbConfig": { "unprovisioned": {} }}'
```

**Note:** You must delete all corpora first before deleting the RAG Engine config.

---

### Option 4: Downgrade to Basic Tier (Already on Basic)

You're already on Basic tier (100 processing units), so this won't help reduce costs further.

---

## Cost Impact

- **Basic Tier:** 100 processing units per region
- **Current Setup:** Basic tier in 5 regions = 500 processing units total
- **Active Usage:** Only us-central1 has an active corpus, so you're likely only being charged for that region
- **Estimated Cost:** ~$65-90/month per region (Basic tier pricing)

---

## Summary Table

| Project | RAG Engine | Corpora Using Spanner | Status | Action Needed |
|---------|------------|----------------------|--------|---------------|
| supparay-voice-rag | ✅ Yes (5 regions) | ✅ 1 ACTIVE corpus | ⚠️ **CHARGING** | **DELETE CORPUS OR MIGRATE** |
| blackwidow-467921 | ✅ Yes (5 regions) | ❌ None | ✅ No charges | Monitor only |
| ssk-newsletter-gcloud | ✅ Yes (5 regions) | ❌ None | ✅ No charges | Monitor only |
| eminent-century-464801-b0 | ✅ Yes (5 regions) | ❌ None | ✅ No charges | Monitor only |

---

## Next Steps

1. **IMMEDIATE:** Delete or migrate the "supparay-voice-corpus" in `supparay-voice-rag` if not needed
2. **OPTIONAL:** Consider deleting RAG Engine configs in other projects if not being used
3. **MONITOR:** Check billing dashboard to confirm charges stop after deletion

---

## Verification Commands

After taking action, verify with:

```bash
# Check if corpus still exists
gcloud config set project supparay-voice-rag
curl -H "Authorization: Bearer $(gcloud auth print-access-token)" \
  "https://us-central1-aiplatform.googleapis.com/v1/projects/supparay-voice-rag/locations/us-central1/ragCorpora"

# Check RAG Engine config
curl -H "Authorization: Bearer $(gcloud auth print-access-token)" \
  "https://us-central1-aiplatform.googleapis.com/v1/projects/supparay-voice-rag/locations/us-central1/ragEngineConfig"
```





