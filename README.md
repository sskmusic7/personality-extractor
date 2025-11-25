# Character Personality Extraction System

A web-based system for extracting personality patterns from character scripts using RAG (Retrieval Augmented Generation) and Google Cloud Platform Vertex AI.

## Features

- 🎭 **Drag-and-Drop Interface**: Easy HTML frontend for uploading script files
- ☁️ **GCP Integration**: Uses Google Cloud Storage and Vertex AI RAG (free tier eligible)
- 📊 **Pattern Extraction**: Analyzes speech patterns, emotional responses, conflict handling
- 🐍 **Static Rules Generation**: Generates production-ready Python classes with personality rules
- 🚀 **Fast Processing**: Uses one-time RAG analysis → static rules (no runtime database overhead)

## Architecture

```
Script Files → GCS Bucket → Vertex AI RAG → Personality Patterns → Static Python Rules
```

## Prerequisites

- Python 3.8+
- Google Cloud Platform account
- GCP project with billing enabled (free tier is sufficient)
- Google Cloud SDK installed

## Installation

### 1. Clone/Download the Repository

```bash
cd "/Users/sskmusic/PErsonality Extractor"
```

### 2. Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 3. Set Up GCP

#### Option A: Automated Setup (Recommended)

```bash
chmod +x setup_gcp.sh
./setup_gcp.sh
```

This script will:
- Authenticate with GCP
- Set project to "my first project"
- Enable required APIs
- Create service account with proper permissions
- Download credentials file

#### Option B: Manual Setup

1. **Install Google Cloud SDK**:
   ```bash
   # macOS
   brew install google-cloud-sdk
   
   # Or download from: https://cloud.google.com/sdk/docs/install
   ```

2. **Authenticate**:
   ```bash
   gcloud auth login
   gcloud config set project "my first project"
   ```

3. **Enable APIs**:
   ```bash
   gcloud services enable storage-component.googleapis.com
   gcloud services enable aiplatform.googleapis.com
   gcloud services enable storage-api.googleapis.com
   ```

4. **Create Service Account**:
   ```bash
   gcloud iam service-accounts create personality-extractor-sa \
       --display-name="Personality Extractor Service Account"
   
   gcloud projects add-iam-policy-binding "my first project" \
       --member="serviceAccount:personality-extractor-sa@my first project.iam.gserviceaccount.com" \
       --role="roles/storage.admin"
   
   gcloud projects add-iam-policy-binding "my first project" \
       --member="serviceAccount:personality-extractor-sa@my first project.iam.gserviceaccount.com" \
       --role="roles/aiplatform.user"
   ```

5. **Download Credentials**:
   ```bash
   gcloud iam service-accounts keys create gcp-credentials.json \
       --iam-account=personality-extractor-sa@my first project.iam.gserviceaccount.com
   ```

6. **Set Environment Variable**:
   ```bash
   export GOOGLE_APPLICATION_CREDENTIALS="$(pwd)/gcp-credentials.json"
   ```

## Usage

### 1. Start the Server

```bash
python app.py
```

The server will start on `http://localhost:5000`

### 2. Open the Web Interface

Open your browser and navigate to:
```
http://localhost:5000
```

### 3. Upload Script Files

1. Enter the character name (e.g., "Tony Stark")
2. Drag and drop script files (`.txt`, `.json`, `.csv`, or `.zip`)
   - **TXT Format**: `CHARACTER: dialogue text`
   - **JSON Format**: See example in `Reference Code/example_usage.py`
   - **CSV Format**: Columns: `character,dialogue,scene,episode,other_characters`
3. Click "Upload & Extract Personality"

### 4. Download Results

After processing, download:
- **Python Rules File**: Static personality class for chatbot integration
- **JSON Patterns File**: Complete analysis data

## Project Structure

```
PErsonality Extractor/
├── app.py                              # Flask backend API
├── static/
│   └── index.html                      # HTML frontend
├── character_personality_extractor_cloud.py  # Cloud-adapted extractor
├── gcp_vertex_rag.py                   # GCP Vertex AI RAG integration
├── config.py                           # Configuration settings
├── requirements.txt                    # Python dependencies
├── setup_gcp.sh                       # GCP setup script
├── uploads/                           # Temporary upload directory (created automatically)
├── output/                            # Generated files (created automatically)
└── Reference Code/                    # Original reference implementation
```

## GCP Free Tier

This system is designed to use GCP's free tier:

- **Cloud Storage**: 
  - 5GB storage/month
  - 50,000 Class A operations/month
  - 1GB network egress/month
- **Vertex AI**:
  - Free tier includes basic vector search operations
  - See [GCP Free Tier](https://cloud.google.com/free) for current limits

## How It Works

### Analysis Phase (One-Time)

1. **Upload**: Scripts uploaded via web interface
2. **Parse**: System parses multiple script formats
3. **Store**: Files stored in GCS bucket (free tier)
4. **Embed**: Creates embeddings using sentence transformers
5. **RAG Query**: Uses semantic similarity to discover patterns
6. **Extract**: Analyzes speech patterns, emotions, conflict handling
7. **Generate**: Creates static Python rules class

### Runtime Phase (Production)

- Load static Python personality class
- No database queries needed
- Fast rule evaluation (~1-5ms)
- Use in chatbot system prompts

## API Endpoints

- `GET /` - Serve HTML frontend
- `POST /api/upload` - Upload script files
- `POST /api/extract` - Extract personality patterns
- `POST /api/query` - Query RAG system
- `GET /api/list-characters` - List processed characters
- `GET /api/download/<filename>` - Download generated files
- `GET /api/health` - Health check

## Configuration

Edit `config.py` to customize:

- GCP project ID
- Region (use `us-central1` for free tier)
- Upload limits
- Model settings

## Troubleshooting

### "GCP not configured" Error

- Verify `GOOGLE_APPLICATION_CREDENTIALS` environment variable is set
- Check that credentials file exists and is valid
- Run `python gcp_vertex_rag.py` to test GCP setup

### "No dialogue entries loaded"

- Check script file format matches expected format
- Verify character name matches exactly (case-sensitive)
- Ensure files have correct extensions (`.txt`, `.json`, `.csv`)

### Import Errors

```bash
pip install --upgrade -r requirements.txt
```

### GCP API Errors

- Verify APIs are enabled: `gcloud services list --enabled`
- Check service account has correct permissions
- Ensure project has billing enabled (even with free tier)

## Security Notes

- **Never commit** `gcp-credentials.json` to git
- Add `gcp-credentials.json` to `.gitignore`
- Rotate service account keys periodically
- Use IAM roles with least privilege

## Example Output

### Generated Python Rules Class

```python
class TonyStarkPersonality:
    def __init__(self):
        self.avg_sentence_length = 15.2
        self.formality_level = "informal"
        self.conflict_style = "humor"
        # ... more attributes
    
    def get_response_style(self, emotion, situation):
        # Returns personality guidelines
```

### JSON Patterns

```json
{
  "character_name": "Tony Stark",
  "total_dialogue_lines": 847,
  "speech_patterns": {
    "sentence_length": {"mean": 15.2, ...},
    "formality_level": "informal",
    ...
  },
  ...
}
```

## Next Steps

1. **Integrate with Chatbot**: Use generated rules in your LLM system prompts
2. **Batch Processing**: Process multiple characters
3. **Refine Rules**: Edit generated Python classes for customization
4. **Deploy to Cloud**: Deploy Flask app to Cloud Run or App Engine

## Support

For issues or questions:
- Check `Reference Code/README (2).md` for detailed documentation
- Review `Reference Code/ARCHITECTURE.md` for system design
- See `Reference Code/QUICKSTART.md` for quick examples

## License

This implementation is provided as-is for educational and development purposes.







