# Quick Start Guide

Get your Character Personality Extraction System running in 5 minutes!

## Step 1: Install Dependencies (2 minutes)

```bash
cd "/Users/sskmusic/PErsonality Extractor"
pip install -r requirements.txt
```

## Step 2: Set Up GCP (2 minutes)

### Automated Setup (Recommended)

```bash
chmod +x setup_gcp.sh
./setup_gcp.sh
```

This will:
- Authenticate with GCP
- Set up your project "my first project"
- Enable required APIs
- Create service account and download credentials

### Manual Setup (If automated fails)

1. Install Google Cloud SDK: https://cloud.google.com/sdk/docs/install
2. Run: `gcloud auth login`
3. Run: `gcloud config set project "my first project"`
4. Enable APIs:
   ```bash
   gcloud services enable storage-component.googleapis.com
   gcloud services enable aiplatform.googleapis.com
   ```
5. Create service account (see README.md for full instructions)

## Step 3: Start the Server (30 seconds)

```bash
python app.py
```

You should see:
```
Starting Character Personality Extraction Server...
Open http://localhost:5000 in your browser
```

## Step 4: Use the Web Interface (1 minute)

1. Open browser: http://localhost:5000
2. Enter character name (e.g., "Tony Stark")
3. Drag and drop script files (.txt, .json, .csv, or .zip)
4. Click "Upload & Extract Personality"
5. Wait for processing (~30 seconds to 2 minutes)
6. Download results!

## Script File Formats

### Text Format (Simplest)
```
TONY STARK: This is my dialogue line.
PEPPER: Response here.
TONY STARK: Another dialogue line.
```

### JSON Format
```json
{
  "episode": "S01E01",
  "scenes": [
    {
      "context": "Office meeting",
      "dialogue": [
        {"character": "Tony Stark", "text": "Dialogue here"},
        {"character": "Pepper", "text": "Response here"}
      ]
    }
  ]
}
```

### CSV Format
```csv
character,dialogue,scene,episode,other_characters
Tony Stark,"Dialogue text",Scene Name,S01E01,"Pepper,Rhodes"
```

## Troubleshooting

### "GCP not configured"
```bash
export GOOGLE_APPLICATION_CREDENTIALS="$(pwd)/gcp-credentials.json"
```

### Import errors
```bash
pip install --upgrade -r requirements.txt
```

### Port already in use
Change port in `app.py` or `config.py`:
```python
app.run(debug=True, host='0.0.0.0', port=5001)
```

## Next Steps

1. **Test with sample data**: Use files from `Reference Code/` folder
2. **Upload your scripts**: Drag and drop your character scripts
3. **Download results**: Get Python rules and JSON patterns
4. **Integrate**: Use generated rules in your chatbot

## Free Tier Limits

- **GCS Storage**: 5GB/month free
- **GCS Operations**: 50,000 Class A operations/month free
- **Vertex AI**: Basic operations included in free tier

See [GCP Free Tier](https://cloud.google.com/free) for current limits.

## Need Help?

- Check `README.md` for detailed documentation
- Review `Reference Code/ARCHITECTURE.md` for system design
- See `Reference Code/example_usage.py` for code examples







