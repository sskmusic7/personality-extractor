"""
Configuration file for Character Personality Extraction System
GCP Settings for Vertex AI RAG
"""

import os
from dotenv import load_dotenv

load_dotenv()

# GCP Configuration
GCP_PROJECT_ID = os.getenv('GCP_PROJECT_ID', 'my first project')
GCP_REGION = os.getenv('GCP_REGION', 'us-central1')  # Free tier eligible region
GCP_CREDENTIALS_PATH = os.getenv('GOOGLE_APPLICATION_CREDENTIALS', None)

# Application Configuration
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'output'
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB
ALLOWED_EXTENSIONS = {'txt', 'json', 'csv', 'zip'}

# Flask Configuration
FLASK_HOST = os.getenv('FLASK_HOST', '0.0.0.0')
FLASK_PORT = int(os.getenv('FLASK_PORT', 5000))
FLASK_DEBUG = os.getenv('FLASK_DEBUG', 'True').lower() == 'true'

# Model Configuration
EMBEDDING_MODEL = 'all-MiniLM-L6-v2'  # Lightweight sentence transformer
CLUSTER_COUNT = 5  # Number of personality trait clusters

# GCS Bucket Configuration
BUCKET_STORAGE_CLASS = 'STANDARD'  # Free tier eligible
BUCKET_LOCATION = GCP_REGION







