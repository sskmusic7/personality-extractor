"""
Flask Backend API for Character Personality Extraction System
Handles file uploads, processes scripts, and integrates with GCP Vertex AI RAG
"""

import os
import json
import tempfile
from flask import Flask, request, jsonify, send_file, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename
import zipfile
from datetime import datetime

from gcp_vertex_rag import GCPVertexRAGManager
from character_personality_extractor_cloud import CharacterPersonalityExtractorCloud

app = Flask(__name__, static_folder='static')
CORS(app)

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'txt', 'json', 'csv', 'zip'}
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB max file size

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs('output', exist_ok=True)

# Initialize GCP RAG Manager
gcp_manager = GCPVertexRAGManager()


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def index():
    """Serve the main HTML page."""
    return send_from_directory('static', 'index.html')


@app.route('/favicon.ico')
def favicon():
    """Serve favicon to prevent 404 errors."""
    return '', 204  # No content, but successful


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'gcp_configured': gcp_manager.is_configured()
    })


@app.route('/api/upload', methods=['POST'])
def upload_files():
    """Handle file uploads via drag-and-drop."""
    try:
        if 'files' not in request.files:
            return jsonify({'error': 'No files provided'}), 400
        
        files = request.files.getlist('files')
        character_name = request.form.get('character_name', 'Unknown Character')
        
        if not files or files[0].filename == '':
            return jsonify({'error': 'No files selected'}), 400
        
        # Create temporary directory for this extraction
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        temp_dir = os.path.join(UPLOAD_FOLDER, f'{secure_filename(character_name)}_{timestamp}')
        os.makedirs(temp_dir, exist_ok=True)
        
        uploaded_files = []
        
        for file in files:
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                filepath = os.path.join(temp_dir, filename)
                file.save(filepath)
                uploaded_files.append(filename)
                
                # Handle ZIP files
                if filename.endswith('.zip'):
                    with zipfile.ZipFile(filepath, 'r') as zip_ref:
                        zip_ref.extractall(temp_dir)
                    uploaded_files.append(f"Extracted from {filename}")
        
        if not uploaded_files:
            return jsonify({'error': 'No valid files uploaded'}), 400
        
        return jsonify({
            'success': True,
            'message': f'Uploaded {len(uploaded_files)} files',
            'files': uploaded_files,
            'temp_dir': temp_dir,
            'character_name': character_name
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/extract', methods=['POST'])
def extract_personality():
    """Extract personality patterns from uploaded scripts."""
    try:
        data = request.json
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400
            
        temp_dir = data.get('temp_dir')
        character_name = data.get('character_name', 'Unknown Character')
        
        if not temp_dir or not os.path.exists(temp_dir):
            return jsonify({'error': 'Invalid upload directory'}), 400
        
        print(f"[EXTRACT] Starting extraction for {character_name}...")
        
        # Initialize extractor
        extractor = CharacterPersonalityExtractorCloud(character_name)
        
        # Load scripts
        extractor.load_scripts(temp_dir)
        
        if not extractor.dialogue_entries:
            return jsonify({'error': 'No dialogue found in uploaded scripts'}), 400
        
        print(f"[EXTRACT] Loaded {len(extractor.dialogue_entries)} dialogue entries")
        
        # REAL RAG PIPELINE WITH VERTEX AI LLM
        
        # Step 1: Upload to GCS (if GCP configured)
        bucket_name = None
        vertex_embeddings = None
        
        if gcp_manager.is_configured():
            try:
                bucket_name = gcp_manager.upload_to_gcs(temp_dir, character_name)
            except Exception as e:
                print(f"[WARN] GCS upload failed: {e}")
        
        # Step 2: Create local embeddings for similarity search (limited set)
        print(f"[EXTRACT] Creating embeddings for {len(extractor.dialogue_entries)} entries...")
        extractor.create_embeddings()
        
        # Step 3: Use Vertex AI LLM to extract ABSTRACT patterns (not quotes!)
        patterns = None
        print(f"[EXTRACT] Extracting patterns...")
        
        # Always do local analysis first (faster, more reliable)
        local_patterns = extractor.extract_personality_patterns()
        patterns = local_patterns
        
        # Then enhance with LLM if available
        if gcp_manager.is_configured():
            try:
                print(f"[EXTRACT] Using Gemini LLM to extract abstract patterns...")
                # Get dialogue samples for LLM analysis (limit to avoid timeout)
                dialogue_samples = [entry.text for entry in extractor.dialogue_entries[:30]]
                
                # Use Gemini LLM to extract abstract personality patterns
                llm_patterns = gcp_manager.extract_patterns_with_llm(dialogue_samples, character_name)
                
                # Merge LLM abstract patterns with local stats
                patterns['llm_extracted_patterns'] = llm_patterns
                patterns['extraction_method'] = 'vertex_ai_llm'
                print(f"[EXTRACT] LLM patterns extracted successfully")
                
            except Exception as e:
                print(f"[WARN] LLM extraction failed: {e}, using local analysis only")
                import traceback
                traceback.print_exc()
                patterns['extraction_method'] = 'local_only'
        else:
            patterns['extraction_method'] = 'local_only'
        
        # Step 4: Use Vertex AI LLM to generate rules (not just templates)
        rules_code = None
        template_rules = None
        print(f"[EXTRACT] Generating rules...")
        
        # Always generate template rules first (fallback, but filtered to prevent quoting)
        template_rules = extractor.generate_static_rules(patterns)
        
        # PRIORITY: Try LLM-generated rules first (better for intelligent generation)
        if gcp_manager.is_configured():
            try:
                print(f"[EXTRACT] Using Gemini LLM to generate intelligent rules (prevents direct quoting)...")
                # Use Gemini LLM to generate Python rules class
                rules_code = gcp_manager.generate_rules_with_llm(patterns, character_name)
                print(f"[EXTRACT] ✅ LLM rules generated successfully (use these for chatbot integration)")
            except Exception as e:
                print(f"[WARN] LLM rule generation failed: {e}, using filtered template rules")
                import traceback
                traceback.print_exc()
                rules_code = template_rules
        else:
            print(f"[EXTRACT] GCP not configured, using filtered template rules (quotes truncated)")
            rules_code = template_rules
        
        # Step 5: Save everything locally first
        output_dir = os.path.join('output', f'{secure_filename(character_name)}_{datetime.now().strftime("%Y%m%d_%H%M%S")}')
        extractor.save_analysis(patterns, output_dir)
        
        # Step 6: Create Vertex AI embeddings for RAG queries and upload to GCS
        print(f"[EXTRACT] Creating Vertex AI embeddings...")
        if gcp_manager.is_configured() and len(extractor.dialogue_entries) > 0:
            try:
                dialogue_texts = [entry.text for entry in extractor.dialogue_entries]
                print(f"[EXTRACT] Creating Vertex AI embeddings for {len(dialogue_texts)} dialogue entries...")
                vertex_embeddings = gcp_manager.create_embeddings_with_vertex(dialogue_texts)
                print(f"[EXTRACT] Vertex AI embeddings created")
                
                # Create vector search index reference
                index_id = gcp_manager.create_vertex_rag_index(
                    bucket_name or "local", 
                    character_name, 
                    vertex_embeddings, 
                    dialogue_texts
                )
                patterns['vertex_index_id'] = index_id
                
                # Upload generated analysis files to GCS bucket
                if bucket_name:
                    try:
                        from google.cloud import storage
                        storage_client = storage.Client(project=gcp_manager.project_id)
                        bucket = storage_client.bucket(bucket_name)
                        
                        sanitized_name = secure_filename(character_name)
                        
                        # Upload patterns.json
                        patterns_file = os.path.join(output_dir, f'{sanitized_name}_patterns.json')
                        if os.path.exists(patterns_file):
                            patterns_blob = bucket.blob(f"{character_name}/analysis/{sanitized_name}_patterns.json")
                            patterns_blob.upload_from_filename(patterns_file)
                            print(f"✓ Uploaded patterns.json to GCS: {bucket_name}")
                        
                        # Upload LLM rules
                        rules_file = os.path.join(output_dir, f'{sanitized_name}_rules_llm.py')
                        if rules_code and os.path.exists(rules_file):
                            rules_blob = bucket.blob(f"{character_name}/analysis/{sanitized_name}_rules_llm.py")
                            rules_blob.upload_from_filename(rules_file)
                            print(f"✓ Uploaded LLM rules to GCS: {bucket_name}")
                        
                        # Upload embeddings
                        embeddings_file = os.path.join(output_dir, f'{sanitized_name}_embeddings.npy')
                        if os.path.exists(embeddings_file):
                            embeddings_blob = bucket.blob(f"{character_name}/analysis/{sanitized_name}_embeddings.npy")
                            embeddings_blob.upload_from_filename(embeddings_file)
                            print(f"✓ Uploaded embeddings to GCS: {bucket_name}")
                        
                    except Exception as e:
                        print(f"[WARN] GCS file upload failed: {e}")
            except Exception as e:
                print(f"[WARN] Vertex embeddings failed: {e}")
        
        # Save rules (prioritize LLM-generated, fallback to template)
        if rules_code:
            # Determine if these are LLM-generated or template rules
            is_llm_rules = gcp_manager.is_configured() and 'KekePPersonality' in rules_code or 'def _get_llm_emotional_style' in rules_code
            
            if is_llm_rules:
                # Save as LLM rules (preferred)
                rules_path = os.path.join(output_dir, f'{secure_filename(character_name)}_rules_llm.py')
                with open(rules_path, 'w') as f:
                    f.write(rules_code)
                print(f"✓ LLM-generated rules saved to {rules_path} (USE THESE for chatbot)")
                
                # Also save template as fallback (with warning)
                if template_rules:
                    template_path = os.path.join(output_dir, f'{secure_filename(character_name)}_rules_template.py')
                    with open(template_path, 'w') as f:
                        f.write(template_rules)
                    print(f"✓ Template rules saved to {template_path} (fallback only, quotes filtered)")
            else:
                # Template rules only (filtered)
                rules_path = os.path.join(output_dir, f'{secure_filename(character_name)}_rules.py')
                with open(rules_path, 'w') as f:
                    f.write(rules_code)
                print(f"✓ Filtered template rules saved to {rules_path} (quotes truncated to prevent copying)")
        
        # Prepare download paths (relative to output/)
        sanitized_name = secure_filename(character_name)
        rules_filename = f'{sanitized_name}_rules_llm.py'
        patterns_filename = f'{sanitized_name}_patterns.json'
        rules_path = os.path.join(output_dir, rules_filename)
        patterns_path = os.path.join(output_dir, patterns_filename)
        
        # Check which files exist
        download_paths = {}
        if os.path.exists(rules_path):
            download_paths['rules'] = os.path.relpath(rules_path, 'output')
        if os.path.exists(patterns_path):
            download_paths['patterns'] = os.path.relpath(patterns_path, 'output')
        
        print(f"[EXTRACT] Extraction complete! Returning results...")
        
        # Limit response size - don't send full patterns dict if too large
        response_patterns = patterns
        if isinstance(patterns, dict) and len(str(patterns)) > 100000:  # ~100KB limit
            # Send summary only
            response_patterns = {
                'character_name': patterns.get('character_name'),
                'extraction_method': patterns.get('extraction_method'),
                'total_dialogue_lines': patterns.get('total_dialogue_lines'),
                'pattern_categories': list(patterns.keys()) if isinstance(patterns, dict) else [],
                'note': 'Full patterns saved to file. Use download link to access complete data.'
            }
        
        return jsonify({
            'success': True,
            'character_name': character_name,
            'patterns': response_patterns,
            'rules_code': rules_code[:5000] if rules_code and len(rules_code) > 5000 else rules_code,  # Limit rules preview
            'output_dir': output_dir,
            'dialogue_count': len(extractor.dialogue_entries),
            'gcs_bucket': bucket_name,
            'gcp_configured': gcp_manager.is_configured(),
            'download_paths': download_paths  # Relative paths for downloads
        })
    
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print(f"[ERROR] Extract endpoint failed: {e}")
        print(error_trace)
        # Always return JSON, never empty response
        return jsonify({
            'error': str(e),
            'message': 'Extraction failed. Check server logs for details.',
            'traceback': error_trace if app.debug else None
        }), 500


@app.route('/api/query', methods=['POST'])
def query_rag():
    """
    REAL RAG Query: Retrieve relevant dialogue + LLM generates response.
    """
    try:
        data = request.json
        query = data.get('query')
        character_name = data.get('character_name')
        top_k = data.get('top_k', 10)
        
        if not query or not character_name:
            return jsonify({'error': 'Query and character_name required'}), 400
        
        if not gcp_manager.is_configured():
            return jsonify({'error': 'GCP not configured. Run: gcloud auth application-default login'}), 400
        
        # Load character's embeddings if available
        # For now, we'll need to reload or cache embeddings
        # In production, these would be in Vector Search index
        
        # REAL RAG FLOW:
        # 1. Embed query with Vertex AI
        # 2. Find similar dialogue (vector search)
        # 3. Use LLM to generate response based on retrieved context
        
        # For demo, we'll use a simplified version
        # In production, you'd query the Vector Search index
        
        # Get sample dialogue for retrieval (in production, from Vector Search)
        output_dir = 'output'
        character_dirs = [d for d in os.listdir(output_dir) 
                         if os.path.isdir(os.path.join(output_dir, d)) 
                         and character_name.lower() in d.lower()]
        
        if not character_dirs:
            return jsonify({'error': f'No processed data found for {character_name}'}), 404
        
        # Load embeddings and dialogue
        latest_dir = max(character_dirs, key=lambda d: os.path.getctime(os.path.join(output_dir, d)))
        embeddings_path = os.path.join(output_dir, latest_dir, f'{secure_filename(character_name)}_embeddings.npy')
        
        if os.path.exists(embeddings_path):
            import numpy as np
            embeddings = np.load(embeddings_path)
            
            # Load dialogue entries (would be from Vector Search in production)
            patterns_path = os.path.join(output_dir, latest_dir, f'{secure_filename(character_name)}_patterns.json')
            if os.path.exists(patterns_path):
                with open(patterns_path, 'r') as f:
                    patterns_data = json.load(f)
                
                # Use Vertex AI for RAG query
                dialogue_texts = [entry.get('text', '') for entry in patterns_data.get('dialogue_samples', [])]
                
                if dialogue_texts and len(embeddings) > 0:
                    # Query Vertex AI RAG
                    retrieved = gcp_manager.query_vertex_rag(
                        character_name, query, embeddings.tolist(), dialogue_texts, top_k
                    )
                    
                    # Use LLM to generate response
                    retrieved_texts = [r['text'] for r in retrieved]
                    llm_response = gcp_manager.query_rag_with_llm(query, retrieved_texts, character_name)
                    
                    return jsonify({
                        'success': True,
                        'query': query,
                        'retrieved_context': retrieved,
                        'llm_response': llm_response
                    })
        
        return jsonify({'error': 'Could not load character data for RAG query'}), 404
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/download/<path:filepath>')
def download_file(filepath):
    """Download generated files."""
    try:
        # Handle both formats: "output/dir/file" or "dir/file"
        if filepath.startswith('output/'):
            full_path = filepath
        else:
            full_path = os.path.join('output', filepath)
        
        # Security: ensure path is within output directory
        full_path = os.path.normpath(full_path)
        if not full_path.startswith('output'):
            return jsonify({'error': 'Invalid path'}), 400
        
        if os.path.exists(full_path) and os.path.isfile(full_path):
            return send_file(full_path, as_attachment=True)
        
        return jsonify({'error': 'File not found', 'path': full_path}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/list-characters', methods=['GET'])
def list_characters():
    """List all processed characters."""
    try:
        output_dir = 'output'
        characters = []
        
        if os.path.exists(output_dir):
            for item in os.listdir(output_dir):
                item_path = os.path.join(output_dir, item)
                if os.path.isdir(item_path):
                    # Look for patterns.json file
                    json_file = os.path.join(item_path, f'{item.split("_")[0]}_patterns.json')
                    if os.path.exists(json_file):
                        with open(json_file, 'r') as f:
                            data = json.load(f)
                            characters.append({
                                'name': data.get('character_name', item),
                                'dialogue_count': data.get('total_dialogue_lines', 0),
                                'output_dir': item,
                                'created': os.path.getctime(item_path)
                            })
        
        return jsonify({
            'success': True,
            'characters': sorted(characters, key=lambda x: x['created'], reverse=True)
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    print("Starting Character Personality Extraction Server...")
    print("Open http://localhost:5000 in your browser")
    # Increase timeout for long-running extraction tasks
    from werkzeug.serving import WSGIRequestHandler
    WSGIRequestHandler.timeout = 300  # 5 minutes
    debug_mode = os.getenv('FLASK_DEBUG', '0') == '1'
    app.run(
        debug=debug_mode,
        use_reloader=debug_mode,
        host='0.0.0.0',
        port=5000,
        threaded=True,
    )

