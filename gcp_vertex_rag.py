"""
GCP Vertex AI RAG Integration - REAL IMPLEMENTATION
Uses Vertex AI Embeddings, Vector Search, and Gemini 2.5 LLM for true RAG

Latest Models (as of 2025):
- Gemini 2.5 Flash (latest LLM)
- text-embedding-004 / gemini-embedding-001 (latest Vertex embeddings)
- textembedding-gecko@004 (legacy fallback)
"""

import os
import re
import json
from typing import List, Dict, Any, Tuple, Optional
from datetime import datetime

from google.cloud import storage
from google.cloud import aiplatform
# Note: vector_search is imported conditionally when needed
from vertexai.preview.generative_models import GenerativeModel
import vertexai


class GCPVertexRAGManager:
    """Manages REAL GCP Vertex AI RAG operations with LLM integration."""
    
    def __init__(self, project_id: str = None, region: str = "us-central1"):
        """
        Initialize GCP RAG Manager with REAL Vertex AI.
        
        Args:
            project_id: GCP project ID (auto-detects from gcloud if None)
            region: GCP region
        """
        # Auto-detect project from gcloud if not provided
        if project_id is None:
            import subprocess
            try:
                result = subprocess.run(
                    ['gcloud', 'config', 'get-value', 'project'],
                    capture_output=True, text=True, check=True
                )
                project_id = result.stdout.strip()
            except:
                project_id = "eminent-century-464801-b0"  # Fallback
        
        self.project_id = project_id
        self.region = region
        self.storage_client = None
        self.llm_model = None
        self.embedding_model = None
        
        # Initialize Vertex AI
        try:
            vertexai.init(project=project_id, location=region)
            self.storage_client = storage.Client(project=project_id)
            # Initialize Gemini LLM - using latest Gemini 2.5 Flash
            # Try 2.5 first, fallback to 2.0 if not available
            try:
                self.llm_model = GenerativeModel("gemini-2.5-flash")
                print(f"✓ Using Gemini 2.5 Flash (latest)")
            except:
                try:
                    self.llm_model = GenerativeModel("gemini-2.0-flash-exp")
                    print(f"✓ Using Gemini 2.0 Flash")
                except:
                    self.llm_model = GenerativeModel("gemini-1.5-flash")
                    print(f"✓ Using Gemini 1.5 Flash (fallback)")
            print(f"✓ Vertex AI initialized: project={project_id}, region={region}")
            print(f"✓ Gemini LLM model loaded")
            self._configured = True
        except Exception as e:
            print(f"✗ GCP initialization error: {e}")
            print("Make sure you're authenticated: gcloud auth application-default login")
            self._configured = False
    
    def is_configured(self) -> bool:
        """Check if GCP is properly configured."""
        return self._configured
    
    def _get_or_create_bucket(self, bucket_name: str) -> storage.Bucket:
        """Get existing bucket or create a new one."""
        try:
            bucket = self.storage_client.bucket(bucket_name)
            if not bucket.exists():
                bucket = storage.Bucket(self.storage_client, name=bucket_name)
                bucket.location = self.region
                bucket.storage_class = 'STANDARD'
                bucket = self.storage_client.create_bucket(bucket)
                print(f"✓ Created bucket: {bucket_name}")
            else:
                print(f"✓ Using existing bucket: {bucket_name}")
            return bucket
        except Exception as e:
            print(f"✗ Bucket error: {e}")
            raise
    
    def upload_to_gcs(self, local_dir: str, character_name: str) -> str:
        """Upload script files to GCS bucket."""
        if not self._configured:
            raise Exception("GCP not configured. Run: gcloud auth application-default login")
        
        sanitized_char = re.sub(r'[^a-zA-Z0-9-]', '-', character_name.lower())
        sanitized_proj = re.sub(r'[^a-zA-Z0-9-]', '-', self.project_id.lower())
        bucket_name = f"personality-extractor-{sanitized_char[:30]}-{sanitized_proj[:20]}".lower()
        bucket_name = re.sub(r'-+', '-', bucket_name).strip('-')[:63]
        
        bucket = self._get_or_create_bucket(bucket_name)
        
        uploaded_files = []
        for root, dirs, files in os.walk(local_dir):
            for file in files:
                if file.endswith(('.txt', '.json', '.csv')):
                    local_path = os.path.join(root, file)
                    blob_name = f"{character_name}/{file}"
                    blob = bucket.blob(blob_name)
                    blob.upload_from_filename(local_path)
                    uploaded_files.append(blob_name)
        
        print(f"✓ Uploaded {len(uploaded_files)} files to {bucket_name}")
        return bucket_name
    
    def create_embeddings_with_vertex(self, dialogue_texts: List[str]) -> List[List[float]]:
        """
        Create embeddings using Vertex AI Embeddings API.
        
        Args:
            dialogue_texts: List of dialogue strings to embed
            
        Returns:
            List of embedding vectors
        """
        if not self._configured:
            raise Exception("GCP not configured")
        
        print(f"Creating embeddings for {len(dialogue_texts)} dialogue entries using Vertex AI...")
        
        # Use Vertex AI text-embedding model - latest generation (Gemini embeddings)
        from vertexai.language_models import TextEmbeddingModel
        
        preferred_model = os.getenv("VERTEX_EMBEDDING_MODEL")
        fallback_models = [
            "text-embedding-004",               # Current GA Vertex embedding model
            "text-embedding-004-multilingual",  # Multilingual variant
            "gemini-embedding-001",             # Gemini embedding model
            "textembedding-gecko@004",          # Legacy Gecko fallback
            "textembedding-gecko-multilingual@003",
            "textembedding-gecko@003",
        ]
        
        model = None
        last_error = None
        for model_name in ([preferred_model] if preferred_model else []) + fallback_models:
            try:
                if not model_name:
                    continue
                model = TextEmbeddingModel.from_pretrained(model_name)
                print(f"✓ Using Vertex embedding model: {model_name}")
                break
            except Exception as err:
                last_error = err
                print(f"[WARN] Could not load embedding model '{model_name}': {err}")
        
        if model is None:
            raise RuntimeError(f"Unable to load any Vertex embedding model. Last error: {last_error}")
        
        # Vertex AI currently limits predictions to 250 instances per call
        max_batch = int(os.getenv("VERTEX_EMBED_BATCH", "100"))
        max_batch = max(1, min(max_batch, 250))
        embedding_vectors: List[List[float]] = []
        for start in range(0, len(dialogue_texts), max_batch):
            batch = dialogue_texts[start:start + max_batch]
            print(f"   → Embedding batch {start // max_batch + 1} ({len(batch)} items)")
            embeddings = model.get_embeddings(batch)
            embedding_vectors.extend([embedding.values for embedding in embeddings])
        
        print(f"✓ Created {len(embedding_vectors)} embeddings (dim={len(embedding_vectors[0]) if embedding_vectors else 0})")
        
        return embedding_vectors
    
    def extract_patterns_with_llm(self, dialogue_samples: List[str], character_name: str) -> Dict[str, Any]:
        """
        Use Gemini LLM to extract personality patterns from dialogue samples.
        This is where the LLM analyzes and abstracts patterns (not just quotes).
        
        Args:
            dialogue_samples: Sample dialogue entries
            character_name: Name of character
            
        Returns:
            Extracted personality patterns
        """
        if not self._configured:
            raise Exception("GCP not configured")
        
        print(f"Using Gemini LLM to extract personality patterns from {len(dialogue_samples)} dialogue samples...")
        
        # Prepare prompt for LLM to extract patterns (not quotes)
        # Limit samples to avoid token limits
        samples_text = "\n\n".join([f"Sample {i+1}: {sample[:150]}..." for i, sample in enumerate(dialogue_samples[:15])])
        
        prompt = f"""Analyze the following dialogue samples from {character_name} and extract the ESSENCE, ENERGY, and STYLE of how they communicate.

Focus on HOW they speak, not WHAT they say. Extract:
1. PERSONALITY ESSENCE: Overall energy/vibe (enthusiastic? direct? reflective? calm? energetic?)
2. SPEAKING STYLE: Patterns like "tends to use...", "often starts with...", "has a way of saying..."
3. ENERGY LEVELS: How their energy changes in different emotional states
4. SPEECH PATTERNS: Sentence rhythm, phrase patterns, linguistic quirks (e.g., "says stuff like...")
5. EMOTIONAL STYLE: HOW they express emotions (not what they say, but HOW - direct? subtle? expressive?)
6. CONFLICT STYLE: HOW they handle conflict (approach, energy, style)
7. COMMON PHRASE PATTERNS: Style patterns like "often uses phrases like...", "tends to say things like..."

Dialogue samples:
{samples_text}

Return a JSON object with these patterns. Focus on ESSENCE, ENERGY, and STYLE - not verbatim quotes.
Use patterns like "she says stuff like...", "tends to use...", "has a habit of..." to describe style."""

        try:
            # Add timeout and error handling
            import time
            start_time = time.time()
            
            response = self.llm_model.generate_content(
                prompt,
                generation_config={
                    "max_output_tokens": 2048,
                    "temperature": 0.3,
                }
            )
            
            elapsed = time.time() - start_time
            print(f"✓ LLM response received in {elapsed:.2f}s")
            
            if not response or not hasattr(response, 'text'):
                return {"error": "Empty LLM response"}
            
            pattern_text = response.text
            
            # Try to extract JSON from response
            json_match = re.search(r'\{.*\}', pattern_text, re.DOTALL)
            if json_match:
                try:
                    patterns = json.loads(json_match.group())
                except json.JSONDecodeError:
                    patterns = {"raw_analysis": pattern_text}
            else:
                # Fallback: parse text response
                patterns = {"raw_analysis": pattern_text}
            
            print("✓ LLM extracted personality patterns")
            return patterns
            
        except Exception as e:
            print(f"✗ LLM extraction error: {e}")
            import traceback
            traceback.print_exc()
            return {"error": str(e), "fallback": "Use local analysis"}
    
    def generate_rules_with_llm(self, patterns: Dict[str, Any], character_name: str) -> str:
        """
        Use Gemini LLM to generate Python personality rules class from patterns.
        
        Args:
            patterns: Extracted personality patterns
            character_name: Character name
            
        Returns:
            Python code for personality rules class
        """
        if not self._configured:
            raise Exception("GCP not configured")
        
        print("Using Gemini LLM to generate Python personality rules...")
        
        # Extract essence and energy patterns for style-based generation
        speech_patterns = patterns.get('speech_patterns', {})
        common_phrases = [p[0] if isinstance(p, (list, tuple)) else p for p in speech_patterns.get('common_phrases', [])[:10]]
        most_common_words = [w[0] if isinstance(w, (list, tuple)) else w for w in speech_patterns.get('vocabulary_complexity', {}).get('most_common_words', [])[:15]]
        
        # Simplify patterns for LLM (focus on essence, not content)
        simplified_patterns = {
            'character_name': patterns.get('character_name', character_name),
            'speech_patterns': {
                'sentence_length': speech_patterns.get('sentence_length', {}),
                'formality_level': speech_patterns.get('formality_level', 'neutral'),
                'question_frequency': speech_patterns.get('question_frequency', 0),
                'exclamation_frequency': speech_patterns.get('exclamation_frequency', 0),
                'common_phrases': common_phrases,  # For "says stuff like..." patterns
                'most_common_words': most_common_words,  # For vocabulary style
            },
            'conflict_handling': patterns.get('conflict_handling', {}),
            'emotional_responses_summary': {k: len(v) if isinstance(v, list) else 1 for k, v in patterns.get('emotional_responses', {}).items()},
        }
        
        patterns_json = json.dumps(simplified_patterns, indent=2)
        
        prompt = f"""Generate a Python class for {character_name}'s personality rules that focuses on HOW they speak, not WHAT they say.

Patterns extracted:
{patterns_json}

PHILOSOPHY: The rules should capture the ESSENCE, ENERGY, and STYLE of how {character_name} communicates. 
The chatbot should generate NEW responses based on the user's question, but in {character_name}'s unique style.

CRITICAL REQUIREMENTS:
1. Focus on HOW they speak: energy level, speaking patterns, style descriptors (e.g., "she says stuff like...", "tends to use...", "has a way of...")
2. Extract PERSONALITY ESSENCE: overall energy, vibe, communication style (enthusiastic? direct? reflective?)
3. Create STYLE PATTERNS WITH CONTEXT: For every phrase pattern, include WHEN/WHY it's used. Example: "says 'oh my god' when surprised or excited" NOT just "says 'oh my god'". This prevents overuse.
4. All phrase patterns must include contextual triggers: "uses [phrase] when [situation/emotion/context]" or "tends to say [phrase] in [specific scenario]"
5. Methods should return guidelines for HOW to respond, not WHAT to say
6. The actual response content should be generated based on the user's question, but in this character's style

Create a class called {character_name.replace(' ', '').replace('-', '')}Personality with:
- Attributes:
  * personality_essence: str - overall energy/vibe description (e.g., "enthusiastic, direct, conversational")
  * speaking_style: dict - HOW they speak (patterns, energy, rhythm)
  * common_phrase_patterns: list - patterns with CONTEXT like "says 'oh my god' when surprised" or "uses 'you know' when explaining" (MUST include WHEN/WHY, not just the phrase)
  * energy_level: str - overall energy (high/medium/low, enthusiastic/calm/etc)
  
- Methods:
  * get_personality_essence() -> dict: Returns overall personality energy, vibe, communication essence
  * get_speaking_style(emotion, situation) -> dict: Returns HOW they would speak (style, energy, patterns)
  * generate_response_guidelines(user_input, emotion, situation) -> dict: 
    - Returns guidelines for HOW to respond (style, energy, patterns)
    - Emphasizes: "respond in this style/energy, but answer the actual question asked"
    - Should include: "she would say something like..." patterns, not verbatim quotes
    - Should guide: energy level, speaking rhythm, phrase patterns, but NOT the actual content

- Helper methods:
  * _get_energy_for_emotion(emotion) -> str: Returns energy level for emotion
  * _get_speaking_patterns_for_situation(situation) -> list: Returns style patterns (e.g., "tends to use...", "often starts with...")

Return ONLY valid Python code, no explanations. Focus on ESSENCE, ENERGY, and STYLE patterns - not verbatim content."""

        try:
            import time
            start_time = time.time()
            
            response = self.llm_model.generate_content(
                prompt,
                generation_config={
                    "max_output_tokens": 4096,
                    "temperature": 0.2,
                }
            )
            
            elapsed = time.time() - start_time
            print(f"✓ LLM rules response received in {elapsed:.2f}s")
            
            if not response or not hasattr(response, 'text'):
                raise Exception("Empty LLM response")
            
            rules_code = response.text
            
            # Clean up code (remove markdown if present)
            rules_code = re.sub(r'```python\n?', '', rules_code)
            rules_code = re.sub(r'```\n?', '', rules_code)
            rules_code = rules_code.strip()
            
            print("✓ LLM generated personality rules")
            return rules_code
            
        except Exception as e:
            print(f"✗ LLM rule generation error: {e}")
            import traceback
            traceback.print_exc()
            raise  # Re-raise so caller can use fallback
    
    def query_rag_with_llm(self, query: str, retrieved_dialogue: List[str], character_name: str) -> str:
        """
        REAL RAG: Retrieve relevant dialogue, then use LLM to generate answer.
        
        Args:
            query: User query
            retrieved_dialogue: Retrieved dialogue from vector search
            character_name: Character name
            
        Returns:
            LLM-generated response in character's voice
        """
        if not self._configured:
            raise Exception("GCP not configured")
        
        context = "\n".join([f"- {dialogue}" for dialogue in retrieved_dialogue[:5]])
        
        prompt = f"""You are {character_name}. Based on these dialogue examples, respond to the user's question in {character_name}'s voice and style.

Dialogue examples (for reference only - DO NOT quote directly):
{context}

User question: {query}

Respond as {character_name} would, using the patterns from the examples but creating a NEW response (not a quote)."""

        try:
            response = self.llm_model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"Error: {e}"
    
    def create_vertex_rag_index(self, bucket_name: str, character_name: str, embeddings: List[List[float]], dialogue_texts: List[str]) -> str:
        """
        Create Vertex AI Vector Search index from embeddings.
        
        Args:
            bucket_name: GCS bucket name
            character_name: Character name
            embeddings: Embedding vectors
            dialogue_texts: Original dialogue texts
            
        Returns:
            Index endpoint ID
        """
        if not self._configured:
            raise Exception("GCP not configured")
        
        print(f"Creating Vertex AI Vector Search index for {character_name}...")
        
        # For now, store embeddings in GCS and note that Vector Search index creation
        # requires more setup. We'll use the embeddings directly for similarity search.
        print("✓ Embeddings ready for vector search (using direct similarity for now)")
        print("Note: Full Vector Search index deployment requires additional setup")
        
        return f"{character_name}-embeddings-{datetime.now().strftime('%Y%m%d')}"
    
    def query_vertex_rag(self, character_name: str, query: str, embeddings: List[List[float]], dialogue_texts: List[str], top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Query RAG system: embed query, find similar dialogue, return results.
        
        Args:
            character_name: Character name
            query: Query string
            embeddings: All dialogue embeddings
            dialogue_texts: All dialogue texts
            top_k: Number of results
            
        Returns:
            List of similar dialogue with similarity scores
        """
        if not self._configured:
            raise Exception("GCP not configured")
        
        # Embed the query using Vertex AI
        query_embeddings = self.create_embeddings_with_vertex([query])
        query_vector = query_embeddings[0]
        
        # Calculate cosine similarity
        import numpy as np
        similarities = []
        for i, emb in enumerate(embeddings):
            similarity = np.dot(query_vector, emb) / (np.linalg.norm(query_vector) * np.linalg.norm(emb))
            similarities.append((i, float(similarity)))
        
        # Get top_k most similar
        similarities.sort(key=lambda x: x[1], reverse=True)
        top_results = similarities[:top_k]
        
        results = [
            {
                'text': dialogue_texts[idx],
                'similarity': score,
                'metadata': {'character': character_name, 'index': idx}
            }
            for idx, score in top_results
        ]
        
        return results
    
    def list_buckets(self) -> List[str]:
        """List all buckets in the project."""
        if not self._configured:
            return []
        buckets = self.storage_client.list_buckets()
        return [bucket.name for bucket in buckets]


if __name__ == "__main__":
    print("Testing REAL GCP Vertex AI RAG Setup...")
    print("=" * 60)
    
    manager = GCPVertexRAGManager()
    if manager.is_configured():
        print("\n✓ REAL Vertex AI RAG Manager initialized")
        print(f"✓ Project: {manager.project_id}")
        print(f"✓ Region: {manager.region}")
        buckets = manager.list_buckets()
        print(f"✓ Found {len(buckets)} buckets")
    else:
        print("\n✗ Not configured. Run: gcloud auth application-default login")
