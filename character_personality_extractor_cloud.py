"""
Character Personality Extraction from Scripts using RAG Embeddings (Cloud Version)
===================================================================================

This is a cloud-adapted version that works with GCP Vertex AI RAG.
Based on the reference implementation but optimized for cloud processing.
"""

import json
import os
import re
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass
from collections import defaultdict, Counter

import numpy as np
from sentence_transformers import SentenceTransformer
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer


@dataclass
class DialogueEntry:
    """Represents a single line of character dialogue with context."""
    character: str
    text: str
    scene_context: str
    emotional_context: str
    other_characters: List[str]
    episode_season: str
    situation_type: str  # conflict, casual, romantic, etc.


class CharacterPersonalityExtractorCloud:
    """Extract personality patterns from character scripts using embeddings (Cloud version)."""
    
    def __init__(self, character_name: str, model_name: str = 'all-MiniLM-L6-v2'):
        self.character_name = character_name
        self.model = SentenceTransformer(model_name)
        self.dialogue_entries: List[DialogueEntry] = []
        self.embeddings: np.ndarray = None
        
    def load_scripts(self, scripts_path: str) -> None:
        """Load and parse character scripts from various formats."""
        print(f"Loading scripts for {self.character_name}...")
        
        if not os.path.exists(scripts_path):
            raise ValueError(f"Scripts path does not exist: {scripts_path}")
        
        # Handle both files and directories
        if os.path.isfile(scripts_path):
            script_files = [os.path.basename(scripts_path)]
            scripts_path = os.path.dirname(scripts_path)
        else:
            script_files = [f for f in os.listdir(scripts_path) 
                           if f.endswith(('.txt', '.json', '.csv'))]
        
        for file in script_files:
            file_path = os.path.join(scripts_path, file)
            
            if not os.path.exists(file_path):
                continue
                
            try:
                if file.endswith('.txt'):
                    self._parse_text_script(file_path)
                elif file.endswith('.json'):
                    self._parse_json_script(file_path)
                elif file.endswith('.csv'):
                    self._parse_csv_script(file_path)
            except Exception as e:
                print(f"Error parsing {file}: {e}")
                continue
                
        print(f"Loaded {len(self.dialogue_entries)} dialogue entries")
        
    def _is_youtube_transcript(self, content: str) -> bool:
        """Detect if this is a YouTube transcript format."""
        # YouTube transcripts typically start with Video ID, Title, URL
        youtube_markers = [
            r'Video ID:',
            r'Title:',
            r'URL:.*youtube\.com',
            r'Downloaded:.*\d{4}-\d{2}-\d{2}'
        ]
        matches = sum(1 for pattern in youtube_markers if re.search(pattern, content, re.IGNORECASE))
        return matches >= 2  # At least 2 markers indicate YouTube transcript
    
    def _parse_youtube_transcript(self, file_path: str, content: str) -> None:
        """Parse YouTube transcript format (no character labels, mostly dialogue)."""
        # Extract metadata
        video_id_match = re.search(r'Video ID:\s*(.+)', content, re.IGNORECASE)
        title_match = re.search(r'Title:\s*(.+)', content, re.IGNORECASE)
        url_match = re.search(r'URL:\s*(.+)', content, re.IGNORECASE)
        
        episode_info = title_match.group(1).strip() if title_match else os.path.basename(file_path).replace('.txt', '')
        
        # Remove metadata header (everything before first "----" or empty line after metadata)
        lines = content.split('\n')
        content_start = 0
        for i, line in enumerate(lines):
            if line.strip() == '----' or (i > 5 and line.strip() and not any(marker in line for marker in ['Video ID:', 'Title:', 'URL:', 'Downloaded:'])):
                content_start = i + 1
                break
        
        # Get dialogue content (everything after metadata)
        dialogue_content = '\n'.join(lines[content_start:]).strip()
        
        # Remove stage directions and sound effects [Music], [Applause], etc.
        dialogue_content = re.sub(r'\[.*?\]', '', dialogue_content)
        
        # Split into dialogue segments
        # Method 1: Lines with >> markers (speaker indicators)
        if '>>' in dialogue_content:
            segments = re.split(r'\n\s*>>', dialogue_content)
            other_chars = ['Interviewer', 'Host']
        else:
            # Method 2: For line-by-line transcripts, split by lines and group consecutive lines
            # YouTube transcripts often have each sentence/thought on a new line
            lines = [line.strip() for line in dialogue_content.split('\n') if line.strip() and len(line.strip()) > 10]
            
            # Group consecutive lines into segments (dialogue blocks)
            segments = []
            current_segment = []
            
            for line in lines:
                # Skip obvious non-dialogue lines
                if re.match(r'^\d+\s+\d+\s+\d+', line):  # Timestamps
                    if current_segment:
                        segments.append(' '.join(current_segment))
                        current_segment = []
                    continue
                
                # If line is very short and looks like intro/outro, break segment
                if len(line) < 20 and re.match(r'^(What\'?s up|Hey|Hello|Hi|Great|It is|Very good)', line, re.IGNORECASE):
                    if current_segment:
                        segments.append(' '.join(current_segment))
                        current_segment = []
                    # Don't add short greetings
                    if len(line) > 15:
                        current_segment.append(line)
                    continue
                
                # Add line to current segment
                current_segment.append(line)
                
                # Break segment if we hit a natural pause (very short line) or question
                if len(current_segment) >= 3 and (line.endswith('.') or line.endswith('!')):
                    segments.append(' '.join(current_segment))
                    current_segment = []
            
            # Add remaining segment
            if current_segment:
                segments.append(' '.join(current_segment))
            
            other_chars = []
        
        # Extract character dialogue
        # In YouTube transcripts, the character (Keke Palmer) is typically speaking most of the time
        # Interviewer questions are usually shorter, more direct questions
        
        for segment_idx, segment in enumerate(segments):
            segment = segment.strip()
            if not segment or len(segment) < 15:  # Skip very short segments
                continue
            
            # Remove >> markers if present
            segment = re.sub(r'^>>\s*', '', segment)
            segment = re.sub(r'\s+', ' ', segment)  # Normalize whitespace
            
            # Skip obvious interviewer questions (very short, only questions, greeting patterns)
            skip_patterns = [
                r'^(What\'?s up|Hey|Hello|Hi)\s+',
                r'^(How|Why|What|When|Where|Who)\s+',
                r'^(Great energy|It is good|Very good)\s+',
                r'^\d+\s+\d+\s+\d+',  # Timestamps like "3 2 1"
                r'^\[.*?\]\s*$',  # Only stage directions
            ]
            
            if any(re.match(pattern, segment, re.IGNORECASE) for pattern in skip_patterns):
                continue
            
            # Heuristics to identify if this is the character speaking:
            # 1. Long segments (character usually speaks more)
            # 2. Contains first person references (I, I'm, I've, me, my)
            # 3. Contains character name references
            # 4. Not obvious interviewer-only content
            
            is_character_speech = False
            
            # Default to including most content (in interviews, most dialogue is the guest)
            # Only exclude obvious interviewer-only segments
            
            # Check if it's likely the character speaking
            if len(segment) > 30:  # Longer segments likely character
                is_character_speech = True
            elif re.search(r'\b(I|I\'m|I\'ve|I\'d|me|my|myself|we|our)\b', segment, re.IGNORECASE):
                is_character_speech = True
            elif self.character_name.lower() in segment.lower():
                is_character_speech = True
            elif not segment.endswith('?'):  # Statements more likely character
                is_character_speech = True
            elif segment_idx > 2:  # After intro, most content is character
                is_character_speech = True
            else:
                # Default: include it (better to have more data)
                is_character_speech = True
            
            # Include the segment if it's likely character speech
            if is_character_speech:
                entry = DialogueEntry(
                    character=self.character_name,
                    text=segment,
                    scene_context=f"Segment {segment_idx}",
                    emotional_context=self._detect_emotional_context(segment),
                    other_characters=other_chars[:3] if other_chars else [],
                    episode_season=episode_info,
                    situation_type=self._classify_situation(segment, other_chars)
                )
                self.dialogue_entries.append(entry)
    
    def _parse_text_script(self, file_path: str) -> None:
        """Parse text-based script files (common format: CHARACTER: dialogue or YouTube transcript)."""
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check if this is a YouTube transcript format
        if self._is_youtube_transcript(content):
            self._parse_youtube_transcript(file_path, content)
            return
            
        # Otherwise, parse as traditional script format
        episode_info = os.path.basename(file_path).replace('.txt', '')
        scenes = re.split(r'\n\s*SCENE\s+\d+|FADE IN:|INT\.|EXT\.', content)
        
        for scene_idx, scene in enumerate(scenes):
            if not scene.strip():
                continue
                
            dialogue_pattern = r'^([A-Z][A-Z\s]+):\s*(.+?)(?=\n[A-Z][A-Z\s]+:|$)'
            matches = re.findall(dialogue_pattern, scene, re.MULTILINE | re.DOTALL)
            
            for speaker, text in matches:
                speaker = speaker.strip()
                text = re.sub(r'\s+', ' ', text.strip())
                
                if speaker.upper() == self.character_name.upper() and len(text) > 10:
                    other_chars = [m[0].strip() for m in matches if m[0].strip().upper() != self.character_name.upper()]
                    
                    entry = DialogueEntry(
                        character=self.character_name,
                        text=text,
                        scene_context=f"Scene {scene_idx}",
                        emotional_context=self._detect_emotional_context(text),
                        other_characters=other_chars[:3],
                        episode_season=episode_info,
                        situation_type=self._classify_situation(text, other_chars)
                    )
                    self.dialogue_entries.append(entry)
                    
    def _parse_json_script(self, file_path: str) -> None:
        """Parse JSON script format."""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        for scene_idx, scene in enumerate(data.get('scenes', [])):
            other_chars = []
            character_lines = []
            
            for line in scene.get('dialogue', []):
                if line.get('character', '').upper() == self.character_name.upper():
                    character_lines.append(line['text'])
                else:
                    other_chars.append(line.get('character', ''))
                    
            for text in character_lines:
                if len(text) > 10:
                    entry = DialogueEntry(
                        character=self.character_name,
                        text=text,
                        scene_context=scene.get('context', f"Scene {scene_idx}"),
                        emotional_context=self._detect_emotional_context(text),
                        other_characters=list(set(other_chars))[:3],
                        episode_season=data.get('episode', 'Unknown'),
                        situation_type=self._classify_situation(text, other_chars)
                    )
                    self.dialogue_entries.append(entry)
                    
    def _parse_csv_script(self, file_path: str) -> None:
        """Parse CSV script format."""
        try:
            df = pd.read_csv(file_path)
            character_df = df[df['character'].str.upper() == self.character_name.upper()]
            
            for _, row in character_df.iterrows():
                dialogue_text = str(row.get('dialogue', ''))
                if len(dialogue_text) > 10:
                    other_chars = str(row.get('other_characters', '')).split(',')
                    entry = DialogueEntry(
                        character=self.character_name,
                        text=dialogue_text,
                        scene_context=str(row.get('scene', 'Unknown')),
                        emotional_context=self._detect_emotional_context(dialogue_text),
                        other_characters=[c.strip() for c in other_chars if c.strip()],
                        episode_season=str(row.get('episode', 'Unknown')),
                        situation_type=self._classify_situation(dialogue_text, other_chars)
                    )
                    self.dialogue_entries.append(entry)
        except Exception as e:
            print(f"Error parsing CSV {file_path}: {e}")
                
    def _detect_emotional_context(self, text: str) -> str:
        """Detect emotional tone from dialogue text."""
        text_lower = text.lower()
        
        if any(word in text_lower for word in ['!', 'damn', 'hell', 'angry', 'furious']):
            return 'angry'
        elif any(word in text_lower for word in ['?', 'what', 'how', 'why', 'confused']):
            return 'questioning'
        elif any(word in text_lower for word in ['haha', 'funny', 'joke', 'laugh']):
            return 'humorous'
        elif any(word in text_lower for word in ['sorry', 'apologize', 'my fault']):
            return 'apologetic'
        elif any(word in text_lower for word in ['love', 'care', 'feel']):
            return 'emotional'
        else:
            return 'neutral'
            
    def _classify_situation(self, text: str, other_characters: List[str]) -> str:
        """Classify the type of situation based on context."""
        text_lower = text.lower()
        
        if any(word in text_lower for word in ['fight', 'argue', 'wrong', 'disagree']):
            return 'conflict'
        elif any(word in text_lower for word in ['love', 'date', 'kiss', 'feel about']):
            return 'romantic'
        elif any(word in text_lower for word in ['work', 'plan', 'need to', 'should']):
            return 'professional'
        elif len(other_characters) > 2:
            return 'group'
        else:
            return 'casual'
            
    def create_embeddings(self) -> None:
        """Create embeddings for all dialogue entries."""
        print("Creating embeddings...")
        
        if not self.dialogue_entries:
            raise ValueError("No dialogue entries to create embeddings for")
        
        texts = [
            f"{entry.text} [Context: {entry.emotional_context}, {entry.situation_type}]"
            for entry in self.dialogue_entries
        ]
        
        self.embeddings = self.model.encode(texts, show_progress_bar=True)
        print(f"Created embeddings: {self.embeddings.shape}")
        
    def query_similar_dialogue(self, query: str, top_k: int = 10) -> List[Tuple[DialogueEntry, float]]:
        """Query for similar dialogue based on semantic similarity."""
        if self.embeddings is None:
            self.create_embeddings()
            
        query_embedding = self.model.encode([query])[0]
        
        similarities = np.dot(self.embeddings, query_embedding) / (
            np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(query_embedding)
        )
        
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        results = [
            (self.dialogue_entries[idx], float(similarities[idx]))
            for idx in top_indices
        ]
        
        return results
        
    def extract_speech_patterns(self) -> Dict[str, Any]:
        """Extract speech patterns and linguistic quirks."""
        print("Extracting speech patterns...")
        
        patterns = {
            'sentence_length': self._analyze_sentence_length(),
            'vocabulary_complexity': self._analyze_vocabulary(),
            'common_phrases': self._extract_common_phrases(),
            'punctuation_style': self._analyze_punctuation(),
            'question_frequency': self._analyze_questions(),
            'exclamation_frequency': self._analyze_exclamations(),
            'curse_frequency': self._analyze_profanity(),
            'formality_level': self._analyze_formality()
        }
        
        return patterns
        
    def _analyze_sentence_length(self) -> Dict[str, float]:
        """Analyze average sentence length."""
        lengths = []
        for entry in self.dialogue_entries:
            sentences = re.split(r'[.!?]+', entry.text)
            for sent in sentences:
                words = sent.strip().split()
                if words:
                    lengths.append(len(words))
                    
        return {
            'mean': float(np.mean(lengths)) if lengths else 0,
            'median': float(np.median(lengths)) if lengths else 0,
            'std': float(np.std(lengths)) if lengths else 0
        }
        
    def _analyze_vocabulary(self) -> Dict[str, Any]:
        """Analyze vocabulary complexity."""
        all_words = []
        for entry in self.dialogue_entries:
            words = re.findall(r'\b\w+\b', entry.text.lower())
            all_words.extend(words)
            
        unique_words = set(all_words)
        word_counts = Counter(all_words)
        
        lexical_diversity = len(unique_words) / len(all_words) if all_words else 0
        
        return {
            'total_words': len(all_words),
            'unique_words': len(unique_words),
            'lexical_diversity': lexical_diversity,
            'most_common_words': word_counts.most_common(20)
        }
        
    def _extract_common_phrases(self, n_phrases: int = 30) -> List[Tuple[str, int]]:
        """Extract commonly used phrases."""
        if not self.dialogue_entries:
            return []
            
        texts = [entry.text for entry in self.dialogue_entries]
        
        try:
            vectorizer = TfidfVectorizer(ngram_range=(2, 4), max_features=100, stop_words='english')
            tfidf_matrix = vectorizer.fit_transform(texts)
            
            feature_names = vectorizer.get_feature_names_out()
            avg_scores = tfidf_matrix.mean(axis=0).A1
            top_indices = avg_scores.argsort()[-n_phrases:][::-1]
            
            common_phrases = [
                (feature_names[idx], int(np.sum(tfidf_matrix[:, idx].toarray() > 0)))
                for idx in top_indices
            ]
            
            return common_phrases
        except Exception as e:
            print(f"Error extracting phrases: {e}")
            return []
        
    def _analyze_punctuation(self) -> Dict[str, float]:
        """Analyze punctuation usage."""
        total_lines = len(self.dialogue_entries)
        if total_lines == 0:
            return {'ellipsis_frequency': 0, 'dash_frequency': 0, 'comma_per_line': 0}
        
        patterns = {
            'ellipsis_frequency': sum(1 for e in self.dialogue_entries if '...' in e.text) / total_lines,
            'dash_frequency': sum(1 for e in self.dialogue_entries if '--' in e.text or '—' in e.text) / total_lines,
            'comma_per_line': sum(e.text.count(',') for e in self.dialogue_entries) / total_lines,
        }
        
        return patterns
        
    def _analyze_questions(self) -> float:
        """Calculate frequency of questions."""
        total_lines = len(self.dialogue_entries)
        question_lines = sum(1 for e in self.dialogue_entries if '?' in e.text)
        return question_lines / total_lines if total_lines > 0 else 0
        
    def _analyze_exclamations(self) -> float:
        """Calculate frequency of exclamations."""
        total_lines = len(self.dialogue_entries)
        exclamation_lines = sum(1 for e in self.dialogue_entries if '!' in e.text)
        return exclamation_lines / total_lines if total_lines > 0 else 0
        
    def _analyze_profanity(self) -> float:
        """Calculate frequency of strong language."""
        profanity_words = {'damn', 'hell', 'crap', 'shit', 'fuck', 'ass'}
        total_lines = len(self.dialogue_entries)
        profanity_lines = sum(
            1 for e in self.dialogue_entries 
            if any(word in e.text.lower() for word in profanity_words)
        )
        return profanity_lines / total_lines if total_lines > 0 else 0
        
    def _analyze_formality(self) -> str:
        """Determine overall formality level."""
        informal_markers = ['gonna', 'wanna', 'gotta', 'yeah', 'nah', "ain't"]
        formal_markers = ['therefore', 'however', 'nevertheless', 'furthermore']
        
        informal_count = sum(
            1 for e in self.dialogue_entries 
            if any(marker in e.text.lower() for marker in informal_markers)
        )
        
        formal_count = sum(
            1 for e in self.dialogue_entries 
            if any(marker in e.text.lower() for marker in formal_markers)
        )
        
        if informal_count > formal_count * 2:
            return 'very_informal'
        elif informal_count > formal_count:
            return 'informal'
        elif formal_count > informal_count:
            return 'formal'
        else:
            return 'neutral'
            
    def extract_emotional_responses(self) -> Dict[str, List[str]]:
        """Extract how character responds in different emotional contexts."""
        print("Extracting emotional response patterns...")
        
        emotional_patterns = defaultdict(list)
        
        for entry in self.dialogue_entries:
            emotion = entry.emotional_context
            first_sentence = re.split(r'[.!?]', entry.text)[0].strip()
            if first_sentence:
                emotional_patterns[emotion].append(first_sentence)
                
        for emotion in emotional_patterns:
            if len(emotional_patterns[emotion]) > 10:
                emotional_patterns[emotion] = emotional_patterns[emotion][:10]
                
        return dict(emotional_patterns)
        
    def extract_conflict_handling(self) -> Dict[str, Any]:
        """Extract how character handles conflict."""
        print("Analyzing conflict handling...")
        
        conflict_dialogue = [e for e in self.dialogue_entries if e.situation_type == 'conflict']
        
        if not conflict_dialogue:
            return {'dominant_style': 'insufficient_data', 'style_frequencies': {}, 'examples': []}
            
        conflict_texts = [e.text for e in conflict_dialogue]
        
        patterns = {
            'avoidance': sum(1 for t in conflict_texts if any(w in t.lower() for w in ['whatever', 'fine', 'forget it'])),
            'aggression': sum(1 for t in conflict_texts if any(w in t.lower() for w in ['damn', 'hell', 'shut up'])),
            'humor': sum(1 for t in conflict_texts if any(w in t.lower() for w in ['joke', 'funny', 'kidding'])),
            'rational': sum(1 for t in conflict_texts if any(w in t.lower() for w in ['think', 'understand', 'reasonable']))
        }
        
        dominant_style = max(patterns, key=patterns.get) if patterns else 'unknown'
        
        return {
            'dominant_style': dominant_style,
            'style_frequencies': patterns,
            'examples': [e.text for e in conflict_dialogue[:5]]
        }
        
    def cluster_personality_traits(self, n_clusters: int = 5) -> Dict[str, List[str]]:
        """Cluster dialogue to identify distinct personality traits."""
        print("Clustering personality traits...")
        
        if not self.dialogue_entries:
            return {}
        
        if self.embeddings is None:
            self.create_embeddings()
        
        # Use fewer clusters if we don't have enough data
        n_clusters = min(n_clusters, len(self.dialogue_entries))
        if n_clusters < 2:
            return {}
            
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(self.embeddings)
        
        clustered_dialogue = defaultdict(list)
        for idx, cluster_id in enumerate(clusters):
            clustered_dialogue[f"trait_{cluster_id}"].append(self.dialogue_entries[idx].text)
            
        for cluster in clustered_dialogue:
            clustered_dialogue[cluster] = clustered_dialogue[cluster][:5]
            
        return dict(clustered_dialogue)
        
    def extract_personality_patterns(self) -> Dict[str, Any]:
        """Main method to extract all personality patterns."""
        print(f"\n{'='*60}")
        print(f"Extracting Personality Patterns for {self.character_name}")
        print(f"{'='*60}\n")
        
        if not self.dialogue_entries:
            return {'error': 'No dialogue entries loaded'}
            
        self.create_embeddings()
        
        patterns = {
            'character_name': self.character_name,
            'total_dialogue_lines': len(self.dialogue_entries),
            'speech_patterns': self.extract_speech_patterns(),
            'emotional_responses': self.extract_emotional_responses(),
            'conflict_handling': self.extract_conflict_handling(),
            'personality_clusters': self.cluster_personality_traits(),
            'metadata': {
                'unique_situations': len(set(e.situation_type for e in self.dialogue_entries)),
                'unique_emotions': len(set(e.emotional_context for e in self.dialogue_entries)),
                'episodes_analyzed': len(set(e.episode_season for e in self.dialogue_entries))
            }
        }
        
        return patterns
        
    def generate_static_rules(self, patterns: Dict[str, Any], output_path: str = None) -> str:
        """Generate Python code for static personality rules based on extracted patterns."""
        print("\nGenerating static rule code...")
        
        speech = patterns['speech_patterns']
        emotions_raw = patterns['emotional_responses']
        conflict = patterns['conflict_handling']
        
        # FILTER: Clean emotional responses to prevent direct quoting
        # Limit quote length and number of examples to avoid chatbot copying
        MAX_QUOTE_LENGTH = 100  # Max characters per quote
        MAX_EXAMPLES_PER_EMOTION = 3  # Max examples per emotion category
        
        emotions_filtered = {}
        for emotion_type, examples in emotions_raw.items():
            if isinstance(examples, list):
                # Filter: only short quotes, limit count
                filtered_examples = []
                for ex in examples[:MAX_EXAMPLES_PER_EMOTION]:
                    if isinstance(ex, str):
                        # Truncate long quotes and add ellipsis
                        if len(ex) > MAX_QUOTE_LENGTH:
                            truncated = ex[:MAX_QUOTE_LENGTH].rsplit(' ', 1)[0] + "..."
                            filtered_examples.append(truncated)
                        else:
                            filtered_examples.append(ex)
                emotions_filtered[emotion_type] = filtered_examples
            else:
                emotions_filtered[emotion_type] = examples
        
        # Clean character name for Python class name
        class_name = patterns['character_name'].replace(" ", "").replace("-", "").replace("_", "")
        
        # Extract personality essence from patterns
        formality = speech['formality_level']
        question_freq = speech['question_frequency']
        exclamation_freq = speech['exclamation_frequency']
        
        # Determine energy level based on patterns
        if exclamation_freq > 0.1 or question_freq > 0.3:
            energy_level = "high_energy"
        elif question_freq > 0.2:
            energy_level = "moderate_energy"
        else:
            energy_level = "calm_energy"
        
        # Create style descriptors
        style_descriptors = []
        if formality == "very_informal":
            style_descriptors.append("very conversational and casual")
        if question_freq > 0.3:
            style_descriptors.append("tends to ask questions")
        if exclamation_freq > 0.05:
            style_descriptors.append("expressive and enthusiastic")
        
        personality_essence = ", ".join(style_descriptors) if style_descriptors else "conversational"
        
        code = f'''"""
Static Personality Rules for {patterns['character_name']}
Generated from {patterns['total_dialogue_lines']} dialogue entries

PHILOSOPHY: These rules capture HOW {patterns['character_name']} speaks (style, energy, essence).
Use them to generate NEW responses in this character's style, based on what the user asks.
Focus on the ESSENCE and ENERGY, not verbatim quotes.

⚠️  WARNING: Examples are for REFERENCE ONLY to understand style patterns.
Generate responses in this style, but answer the actual question asked.
"""

class {class_name}Personality:
    """Static personality rules - focuses on HOW they speak, not WHAT they say."""
    
    def __init__(self):
        # Personality Essence & Energy
        self.personality_essence = "{personality_essence}"
        self.energy_level = "{energy_level}"
        
        # Speech patterns (HOW they speak)
        self.avg_sentence_length = {speech['sentence_length']['mean']:.1f}
        self.formality_level = "{formality}"
        self.question_frequency = {speech['question_frequency']:.2f}
        self.exclamation_frequency = {speech['exclamation_frequency']:.2f}
        
        # Common phrase patterns (for "says stuff like..." style patterns)
        self.signature_phrases = {speech['common_phrases'][:10] if speech['common_phrases'] else []}
        
        # Emotional response patterns (FILTERED: short examples only, for style reference)
        # Use these to understand HOW they express emotions, not WHAT they say verbatim
        self.emotional_responses = {json.dumps(emotions_filtered, indent=8)}
        
        # Conflict handling
        self.conflict_style = "{conflict.get('dominant_style', 'unknown')}"
        
    def get_personality_essence(self) -> dict:
        """Get overall personality essence and energy."""
        return {{
            'essence': self.personality_essence,
            'energy_level': self.energy_level,
            'formality': self.formality_level,
            'speaking_style': f"{{self.personality_essence}} with {{self.energy_level}} energy"
        }}
    
    def get_response_style(self, emotion: str, situation: str) -> dict:
        """Get HOW they would respond (style, energy, patterns) - not WHAT they would say."""
        style = {{
            'personality_essence': self.personality_essence,
            'energy_level': self.energy_level,
            'sentence_length_target': self.avg_sentence_length,
            'formality': self.formality_level,
            'speaking_patterns': [],
            'add_question': emotion == 'questioning' and self.question_frequency > 0.3,
            'add_exclamation': emotion == 'angry' and self.exclamation_frequency > 0.2,
        }}
        
        # Add style patterns based on signature phrases
        if self.signature_phrases:
            top_phrases = [p[0] if isinstance(p, (list, tuple)) else p for p in self.signature_phrases[:3]]
            style['speaking_patterns'] = [f"tends to use phrases like '{{p}}'".format(p=p) for p in top_phrases]
        
        if situation == 'conflict':
            style['conflict_approach'] = self.conflict_style
            
        return style
        
    def should_use_phrase(self, context: str) -> str:
        """Suggest signature phrase if appropriate."""
        # Return random signature phrase based on context
        import random
        phrases = [p[0] if isinstance(p, tuple) else p for p in self.signature_phrases]
        return random.choice(phrases) if phrases and random.random() < 0.2 else ""
        
    def generate_response_guidelines(self, user_input: str, emotion: str, situation: str) -> str:
        """Generate guidelines for HOW to respond (style, energy, essence) - not WHAT to say."""
        essence = self.get_personality_essence()
        style = self.get_response_style(emotion, situation)
        
        guidelines = f"""
PERSONALITY ESSENCE & ENERGY:
- Overall essence: {{essence['essence']}}
- Energy level: {{essence['energy_level']}}
- Speaking style: {{essence['speaking_style']}}

HOW TO RESPOND (Style Guidelines):
- Formality: {{self.formality_level}}
- Sentence length: ~{{style['sentence_length_target']}} words
- Energy: {{style['energy_level']}}
- Emotional context: {{emotion}}
- Situation: {{situation}}

CRITICAL: Answer the user's question ("{{user_input}}") but respond in this character's style.
Generate a NEW response based on what was asked, but using:
- This energy level and essence
- These speaking patterns
- This formality level

⚠️  DO NOT copy examples verbatim. Use them to understand the STYLE, then generate NEW content.
"""
        
        if style.get('speaking_patterns'):
            guidelines += f"\\nSpeaking patterns (style reference):\\n"
            for pattern in style['speaking_patterns'][:3]:
                guidelines += f"- {{pattern}}\\n"
        
        if emotion in self.emotional_responses:
            examples = self.emotional_responses[emotion][:2]  # Limit to 2 examples
            guidelines += f"\\nStyle reference for {{emotion}} emotion (understand the ENERGY/STYLE, don't copy):\\n"
            for ex in examples:
                # Truncate if still too long
                ex_display = ex[:60] + "..." if len(ex) > 60 else ex
                guidelines += f"- Energy/style example: {{ex_display}}\\n"
                
        return guidelines
'''
        
        if output_path:
            with open(output_path, 'w') as f:
                f.write(code)
            print(f"Static rules saved to {output_path}")
            
        return code
        
    def save_analysis(self, patterns: Dict[str, Any], output_dir: str) -> None:
        """Save complete analysis to files."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save JSON patterns
        json_path = os.path.join(output_dir, f'{patterns["character_name"].replace(" ", "_")}_patterns.json')
        with open(json_path, 'w') as f:
            json.dump(patterns, f, indent=2)
        print(f"Patterns saved to {json_path}")
        
        # Save Python rules
        rules_path = os.path.join(output_dir, f'{patterns["character_name"].replace(" ", "_")}_rules.py')
        self.generate_static_rules(patterns, rules_path)
        
        # Save embeddings if available
        if self.embeddings is not None:
            embeddings_path = os.path.join(output_dir, f'{patterns["character_name"].replace(" ", "_")}_embeddings.npy')
            np.save(embeddings_path, self.embeddings)
            print(f"Embeddings saved to {embeddings_path}")
        
        print(f"\nAnalysis complete! Files saved to {output_dir}")

