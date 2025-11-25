"""
Character Personality Extraction from Scripts using RAG Embeddings
================================================================

This implementation analyzes character dialogue and scripts to extract personality patterns
that can be converted into static rules for chatbot development.

Usage:
    extractor = CharacterPersonalityExtractor("character_name")
    extractor.load_scripts("path/to/scripts/")
    personality_rules = extractor.extract_personality_patterns()
"""

import json
import os
import re
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass, asdict
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


@dataclass
class PersonalityPattern:
    """Extracted personality pattern with confidence score."""
    category: str  # speech_style, emotional_response, conflict_handling, etc.
    pattern: str
    examples: List[str]
    confidence: float
    frequency: int


class CharacterPersonalityExtractor:
    """Extract personality patterns from character scripts using embeddings."""
    
    def __init__(self, character_name: str, model_name: str = 'all-MiniLM-L6-v2'):
        self.character_name = character_name
        self.model = SentenceTransformer(model_name)
        self.dialogue_entries: List[DialogueEntry] = []
        self.embeddings: np.ndarray = None
        self.personality_patterns: List[PersonalityPattern] = []
        
    def load_scripts(self, scripts_path: str) -> None:
        """Load and parse character scripts from various formats."""
        print(f"Loading scripts for {self.character_name}...")
        
        script_files = [f for f in os.listdir(scripts_path) 
                       if f.endswith(('.txt', '.json', '.csv'))]
        
        for file in script_files:
            file_path = os.path.join(scripts_path, file)
            
            if file.endswith('.txt'):
                self._parse_text_script(file_path)
            elif file.endswith('.json'):
                self._parse_json_script(file_path)
            elif file.endswith('.csv'):
                self._parse_csv_script(file_path)
                
        print(f"Loaded {len(self.dialogue_entries)} dialogue entries")
        
    def _parse_text_script(self, file_path: str) -> None:
        """Parse text-based script files (common format: CHARACTER: dialogue)."""
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
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
        df = pd.read_csv(file_path)
        character_df = df[df['character'].str.upper() == self.character_name.upper()]
        
        for _, row in character_df.iterrows():
            if len(str(row['dialogue'])) > 10:
                other_chars = str(row.get('other_characters', '')).split(',')
                entry = DialogueEntry(
                    character=self.character_name,
                    text=str(row['dialogue']),
                    scene_context=str(row.get('scene', 'Unknown')),
                    emotional_context=self._detect_emotional_context(str(row['dialogue'])),
                    other_characters=[c.strip() for c in other_chars if c.strip()],
                    episode_season=str(row.get('episode', 'Unknown')),
                    situation_type=self._classify_situation(str(row['dialogue']), other_chars)
                )
                self.dialogue_entries.append(entry)
                
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
        texts = [entry.text for entry in self.dialogue_entries]
        
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
        
    def _analyze_punctuation(self) -> Dict[str, float]:
        """Analyze punctuation usage."""
        total_lines = len(self.dialogue_entries)
        
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
            return {'pattern': 'insufficient_data', 'examples': []}
            
        conflict_texts = [e.text for e in conflict_dialogue]
        
        patterns = {
            'avoidance': sum(1 for t in conflict_texts if any(w in t.lower() for w in ['whatever', 'fine', 'forget it'])),
            'aggression': sum(1 for t in conflict_texts if any(w in t.lower() for w in ['damn', 'hell', 'shut up'])),
            'humor': sum(1 for t in conflict_texts if any(w in t.lower() for w in ['joke', 'funny', 'kidding'])),
            'rational': sum(1 for t in conflict_texts if any(w in t.lower() for w in ['think', 'understand', 'reasonable']))
        }
        
        dominant_style = max(patterns, key=patterns.get)
        
        return {
            'dominant_style': dominant_style,
            'style_frequencies': patterns,
            'examples': [e.text for e in conflict_dialogue[:5]]
        }
        
    def cluster_personality_traits(self, n_clusters: int = 5) -> Dict[str, List[str]]:
        """Cluster dialogue to identify distinct personality traits."""
        print("Clustering personality traits...")
        
        if self.embeddings is None:
            self.create_embeddings()
            
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
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
        emotions = patterns['emotional_responses']
        conflict = patterns['conflict_handling']
        
        code = f'''"""
Static Personality Rules for {patterns['character_name']}
Generated from {patterns['total_dialogue_lines']} dialogue entries
"""

class {patterns['character_name'].replace(" ", "")}Personality:
    """Static personality rules extracted from character analysis."""
    
    def __init__(self):
        # Speech patterns
        self.avg_sentence_length = {speech['sentence_length']['mean']:.1f}
        self.formality_level = "{speech['formality_level']}"
        self.question_frequency = {speech['question_frequency']:.2f}
        self.exclamation_frequency = {speech['exclamation_frequency']:.2f}
        
        # Common phrases (top 10)
        self.signature_phrases = {speech['common_phrases'][:10]}
        
        # Emotional response patterns
        self.emotional_responses = {json.dumps(emotions, indent=8)}
        
        # Conflict handling
        self.conflict_style = "{conflict.get('dominant_style', 'unknown')}"
        
    def get_response_style(self, emotion: str, situation: str) -> dict:
        """Get appropriate response style based on context."""
        style = {{
            'sentence_length_target': self.avg_sentence_length,
            'formality': self.formality_level,
            'add_question': emotion == 'questioning' and self.question_frequency > 0.3,
            'add_exclamation': emotion == 'angry' and self.exclamation_frequency > 0.2,
        }}
        
        if situation == 'conflict':
            style['conflict_approach'] = self.conflict_style
            
        return style
        
    def should_use_phrase(self, context: str) -> str:
        """Suggest signature phrase if appropriate."""
        # Return random signature phrase based on context
        import random
        phrases = [p[0] for p in self.signature_phrases]
        return random.choice(phrases) if phrases and random.random() < 0.2 else ""
        
    def generate_response_guidelines(self, user_input: str, emotion: str, situation: str) -> str:
        """Generate guidelines for response generation (for LLM system prompt)."""
        style = self.get_response_style(emotion, situation)
        
        guidelines = f"""
Personality Guidelines:
- Maintain {self.formality_level} formality level
- Target sentence length: {style['sentence_length_target']} words
- Conflict handling: {self.conflict_style}
- Emotional context: {emotion}
"""
        
        if emotion in self.emotional_responses:
            examples = self.emotional_responses[emotion][:3]
            guidelines += f"\\nExample responses in this emotional state:\\n"
            for ex in examples:
                guidelines += f"- {ex}\\n"
                
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
        json_path = os.path.join(output_dir, f'{self.character_name}_patterns.json')
        with open(json_path, 'w') as f:
            json.dump(patterns, f, indent=2)
        print(f"Patterns saved to {json_path}")
        
        # Save Python rules
        rules_path = os.path.join(output_dir, f'{self.character_name}_rules.py')
        self.generate_static_rules(patterns, rules_path)
        
        # Save embeddings
        embeddings_path = os.path.join(output_dir, f'{self.character_name}_embeddings.npy')
        np.save(embeddings_path, self.embeddings)
        print(f"Embeddings saved to {embeddings_path}")
        
        print(f"\nAnalysis complete! Files saved to {output_dir}")


def demonstrate_rag_queries(extractor: CharacterPersonalityExtractor):
    """Demonstrate how to use RAG queries during analysis phase."""
    print("\n" + "="*60)
    print("RAG Query Examples (Analysis Phase)")
    print("="*60 + "\n")
    
    queries = [
        "How does the character respond when angry?",
        "What does the character say in romantic situations?",
        "How does the character handle professional conflicts?",
        "What are the character's typical greetings?",
        "How does the character express sadness?"
    ]
    
    for query in queries:
        print(f"\nQuery: {query}")
        results = extractor.query_similar_dialogue(query, top_k=3)
        
        for idx, (dialogue, score) in enumerate(results, 1):
            print(f"\n  Result {idx} (similarity: {score:.3f}):")
            print(f"    Text: {dialogue.text[:100]}...")
            print(f"    Context: {dialogue.emotional_context}, {dialogue.situation_type}")


if __name__ == "__main__":
    # Example usage
    print("Character Personality Extraction System")
    print("=" * 60)
    print("\nThis system extracts personality patterns from scripts")
    print("and generates static rules for chatbot personalities.\n")
    
    # Example setup
    character_name = "Tony Stark"  # Replace with your character
    scripts_path = "/path/to/scripts"  # Replace with your scripts directory
    
    print(f"Character: {character_name}")
    print(f"Scripts location: {scripts_path}")
    print("\nTo use:")
    print("1. Place scripts in the scripts directory")
    print("2. Run: extractor = CharacterPersonalityExtractor('CharacterName')")
    print("3. Run: extractor.load_scripts('path/to/scripts')")
    print("4. Run: patterns = extractor.extract_personality_patterns()")
    print("5. Run: extractor.save_analysis(patterns, 'output/directory')")
