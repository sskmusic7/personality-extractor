"""
Example Usage: Character Personality Extraction Pipeline

This script demonstrates the complete workflow from script analysis
to static rule generation.
"""

import json
import os
from character_personality_extractor import CharacterPersonalityExtractor, demonstrate_rag_queries


def create_sample_data(output_dir: str):
    """Create sample script data for demonstration."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Sample JSON format
    sample_json = {
        "episode": "S01E01",
        "scenes": [
            {
                "context": "Office meeting, tense atmosphere",
                "dialogue": [
                    {"character": "Tony Stark", "text": "Well, that's an interesting proposal. Have you considered the complete lack of feasibility?"},
                    {"character": "Pepper", "text": "Tony, can you just listen for once?"},
                    {"character": "Tony Stark", "text": "I am listening. I'm also calculating the probability of failure, which is currently at 99.7%."},
                ]
            },
            {
                "context": "Lab, working on project",
                "dialogue": [
                    {"character": "Tony Stark", "text": "JARVIS, run diagnostics on the new arc reactor design."},
                    {"character": "JARVIS", "text": "Right away, sir."},
                    {"character": "Tony Stark", "text": "And while you're at it, order me some coffee. The expensive kind."},
                ]
            }
        ]
    }
    
    with open(os.path.join(output_dir, "sample_episode.json"), 'w') as f:
        json.dump(sample_json, f, indent=2)
    
    # Sample CSV format
    csv_content = """character,dialogue,scene,episode,other_characters
Tony Stark,"Sometimes you gotta run before you can walk.",Lab Scene,S01E02,"Pepper,Rhodes"
Tony Stark,"I'm not saying I'm responsible for this country's longest run of uninterrupted peace in 35 years. I'm not saying that from the ashes of captivity, never has a greater Phoenix metaphor been personified in human history.",Press Conference,S01E02,"Reporters"
Tony Stark,"If you douse me again, and I'm not on fire, I'm donating you to a city college.",Lab Scene,S01E03,JARVIS
"""
    
    with open(os.path.join(output_dir, "sample_dialogue.csv"), 'w') as f:
        f.write(csv_content)
    
    # Sample text script format
    text_script = """FADE IN:

INT. STARK TOWER - NIGHT

TONY STARK stands at the window, drink in hand.

TONY STARK: You know what the problem with being a genius is? Everyone expects you to have all the answers.

PEPPER enters.

PEPPER: Tony, we need to talk about the board meeting.

TONY STARK: Ah, the board meeting. Where people who can't understand my technology try to tell me how to run my company. Sounds thrilling.

PEPPER: This is serious.

TONY STARK: I'm always serious, Pepper. I'm just better at hiding it behind sarcasm and expensive suits.

SCENE 2

EXT. MALIBU MANSION - DAY

TONY STARK: JARVIS, what's the status on the suit upgrades?

JARVIS: All systems are operational, sir.

TONY STARK: Good. Because I've got a feeling we're going to need them.
"""
    
    with open(os.path.join(output_dir, "sample_script.txt"), 'w') as f:
        f.write(text_script)
    
    print(f"Sample data created in {output_dir}")


def run_complete_pipeline(character_name: str, scripts_path: str, output_dir: str):
    """Run the complete personality extraction pipeline."""
    
    print("\n" + "="*70)
    print(f"PERSONALITY EXTRACTION PIPELINE FOR: {character_name}")
    print("="*70 + "\n")
    
    # Step 1: Initialize extractor
    print("STEP 1: Initializing extractor...")
    extractor = CharacterPersonalityExtractor(character_name)
    
    # Step 2: Load scripts
    print("\nSTEP 2: Loading scripts...")
    extractor.load_scripts(scripts_path)
    
    if not extractor.dialogue_entries:
        print("ERROR: No dialogue found. Check your script files.")
        return
    
    # Step 3: Create embeddings
    print("\nSTEP 3: Creating embeddings...")
    extractor.create_embeddings()
    
    # Step 4: Demonstrate RAG queries (analysis phase)
    print("\nSTEP 4: Demonstrating RAG queries for pattern discovery...")
    demonstrate_rag_queries(extractor)
    
    # Step 5: Extract personality patterns
    print("\nSTEP 5: Extracting personality patterns...")
    patterns = extractor.extract_personality_patterns()
    
    # Step 6: Display key findings
    print("\n" + "="*70)
    print("KEY FINDINGS")
    print("="*70)
    
    speech = patterns['speech_patterns']
    print(f"\nSpeech Patterns:")
    print(f"  - Average sentence length: {speech['sentence_length']['mean']:.1f} words")
    print(f"  - Formality level: {speech['formality_level']}")
    print(f"  - Question frequency: {speech['question_frequency']:.1%}")
    print(f"  - Exclamation frequency: {speech['exclamation_frequency']:.1%}")
    print(f"  - Top phrases: {', '.join([p[0] for p in speech['common_phrases'][:5]])}")
    
    conflict = patterns['conflict_handling']
    print(f"\nConflict Handling:")
    print(f"  - Dominant style: {conflict.get('dominant_style', 'N/A')}")
    
    emotions = patterns['emotional_responses']
    print(f"\nEmotional Contexts Found: {len(emotions)}")
    for emotion, examples in list(emotions.items())[:3]:
        print(f"  - {emotion}: {len(examples)} examples")
    
    # Step 7: Generate static rules
    print("\n" + "="*70)
    print("STEP 6: Generating static personality rules...")
    print("="*70 + "\n")
    
    rules_code = extractor.generate_static_rules(patterns)
    print("\nGenerated rules preview:")
    print(rules_code[:500] + "\n...\n")
    
    # Step 8: Save everything
    print("\nSTEP 7: Saving analysis...")
    extractor.save_analysis(patterns, output_dir)
    
    print("\n" + "="*70)
    print("PIPELINE COMPLETE!")
    print("="*70)
    print(f"\nOutput files in: {output_dir}")
    print(f"  - {character_name}_patterns.json (full analysis)")
    print(f"  - {character_name}_rules.py (static personality rules)")
    print(f"  - {character_name}_embeddings.npy (vector embeddings)")
    
    return patterns, extractor


def compare_approaches():
    """Compare the static rules vs runtime RAG approaches."""
    
    print("\n" + "="*70)
    print("COMPARISON: Static Rules vs Runtime RAG")
    print("="*70 + "\n")
    
    comparison = {
        "Approach": ["Static Rules (One-time RAG)", "Runtime RAG"],
        
        "Response Time": [
            "~50-200ms (fast)",
            "~500-2000ms (slower, requires embedding lookup)"
        ],
        
        "Consistency": [
            "Highly consistent - same rules always apply",
            "Variable - depends on which embeddings retrieved"
        ],
        
        "Memory Usage": [
            "Low - just rule code (~100KB)",
            "High - full embedding database (100MB+)"
        ],
        
        "Cost": [
            "Zero after initial extraction",
            "Ongoing vector DB hosting + API calls"
        ],
        
        "Adaptability": [
            "Static - requires re-extraction to update",
            "Dynamic - always pulls from source material"
        ],
        
        "Quote Risk": [
            "No risk - generates based on patterns",
            "High risk - may return direct quotes"
        ],
        
        "Debugging": [
            "Easy - trace exact rule that fired",
            "Hard - embedding similarity is opaque"
        ],
        
        "Initial Setup": [
            "Time-intensive - run full extraction pipeline",
            "Moderate - just load scripts into vector DB"
        ]
    }
    
    print(f"{'Metric':<20} | {'Static Rules':<40} | {'Runtime RAG':<40}")
    print("-" * 105)
    
    for i in range(len(comparison["Approach"])):
        for key in list(comparison.keys())[1:]:
            print(f"{key:<20} | {comparison[key][0]:<40} | {comparison[key][1]:<40}")
        if i == 0:
            print("-" * 105)
    
    print("\n" + "="*70)
    print("RECOMMENDATION")
    print("="*70)
    print("""
Use Static Rules When:
- Building production chatbots that need fast, consistent responses
- You want full control over personality traits
- Cost and infrastructure simplicity matter
- The character's personality is well-defined and stable

Use Runtime RAG When:
- Character knowledge needs to stay current (evolving characters)
- You want maximum flexibility to adjust on the fly
- The character has massive amounts of source material
- You're okay with higher latency and costs
    
Hybrid Approach (Best of Both):
- Use Static Rules for personality/speech patterns (how they speak)
- Use Runtime RAG for knowledge/facts (what they know)
- This gives you fast, consistent personality with dynamic knowledge
    """)


if __name__ == "__main__":
    # Configuration
    CHARACTER_NAME = "Tony Stark"
    SCRIPTS_DIR = "./sample_scripts"
    OUTPUT_DIR = "./personality_output"
    
    print("="*70)
    print("CHARACTER PERSONALITY EXTRACTION - COMPLETE EXAMPLE")
    print("="*70)
    
    # Create sample data
    print("\nCreating sample script data...")
    create_sample_data(SCRIPTS_DIR)
    
    # Run the pipeline
    patterns, extractor = run_complete_pipeline(
        character_name=CHARACTER_NAME,
        scripts_path=SCRIPTS_DIR,
        output_dir=OUTPUT_DIR
    )
    
    # Show comparison
    compare_approaches()
    
    print("\n" + "="*70)
    print("NEXT STEPS")
    print("="*70)
    print("""
1. Replace sample data with real character scripts
2. Run the extraction pipeline on your character
3. Review generated rules in {character}_rules.py
4. Integrate rules into your chatbot system prompt
5. Test chatbot responses against character examples
6. Iterate and refine rules as needed

For production use:
- Add more sophisticated emotional detection
- Include context-aware rule selection
- Implement rule confidence scoring
- Add A/B testing framework for rule effectiveness
    """)
