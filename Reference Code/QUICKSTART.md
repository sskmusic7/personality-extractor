# Quick Start Guide

## Installation

```bash
# Install dependencies
pip install sentence-transformers numpy pandas scikit-learn --break-system-packages

# Or if you have a requirements.txt
pip install -r requirements.txt --break-system-packages
```

## 5-Minute Setup

### 1. Prepare Your Script Data

Create a directory with your character's dialogue:

```bash
mkdir my_character_scripts
cd my_character_scripts
```

Add script files in any format:
- `episode1.txt` - Plain text scripts
- `dialogue.csv` - CSV with columns: character, dialogue, scene, episode
- `scripts.json` - JSON with structured scene data

### 2. Run the Extractor

```python
from character_personality_extractor import CharacterPersonalityExtractor

# Initialize
extractor = CharacterPersonalityExtractor("Your Character Name")

# Load scripts
extractor.load_scripts("my_character_scripts/")

# Extract patterns (this does everything)
patterns = extractor.extract_personality_patterns()

# Save results
extractor.save_analysis(patterns, "output/")
```

### 3. Use the Generated Rules

```python
# Import your generated personality class
from output.YourCharacterName_rules import YourCharacterNamePersonality

# Initialize personality
personality = YourCharacterNamePersonality()

# Get response guidelines for any situation
guidelines = personality.generate_response_guidelines(
    user_input="Hello!",
    emotion="neutral",
    situation="casual"
)

# Use in your chatbot system prompt
system_prompt = f"""
You are Your Character Name.

{guidelines}

Now respond to the user.
"""
```

## Complete Example

```python
# example.py
from character_personality_extractor import CharacterPersonalityExtractor

def main():
    # Step 1: Extract personality
    print("Extracting personality patterns...")
    extractor = CharacterPersonalityExtractor("Tony Stark")
    extractor.load_scripts("scripts/tony_stark/")
    patterns = extractor.extract_personality_patterns()
    extractor.save_analysis(patterns, "tony_stark_output/")
    
    # Step 2: Use the rules
    print("\nLoading generated personality...")
    from tony_stark_output.TonyStark_rules import TonyStarkPersonality
    
    personality = TonyStarkPersonality()
    
    # Step 3: Generate a response
    print("\nGenerating response...")
    style = personality.get_response_style("neutral", "casual")
    print(f"Response style: {style}")
    
    # Step 4: Create chatbot prompt
    user_message = "What do you think about AI?"
    guidelines = personality.generate_response_guidelines(
        user_message, "questioning", "professional"
    )
    
    print(f"\nSystem Prompt:\n{guidelines}")

if __name__ == "__main__":
    main()
```

Run it:
```bash
python example.py
```

## Sample Script Formats

### CSV Format (Easiest)
```csv
character,dialogue,scene,episode,other_characters
Tony Stark,"Genius billionaire playboy philanthropist.",Press Conference,S01E01,"Reporters"
Tony Stark,"Sometimes you gotta run before you can walk.",Lab,S01E02,"Pepper,Rhodes"
```

### JSON Format (Most Structured)
```json
{
  "episode": "S01E01",
  "scenes": [
    {
      "context": "Press conference",
      "dialogue": [
        {"character": "Tony Stark", "text": "I am Iron Man."},
        {"character": "Reporter", "text": "Mr. Stark, is it true?"}
      ]
    }
  ]
}
```

### Text Format (Most Flexible)
```
INT. STARK TOWER - DAY

TONY STARK: JARVIS, run a diagnostic.

JARVIS: Right away, sir.

TONY STARK: And order me some coffee. The expensive kind.
```

## Testing Your Output

After extraction, verify the results:

```python
import json

# Load patterns
with open('output/Character_patterns.json', 'r') as f:
    patterns = json.load(f)

# Check key metrics
print(f"Dialogue lines analyzed: {patterns['total_dialogue_lines']}")
print(f"Formality level: {patterns['speech_patterns']['formality_level']}")
print(f"Avg sentence length: {patterns['speech_patterns']['sentence_length']['mean']}")
print(f"Top phrases: {patterns['speech_patterns']['common_phrases'][:5]}")
```

## Integration with Your Chatbot

### Using OpenAI API
```python
import openai
from your_character_rules import YourCharacterPersonality

personality = YourCharacterPersonality()

def chat(user_message):
    guidelines = personality.generate_response_guidelines(
        user_message, "neutral", "casual"
    )
    
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": guidelines},
            {"role": "user", "content": user_message}
        ]
    )
    
    return response.choices[0].message.content
```

### Using Anthropic Claude
```python
import anthropic
from your_character_rules import YourCharacterPersonality

personality = YourCharacterPersonality()
client = anthropic.Client(api_key="your-key")

def chat(user_message):
    guidelines = personality.generate_response_guidelines(
        user_message, "neutral", "casual"
    )
    
    response = client.messages.create(
        model="claude-3-sonnet-20240229",
        max_tokens=1024,
        messages=[
            {"role": "user", "content": f"{guidelines}\n\nUser: {user_message}"}
        ]
    )
    
    return response.content[0].text
```

## Troubleshooting

### "No module named 'sentence_transformers'"
```bash
pip install sentence-transformers --break-system-packages
```

### "No dialogue entries loaded"
- Check that character name matches exactly (case-sensitive)
- Verify script files are in correct format
- Check file extensions (.txt, .json, .csv)

### "Insufficient data"
- Need at least 100+ dialogue lines
- Add more script files
- Ensure dialogue is properly tagged with character name

### Generated rules seem generic
- Need more unique character dialogue
- Add dialogue from diverse situations
- Character may actually be generic in source material

## Next Steps

1. **Refine Rules**: Edit the generated Python class to adjust patterns
2. **Test Chatbot**: Integrate with your LLM and test responses
3. **Iterate**: Re-run extraction with more/better scripts
4. **Monitor**: Track which rules fire and adjust accordingly
5. **Scale**: Use the same process for multiple characters

## Advanced Features

### Query Specific Patterns
```python
# Find how character responds in specific situations
conflict_responses = extractor.query_similar_dialogue(
    "How does character handle arguments?", 
    top_k=10
)

for dialogue, score in conflict_responses:
    print(f"{score:.3f}: {dialogue.text}")
```

### Custom Emotional Detection
```python
class CustomExtractor(CharacterPersonalityExtractor):
    def _detect_emotional_context(self, text):
        # Your custom logic
        # Could use sentiment analysis API
        return "happy"  # or other emotion
```

### Multi-Character Comparison
```python
characters = ["Character A", "Character B"]
results = {}

for char in characters:
    ext = CharacterPersonalityExtractor(char)
    ext.load_scripts(f"scripts/{char}/")
    results[char] = ext.extract_personality_patterns()

# Compare
for char, patterns in results.items():
    print(f"{char}: {patterns['speech_patterns']['formality_level']}")
```

## Resources

- `README.md` - Full documentation
- `ARCHITECTURE.md` - System design and comparison
- `example_usage.py` - Complete walkthrough with examples
- `character_personality_extractor.py` - Main implementation

## Support

Common issues:
1. **Slow embedding creation**: Normal for large datasets, be patient
2. **Memory errors**: Process scripts in smaller batches
3. **Import errors**: Ensure all dependencies installed

For more help, review the full documentation in README.md
