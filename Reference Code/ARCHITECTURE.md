# System Architecture & Workflow

## Approach 1: One-Time RAG Training → Static Rules

```
┌─────────────────────────────────────────────────────────────────┐
│                    ANALYSIS PHASE (One-Time)                     │
└─────────────────────────────────────────────────────────────────┘

Step 1: Data Collection
┌────────────────┐
│ Character      │
│ Scripts        │──→ Parse & Structure
│ (.txt/json/csv)│
└────────────────┘
         ↓
    ┌────────────────────┐
    │ Structured         │
    │ Dialogue Entries   │
    │ + Context/Metadata │
    └────────────────────┘

Step 2: Embedding Creation
         ↓
    ┌────────────────────┐
    │ Sentence           │
    │ Transformer Model  │ ──→ all-MiniLM-L6-v2
    └────────────────────┘
         ↓
    ┌────────────────────┐
    │ Vector Embeddings  │
    │ (384 dimensions)   │
    └────────────────────┘

Step 3: Pattern Discovery (Using RAG Queries)
         ↓
    ┌─────────────────────────────────────────┐
    │ Query Embeddings for Patterns:          │
    │                                         │
    │ • "How responds when angry?"            │
    │ • "Common phrases used?"                │
    │ • "Conflict handling style?"            │
    │ • "Emotional expression patterns?"      │
    └─────────────────────────────────────────┘
         ↓
    ┌─────────────────────────────────────────┐
    │ Semantic Similarity Search              │
    │ (Cosine similarity on embeddings)       │
    └─────────────────────────────────────────┘
         ↓
    ┌─────────────────────────────────────────┐
    │ Clustered Similar Responses             │
    │ + Statistical Analysis                  │
    └─────────────────────────────────────────┘

Step 4: Rule Extraction
         ↓
    ┌─────────────────────────────────────────┐
    │ Extract Patterns:                       │
    │ • Speech: sentence length, formality    │
    │ • Emotional: responses by emotion       │
    │ • Behavioral: conflict, romance style   │
    │ • Linguistic: phrases, vocabulary       │
    └─────────────────────────────────────────┘
         ↓
    ┌─────────────────────────────────────────┐
    │ Generate Static Python Class            │
    │                                         │
    │ class CharacterPersonality:             │
    │     avg_sentence_length = 15.2          │
    │     formality = "informal"              │
    │     signature_phrases = [...]           │
    │     conflict_style = "humor"            │
    └─────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                   RUNTIME PHASE (Production)                     │
└─────────────────────────────────────────────────────────────────┘

User Input
    ↓
┌──────────────────┐
│ Detect Context   │
│ • Emotion        │──→ Simple rules (fast)
│ • Situation      │
└──────────────────┘
    ↓
┌───────────────────────────────────────┐
│ Load Static Personality Rules         │
│ (No database lookup needed!)          │
└───────────────────────────────────────┘
    ↓
┌───────────────────────────────────────┐
│ personality.get_response_style()      │
│ • Returns sentence length target      │
│ • Returns formality level             │
│ • Returns signature phrases           │
│ • Returns conflict approach           │
└───────────────────────────────────────┘
    ↓
┌───────────────────────────────────────┐
│ Generate System Prompt with Rules     │
└───────────────────────────────────────┘
    ↓
┌───────────────────────────────────────┐
│ LLM Generation (GPT/Claude)           │
│ Using personality guidelines          │
└───────────────────────────────────────┘
    ↓
Response (50-200ms)
```

## Approach 2: Runtime RAG

```
┌─────────────────────────────────────────────────────────────────┐
│                    SETUP PHASE (One-Time)                        │
└─────────────────────────────────────────────────────────────────┘

Step 1: Data Loading
┌────────────────┐
│ Character      │
│ Scripts        │──→ Parse
└────────────────┘
         ↓
    ┌────────────────────┐
    │ Create Embeddings  │
    └────────────────────┘
         ↓
    ┌────────────────────┐
    │ Vector Database    │
    │ (Pinecone/Weaviate)│
    │ Always Running     │ $$$ Monthly Cost
    └────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                   RUNTIME PHASE (Every Request)                  │
└─────────────────────────────────────────────────────────────────┘

User Input
    ↓
┌──────────────────────────┐
│ Embed User Input         │ ─→ 50-100ms
└──────────────────────────┘
    ↓
┌──────────────────────────┐
│ Query Vector DB          │ ─→ 200-500ms
│ Find Similar Dialogue    │
└──────────────────────────┘
    ↓
┌──────────────────────────┐
│ Retrieved Examples:      │
│ "Well that's..."         │
│ "I think you're..."      │
│ "Have you considered..." │
└──────────────────────────┘
    ↓
┌──────────────────────────┐
│ Anti-Quote Logic         │ ─→ 50-100ms
│ (Prevent verbatim copy)  │
└──────────────────────────┘
    ↓
┌──────────────────────────┐
│ Generate System Prompt   │
│ with Retrieved Context   │
└──────────────────────────┘
    ↓
┌──────────────────────────┐
│ LLM Generation           │ ─→ 200-500ms
└──────────────────────────┘
    ↓
Response (500-2000ms)
```

## Side-by-Side Feature Comparison

```
┌─────────────────────────────────────────────────────────────────┐
│                     FEATURE COMPARISON                           │
├──────────────────┬──────────────────┬──────────────────────────┤
│ Feature          │ Static Rules     │ Runtime RAG              │
├──────────────────┼──────────────────┼──────────────────────────┤
│ Latency          │ ⚡ 50-200ms      │ 🐌 500-2000ms            │
│ Cost             │ 💰 $0            │ 💸 $50-200/month         │
│ Consistency      │ ✅ 100%          │ ⚠️  Variable             │
│ Debugging        │ ✅ Easy          │ ❌ Hard                  │
│ Memory           │ ✅ ~100KB        │ ❌ ~100MB+               │
│ Infrastructure   │ ✅ Simple        │ ❌ Complex (Vector DB)   │
│ Quote Risk       │ ✅ None          │ ⚠️  High                 │
│ Adaptability     │ ⚠️  Static       │ ✅ Dynamic               │
│ Setup Time       │ ⚠️  Hours        │ ✅ Minutes               │
│ Scalability      │ ✅ Infinite      │ ⚠️  DB limits            │
└──────────────────┴──────────────────┴──────────────────────────┘
```

## Decision Tree

```
Do you need the chatbot personality to evolve in real-time?
│
├─ NO ──→ Do you have 100+ dialogue examples?
│         │
│         ├─ YES ──→ USE STATIC RULES ✅
│         │         (Best for most use cases)
│         │
│         └─ NO ──→ Need more data first
│
└─ YES ──→ Is consistency critical?
          │
          ├─ YES ──→ USE HYBRID APPROACH
          │         Static Rules (personality)
          │         + Runtime RAG (knowledge)
          │
          └─ NO ──→ USE RUNTIME RAG
                    (Accept variable responses)
```

## Code Architecture

### Static Rules Implementation

```python
# Character-specific personality class (generated)
class TonyStarkPersonality:
    """Static rules extracted from 847 dialogue lines"""
    
    # Core attributes
    avg_sentence_length = 15.2
    formality_level = "informal"
    signature_phrases = [...]
    conflict_style = "humor_deflection"
    
    # Response generation
    def get_response_style(self, emotion, situation):
        return {
            'sentence_length': self.avg_sentence_length,
            'should_use_sarcasm': situation == 'conflict',
            'technical_vocabulary': situation == 'professional',
            # ... more rules
        }

# Chatbot integration
personality = TonyStarkPersonality()

def generate_response(user_input, context):
    # Get personality rules (instant)
    style = personality.get_response_style(
        context.emotion, 
        context.situation
    )
    
    # Create system prompt with rules
    prompt = f"""
    You are Tony Stark. Follow these guidelines:
    - Use {style['sentence_length']} word sentences
    - Formality: {personality.formality_level}
    - In conflicts: use {personality.conflict_style}
    - Signature phrases: {personality.signature_phrases[:3]}
    
    User: {user_input}
    """
    
    return llm.generate(prompt)
```

### Runtime RAG Implementation

```python
# Vector database setup
import pinecone
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')
index = pinecone.Index('character-dialogue')

# Chatbot integration
def generate_response(user_input):
    # Embed user input (50-100ms)
    query_embedding = model.encode([user_input])[0]
    
    # Query vector DB (200-500ms)
    results = index.query(
        vector=query_embedding.tolist(),
        top_k=5,
        include_metadata=True
    )
    
    # Extract similar dialogue
    similar_examples = [
        r['metadata']['text'] 
        for r in results['matches']
    ]
    
    # Anti-quote logic (50-100ms)
    # Ensure we don't just copy examples
    filtered_examples = filter_quotes(similar_examples)
    
    # Create prompt
    prompt = f"""
    You are Tony Stark. Here are examples of how you respond:
    {filtered_examples}
    
    Now respond to (in your own words, not quotes):
    User: {user_input}
    """
    
    # Generate (200-500ms)
    return llm.generate(prompt)
```

## Performance Metrics

### Static Rules
```
Response Time Breakdown:
├─ Context detection:       5-10ms
├─ Rule lookup:            1-5ms
├─ Prompt construction:    10-20ms
├─ LLM generation:         30-150ms
└─ TOTAL:                  50-200ms

Memory Usage:
├─ Rule code:              ~100KB
├─ Model (if local):       ~500MB
└─ Total minimal:          ~100KB

Cost per 1M requests:
├─ Infrastructure:         $0
├─ LLM API calls:          $20-50
└─ Total:                  $20-50
```

### Runtime RAG
```
Response Time Breakdown:
├─ Embedding user input:   50-100ms
├─ Vector DB query:        200-500ms
├─ Post-processing:        50-100ms
├─ LLM generation:         200-500ms
└─ TOTAL:                  500-2000ms

Memory Usage:
├─ Vector DB:              100MB+
├─ Embedding model:        500MB
└─ Total:                  600MB+

Cost per 1M requests:
├─ Vector DB hosting:      $50-100/month
├─ Embedding API:          $10-20
├─ LLM API calls:          $20-50
└─ Total:                  $80-170/month + per-request
```

## When to Use Each Approach

### Use Static Rules For:
✅ Production chatbots needing consistency
✅ Cost-sensitive applications
✅ High-traffic scenarios
✅ Well-defined character personalities
✅ Situations where debugging is important
✅ When infrastructure simplicity matters

### Use Runtime RAG For:
✅ Characters that evolve over time
✅ Experimental/prototype phase
✅ When you have massive script corpus
✅ Characters with complex, nuanced knowledge
✅ When flexibility > consistency
✅ Research and analysis projects

### Use Hybrid Approach For:
✅ Best of both worlds
✅ Static rules for HOW they speak
✅ Runtime RAG for WHAT they know
✅ Production chatbots with knowledge bases
✅ When you need both consistency AND freshness
