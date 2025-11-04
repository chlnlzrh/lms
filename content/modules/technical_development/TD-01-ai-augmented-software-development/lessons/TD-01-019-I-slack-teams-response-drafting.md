# Slack/Teams Response Drafting with LLMs

## Core Concepts

### Technical Definition

Slack/Teams response drafting is the application of Large Language Models to generate contextually appropriate written communications in workplace chat environments. Unlike simple template-based systems or autocomplete features, LLM-powered drafting analyzes conversation history, organizational context, and communication patterns to synthesize responses that match tone, provide relevant information, and advance conversations productively.

This is fundamentally a **context-to-text transformation problem** where the model must:
1. Parse conversation threads with temporal and relational structure
2. Infer intent from incomplete or ambiguous queries
3. Generate responses constrained by professional communication norms
4. Balance informativeness with brevity appropriate to chat mediums

### Engineering Analogy: Traditional vs. Modern Approaches

```python
# Traditional Approach: Template-based response system
class TemplateResponseSystem:
    def __init__(self):
        self.templates = {
            "meeting_request": "I'm available {time}. Does that work for you?",
            "question_acknowledgment": "Thanks for asking. Let me check on that.",
            "status_update": "The {project} is currently {status}."
        }
    
    def generate_response(self, message: str, intent: str) -> str:
        """Rigid pattern matching with slot filling"""
        if intent not in self.templates:
            return "I'll get back to you on that."
        
        # Manual extraction and substitution
        template = self.templates[intent]
        # Limited flexibility, breaks on edge cases
        return template

# Modern LLM Approach: Context-aware generation
from typing import List, Dict
import anthropic

class LLMResponseDrafter:
    def __init__(self, api_key: str):
        self.client = anthropic.Anthropic(api_key=api_key)
    
    def generate_response(
        self, 
        conversation_history: List[Dict[str, str]],
        user_context: Dict[str, str],
        constraints: Dict[str, any]
    ) -> str:
        """Dynamic generation adapting to full conversation context"""
        
        # Convert conversation to structured prompt
        context = self._build_context(conversation_history, user_context)
        
        response = self.client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=constraints.get("max_tokens", 300),
            temperature=constraints.get("temperature", 0.7),
            messages=[{
                "role": "user",
                "content": f"""{context}
                
Draft a response that:
- Addresses the most recent message directly
- Maintains professional tone matching previous messages
- Provides specific information or clear next steps
- Keeps response under {constraints.get('word_limit', 100)} words"""
            }]
        )
        
        return response.content[0].text
    
    def _build_context(
        self, 
        history: List[Dict[str, str]], 
        user_context: Dict[str, str]
    ) -> str:
        """Structure conversation with metadata"""
        context_parts = [
            f"You are {user_context.get('name', 'a team member')}",
            f"Role: {user_context.get('role', 'team member')}",
            "\nConversation history:"
        ]
        
        for msg in history[-10:]:  # Last 10 messages for context
            context_parts.append(
                f"{msg['sender']}: {msg['content']} ({msg['timestamp']})"
            )
        
        return "\n".join(context_parts)
```

**Key Difference**: The template system requires predefined patterns and explicit intent classification. The LLM approach handles ambiguity, understands implicit context, and adapts tone dynamically without hard-coded rules.

### Key Insights That Change Engineering Thinking

**1. Context Window as Database Query**  
Instead of thinking about "prompting the model," conceptualize response drafting as querying a context window. Your conversation history is the database schema, and the prompt is your SELECT statement. More structured context yields more predictable outputs.

**2. Temperature Controls Personality Variance**  
For professional communications, `temperature=0.3-0.5` provides consistency while avoiding robotic repetition. Going lower (`0.1`) makes responses more deterministic but can sound formulaic across similar queries.

**3. Prompt Structure Replaces Business Logic**  
Traditional systems encode communication rules in code (`if urgent then escalate`). LLM systems encode these rules in natural language instructions within prompts. This shifts complexity from branching logic to prompt engineering.

**4. Token Economics Drive Architecture**  
At scale, including full conversation history is expensive. Efficient systems implement conversation summarization, relevant message extraction, or semantic search to minimize tokens while preserving context quality.

### Why This Matters Now

**Immediate productivity gain**: Engineers spend 20-30% of workday on chat communications. Even 10% time savings (drafting 2-3 messages per hour) yields 2-3 hours weekly.

**Quality consistency**: LLMs maintain professional tone during fatigue, frustration, or time pressure when humans tend to send rushed or unclear messages.

**Onboarding acceleration**: New team members can draft responses matching team communication patterns without months of implicit learning.

**Accessibility**: Non-native speakers gain equal footing in written communication quality, reducing bias and improving team dynamics.

## Technical Components

### Component 1: Context Extraction and Structure

**Technical Explanation**  
LLMs don't inherently understand message threading, timestamps, or sender relationships. You must explicitly structure conversation data to make these relationships clear. This involves:

- **Temporal ordering**: Presenting messages chronologically with timestamps
- **Speaker identification**: Clearly marking who said what
- **Thread hierarchy**: Representing reply chains and topic branches
- **Metadata enrichment**: Including user roles, project context, urgency markers

**Practical Implications**  
Poor context structure leads to responses that ignore recent messages, misattribute statements, or miss conversation flow. Well-structured context enables accurate, relevant responses.

**Real Constraints**  
- Context window limits (200k tokens for Claude Sonnet) mean you can't include entire channel history
- API costs scale linearly with input tokens
- More context ≠ better responses beyond ~10-15 relevant messages

**Concrete Example**

```python
from datetime import datetime
from typing import List, Dict, Optional
from dataclasses import dataclass

@dataclass
class Message:
    id: str
    sender: str
    content: str
    timestamp: datetime
    thread_id: Optional[str] = None
    reactions: List[str] = None

class ConversationContextBuilder:
    def __init__(self, max_messages: int = 15):
        self.max_messages = max_messages
    
    def build_structured_context(
        self, 
        messages: List[Message],
        current_user: str,
        focus_message_id: Optional[str] = None
    ) -> str:
        """
        Build context prioritizing recent and thread-relevant messages
        """
        # If focusing on specific message, extract its thread
        if focus_message_id:
            messages = self._extract_thread(messages, focus_message_id)
        
        # Take most recent messages up to limit
        relevant_messages = messages[-self.max_messages:]
        
        # Structure context with clear markers
        context_lines = [
            f"Current user: {current_user}",
            f"Current time: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            "\n--- Conversation History ---\n"
        ]
        
        for msg in relevant_messages:
            timestamp_str = msg.timestamp.strftime('%H:%M')
            thread_marker = f" [replying to {msg.thread_id}]" if msg.thread_id else ""
            reactions_str = f" [{', '.join(msg.reactions)}]" if msg.reactions else ""
            
            context_lines.append(
                f"[{timestamp_str}] {msg.sender}{thread_marker}: {msg.content}{reactions_str}"
            )
        
        context_lines.append("\n--- End History ---\n")
        
        return "\n".join(context_lines)
    
    def _extract_thread(
        self, 
        messages: List[Message], 
        focus_id: str
    ) -> List[Message]:
        """Extract all messages in a thread"""
        # Find the root message
        focus_msg = next(m for m in messages if m.id == focus_id)
        thread_root = focus_msg.thread_id or focus_msg.id
        
        # Get all messages in thread plus some surrounding context
        thread_messages = [
            m for m in messages 
            if m.id == thread_root or m.thread_id == thread_root
        ]
        
        return thread_messages
```

### Component 2: Tone and Style Consistency

**Technical Explanation**  
LLMs have default writing styles (often formal, verbose) that don't match casual workplace chat. Achieving consistent tone requires:

- **Few-shot examples**: Showing 2-3 example messages in your target style
- **Explicit style constraints**: "Use casual language, avoid formalities, keep under 50 words"
- **User-specific patterns**: Learning from past messages the user has sent

**Practical Implications**  
Without tone controls, responses feel "off" even when informationally correct. Users won't adopt tools that make them sound different from their authentic voice.

**Real Constraints**  
- One-shot tone calibration isn't perfect; expect 10-15% of responses to need editing
- Extremely casual or profanity-laden communication patterns may get sanitized by model safety features
- Tone transfer works better within the same language; cross-language style matching is harder

**Concrete Example**

```python
class ToneCalibrator:
    def __init__(self, client):
        self.client = client
    
    def extract_user_style(self, user_messages: List[str]) -> Dict[str, any]:
        """Analyze user's writing patterns"""
        
        analysis_prompt = f"""Analyze these messages and describe the writing style:

Messages:
{chr(10).join(f'- {msg}' for msg in user_messages[-20:])}

Provide analysis as JSON:
{{
    "formality": "casual|professional|formal",
    "avg_length": number,
    "common_phrases": ["phrase1", "phrase2"],
    "emoji_usage": "frequent|occasional|rare",
    "sentence_structure": "description"
}}"""
        
        response = self.client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=500,
            messages=[{"role": "user", "content": analysis_prompt}]
        )
        
        import json
        return json.loads(response.content[0].text)
    
    def draft_with_style(
        self, 
        context: str, 
        style_profile: Dict[str, any],
        example_messages: List[str]
    ) -> str:
        """Generate response matching user's style"""
        
        style_instructions = f"""
Style guidelines for this response:
- Formality: {style_profile['formality']}
- Target length: ~{style_profile['avg_length']} words
- Emoji usage: {style_profile['emoji_usage']}
- Sentence structure: {style_profile['sentence_structure']}

Example messages from this user showing their style:
{chr(10).join(f'"{msg}"' for msg in example_messages[-3:])}

{context}

Draft a response matching the user's authentic communication style.
"""
        
        response = self.client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=300,
            temperature=0.4,  # Lower temp for style consistency
            messages=[{"role": "user", "content": style_instructions}]
        )
        
        return response.content[0].text
```

### Component 3: Response Options and Variation

**Technical Explanation**  
Single response generation creates dependency and removes user agency. Better systems provide 2-4 response options with different:

- **Tone variants**: Professional vs. casual versions
- **Length variants**: Brief vs. detailed explanations
- **Approach variants**: Direct answer vs. clarifying question

This requires batching generation or using structured output to produce multiple coherent alternatives.

**Practical Implications**  
Users retain control and learn effective communication patterns by comparing options. Increases adoption because users feel they're enhancing—not replacing—their communication.

**Real Constraints**  
- Generating multiple responses costs 3-4x more tokens
- Need to deduplicate similar responses
- UI complexity increases with too many options (2-3 is optimal)

**Concrete Example**

```python
from typing import List, Literal
from pydantic import BaseModel

class ResponseOption(BaseModel):
    text: str
    variant: Literal["brief", "detailed", "clarifying"]
    tone: Literal["casual", "professional"]

class ResponseOptionGenerator:
    def __init__(self, client):
        self.client = client
    
    def generate_options(
        self, 
        context: str,
        num_options: int = 3
    ) -> List[ResponseOption]:
        """Generate multiple response variants"""
        
        prompt = f"""{context}

Generate {num_options} different response options as a JSON array. Each option should offer a distinct approach:

1. Brief & casual: Quick, friendly response (20-30 words)
2. Detailed & professional: Comprehensive response with specifics (50-80 words)
3. Clarifying question: Seeks more information before answering (15-25 words)

Format as JSON array:
[
  {{"text": "response text", "variant": "brief", "tone": "casual"}},
  ...
]"""
        
        response = self.client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=800,
            temperature=0.7,  # Higher for diversity
            messages=[{"role": "user", "content": prompt}]
        )
        
        import json
        options_data = json.loads(response.content[0].text)
        
        return [ResponseOption(**opt) for opt in options_data]
    
    def generate_parallel_options(
        self, 
        context: str
    ) -> List[ResponseOption]:
        """
        Generate options in parallel for faster response
        (requires async implementation or threading)
        """
        import concurrent.futures
        
        variant_prompts = {
            "brief": f"{context}\n\nDraft a brief, casual response (20-30 words).",
            "detailed": f"{context}\n\nDraft a detailed, professional response (50-80 words).",
            "clarifying": f"{context}\n\nDraft a clarifying question to get more info (15-25 words)."
        }
        
        def generate_single(variant_type: str, prompt: str) -> ResponseOption:
            resp = self.client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=300,
                temperature=0.6,
                messages=[{"role": "user", "content": prompt}]
            )
            
            tone = "professional" if variant_type == "detailed" else "casual"
            return ResponseOption(
                text=resp.content[0].text,
                variant=variant_type,
                tone=tone
            )
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            futures = {
                executor.submit(generate_single, vtype, prompt): vtype
                for vtype, prompt in variant_prompts.items()
            }
            
            options = []
            for future in concurrent.futures.as_completed(futures):
                options.append(future.result())
        
        return options
```

### Component 4: Intent Recognition and Action Routing

**Technical Explanation**  
Not all messages need drafted responses. Some require actions:
- Meeting requests → Calendar integration
- File requests → Document search
- Status queries → System lookups

Effective systems classify intent before generating responses and route to appropriate handlers.

**Practical Implications**  
Reduces token waste and latency. Enables hybrid systems where LLMs handle ambiguous communication while structured APIs handle deterministic actions.

**Real Constraints**  
- Intent classification adds latency (typically 200-400ms)
- Ambiguous messages may need clarification before action
- Over-routing to APIs removes conversational flexibility

**Concrete Example**

```python
from enum import Enum
from typing import Optional, Union

class IntentType(Enum):
    RESPONSE_NEEDED = "response"
    ACTION_REQUIRED = "action"
    INFORMATION_QUERY = "query"