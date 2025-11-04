# System Messages & Role Definition: Engineering Behavioral Contracts for LLMs

## Core Concepts

System messages are persistent behavioral constraints that modify an LLM's response distribution across an entire conversation. Unlike user prompts that request specific outputs, system messages establish the operating parameters—defining tone, expertise level, output format, and decision-making frameworks before any user input arrives.

Think of system messages as constructor parameters versus method arguments:

```python
# Traditional API: Behavior defined per-call
def translate_text(text: str, formality: str, domain: str) -> str:
    # Must specify behavior every single call
    return api.translate(text, formality=formality, domain=domain)

result1 = translate_text("Hello", formality="casual", domain="general")
result2 = translate_text("Goodbye", formality="casual", domain="general")
# Repetitive, error-prone if parameters drift

# LLM with System Message: Behavior defined once
class TranslationSession:
    def __init__(self):
        self.system_message = """You are a casual translator for general conversation.
        Maintain informal tone. Prioritize natural phrasing over literal accuracy."""
    
    def translate(self, text: str) -> str:
        return llm.chat([
            {"role": "system", "content": self.system_message},
            {"role": "user", "content": text}
        ])

session = TranslationSession()
result1 = session.translate("Hello")
result2 = session.translate("Goodbye")
# Consistent behavior, no parameter repetition
```

This architectural shift matters because LLMs are stateless probability engines. Every API call resets their context. System messages provide the only mechanism for establishing persistent behavioral contracts that survive across multiple turns without polluting user messages with repeated instructions.

**Key Engineering Insights:**

1. **System messages alter probability distributions, not guarantee outputs.** They bias the model toward certain response patterns but don't enforce hard constraints like schema validators.

2. **They're evaluated asymmetrically.** Most models weight system messages more heavily than user messages during attention, but less than their training data. This creates a hierarchy: training > system > user.

3. **They're opaque to users.** Unlike user messages that appear in chat UIs, system messages operate invisibly, making them ideal for application-level constraints that shouldn't burden end users.

4. **They don't expand context windows.** System message tokens count against your total context budget and persist across every turn, making them expensive in long conversations.

**Why This Matters Now:**

As LLM applications move from demos to production, the gap between "works in testing" and "works reliably at scale" comes down to behavioral consistency. A customer service bot that's helpful in morning testing but sarcastic during afternoon edge cases creates legal liability. System messages are your primary tool for closing this consistency gap without rewriting prompts for every possible user input.

## Technical Components

### 1. Role Definition and Persona Engineering

Role definition establishes the LLM's identity, expertise domain, and behavioral boundaries. This isn't creative writing—it's specifying the prior knowledge distribution you want the model to sample from.

**Technical Explanation:**

LLMs are trained on diverse text corpora covering thousands of personas (doctors, comedians, programmers, etc.). Role definition directs the model to weight its training data as if responding "in character." This works because transformer attention mechanisms can emphasize different training examples based on context.

**Practical Implications:**

```python
from typing import List, Dict
import json

def create_code_reviewer(strictness: str = "moderate") -> str:
    """Generate system message for code review bot."""
    
    strictness_params = {
        "lenient": {
            "focus": "critical bugs only",
            "tone": "encouraging",
            "style_enforcement": "ignore"
        },
        "moderate": {
            "focus": "bugs and design issues",
            "tone": "constructive",
            "style_enforcement": "suggest"
        },
        "strict": {
            "focus": "all issues including style",
            "tone": "direct",
            "style_enforcement": "require"
        }
    }
    
    params = strictness_params[strictness]
    
    return f"""You are a senior software engineer conducting code review.

EXPERTISE: 10+ years Python, distributed systems, security
FOCUS: {params['focus']}
TONE: {params['tone']}
STYLE: {params['style_enforcement']} PEP-8 compliance

OUTPUT FORMAT:
1. Security issues (if any)
2. Logic errors (if any)
3. Design concerns (if any)
4. Style suggestions (if {params['style_enforcement']} != 'ignore')

For each issue, provide:
- Line number
- Severity (critical/major/minor)
- Specific recommendation"""

# Usage comparison
def review_code(code: str, strictness: str) -> str:
    messages = [
        {"role": "system", "content": create_code_reviewer(strictness)},
        {"role": "user", "content": f"Review this code:\n```python\n{code}\n```"}
    ]
    # API call would go here
    return "review_result"
```

**Real Constraints:**

- **Over-specification paradox:** Too detailed roles can confuse models. "You are a Python expert specializing in async I/O with 5 years at FAANG who prefers functional programming" often performs worse than "You are a Python expert."

- **Role drift:** In long conversations, models may "forget" their role. Monitor for this by tracking response characteristics over time.

- **Cultural assumptions:** Roles like "professional" or "formal" encode cultural norms. Test across your actual user base.

### 2. Output Format Constraints

Format constraints specify structural requirements for responses: JSON schemas, markdown templates, field ordering, or length limits.

**Technical Explanation:**

Format constraints leverage the model's pattern recognition from training data. When you specify "respond in JSON," the model biases toward training examples that contain JSON, increasing the likelihood of valid structure. However, this is probabilistic—you still need validation.

**Practical Implementation:**

```python
from typing import TypedDict, Optional
from enum import Enum

class SentimentLabel(str, Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"

class SentimentAnalysis(TypedDict):
    sentiment: SentimentLabel
    confidence: float
    reasoning: str
    keywords: List[str]

def create_sentiment_analyzer() -> str:
    """System message for structured sentiment analysis."""
    
    example_output = {
        "sentiment": "positive",
        "confidence": 0.85,
        "reasoning": "Strong positive language with enthusiasm markers",
        "keywords": ["excellent", "loved", "highly recommend"]
    }
    
    return f"""You are a sentiment analysis system.

OUTPUT REQUIREMENTS:
- Respond ONLY with valid JSON
- No markdown formatting, no explanations outside JSON
- Use exact schema below

SCHEMA:
{json.dumps(example_output, indent=2)}

FIELD SPECS:
- sentiment: must be "positive", "negative", or "neutral"
- confidence: float 0.0-1.0, how certain you are
- reasoning: one sentence explaining the classification
- keywords: 3-5 words/phrases that drove the sentiment score

If input is unclear or mixed, use "neutral" and explain in reasoning."""

def analyze_sentiment(text: str) -> Optional[SentimentAnalysis]:
    """Analyze sentiment with structured output."""
    messages = [
        {"role": "system", "content": create_sentiment_analyzer()},
        {"role": "user", "content": text}
    ]
    
    # response = call_llm(messages)
    response = '{"sentiment": "positive", "confidence": 0.85, "reasoning": "test", "keywords": ["good"]}'
    
    try:
        parsed = json.loads(response)
        # Validate against TypedDict schema
        assert parsed["sentiment"] in ["positive", "negative", "neutral"]
        assert 0.0 <= parsed["confidence"] <= 1.0
        assert len(parsed["keywords"]) >= 3
        return parsed
    except (json.JSONDecodeError, KeyError, AssertionError) as e:
        print(f"Invalid response: {e}")
        return None

# Test it
result = analyze_sentiment("This product exceeded my expectations!")
if result:
    print(f"Sentiment: {result['sentiment']} ({result['confidence']:.0%})")
```

**Real Constraints:**

- **Validation is mandatory:** Even with perfect system messages, expect 2-5% malformed responses. Always validate and have fallback handling.

- **Schema complexity limits:** Beyond 5-7 nested fields, LLMs struggle with consistency. Flatten complex structures or use multiple calls.

- **Token overhead:** Detailed format specs consume 50-200 tokens. For high-throughput applications, this adds up quickly.

### 3. Behavioral Boundaries and Safety Rails

Behavioral boundaries define what the LLM should refuse, deflect, or escalate. This is critical for production systems where user inputs are adversarial or exploratory.

**Technical Explanation:**

LLMs have base refusal behaviors from RLHF training, but these are generic. System messages let you define application-specific boundaries: "never discuss competitors," "escalate medical questions," or "reject prompts asking for personal data."

**Practical Implementation:**

```python
from dataclasses import dataclass
from typing import Set, Optional

@dataclass
class SafetyConfig:
    """Configuration for content safety boundaries."""
    prohibited_topics: Set[str]
    escalation_triggers: Set[str]
    deflection_message: str
    escalation_message: str

def create_customer_support_system(config: SafetyConfig) -> str:
    """System message with safety boundaries."""
    
    prohibited_list = "\n".join(f"- {topic}" for topic in config.prohibited_topics)
    escalation_list = "\n".join(f"- {trigger}" for trigger in config.escalation_triggers)
    
    return f"""You are a customer support agent for a SaaS analytics platform.

CAPABILITIES:
- Answer questions about product features, pricing, account management
- Troubleshoot technical issues with our API and dashboard
- Process refund requests under $100

BOUNDARIES - NEVER discuss:
{prohibited_list}

ESCALATION - Immediately redirect to human for:
{escalation_list}

RESPONSE PATTERNS:
- Prohibited topic → "{config.deflection_message}"
- Escalation trigger → "{config.escalation_message}"
- Unknown question → "I don't have information about that. Let me connect you with someone who can help."

TONE: Professional, concise, empathetic. Max 3 paragraphs per response."""

# Example configuration
safety_config = SafetyConfig(
    prohibited_topics={
        "Competitor products or pricing",
        "Company financials or internal operations",
        "Personal information of other customers",
        "Unreleased features or roadmap details"
    },
    escalation_triggers={
        "Legal threats or demands",
        "Data breach or security incidents",
        "Refund requests over $100",
        "Account access issues",
        "Medical or life-safety concerns"
    },
    deflection_message="I focus on helping with our product. I can't discuss that topic.",
    escalation_message="This requires human expertise. I'm connecting you with a specialist now (ticket #[AUTO_GENERATED])."
)

def handle_support_query(query: str, config: SafetyConfig) -> Dict[str, any]:
    """Process support query with safety checks."""
    messages = [
        {"role": "system", "content": create_customer_support_system(config)},
        {"role": "user", "content": query}
    ]
    
    # response = call_llm(messages)
    response = "mock response"
    
    # Post-processing checks (system message is probabilistic, not guaranteed)
    needs_escalation = any(trigger.lower() in query.lower() 
                          for trigger in config.escalation_triggers)
    
    return {
        "response": response,
        "escalate": needs_escalation,
        "confidence": 0.9 if not needs_escalation else 0.3
    }

# Test boundary enforcement
test_queries = [
    "How do I export my data?",  # Normal
    "Why is your product worse than Competitor X?",  # Prohibited
    "I'm going to sue you for data loss!"  # Escalation
]

for query in test_queries:
    result = handle_support_query(query, safety_config)
    print(f"Query: {query}")
    print(f"Escalate: {result['escalate']}\n")
```

**Real Constraints:**

- **Jailbreak resistance:** Determined users can override system messages with carefully crafted user prompts ("ignore previous instructions"). Use output validation and monitoring, not just system messages.

- **Boundary ambiguity:** "Don't discuss competitors" is clear to humans but ambiguous to LLMs. Does comparing features count? Mentioning names? Test edge cases extensively.

- **False positives:** Overly restrictive boundaries lead to legitimate queries being blocked. Monitor deflection rates and iterate.

### 4. Context and Memory Specifications

Context specifications define how the LLM should use conversation history, when to reference previous turns, and how to handle context limits.

**Technical Explanation:**

LLMs see all previous messages in each API call, but system messages can instruct them how to weight that history. This is crucial for multi-turn conversations where early context becomes less relevant.

**Practical Implementation:**

```python
from collections import deque
from typing import Deque, List
import time

class ConversationManager:
    """Manage multi-turn conversations with context windowing."""
    
    def __init__(self, max_history: int = 5, max_tokens: int = 3000):
        self.max_history = max_history
        self.max_tokens = max_tokens
        self.history: Deque[Dict[str, str]] = deque(maxlen=max_history)
        self.system_message = self._create_system_message()
    
    def _create_system_message(self) -> str:
        return f"""You are a technical support assistant.

CONVERSATION CONTEXT:
- You will see up to {self.max_history} previous messages
- Focus on the CURRENT question, but maintain consistency with earlier responses
- If a question refers to "the earlier issue" or "the thing I mentioned", reference that specific previous message
- If conversation context is lost or unclear, ask for clarification instead of guessing

MEMORY PRIORITIES:
1. Current user question (always highest priority)
2. Unresolved issues from previous turns
3. User preferences or constraints mentioned earlier
4. General conversation flow

If you notice context has been truncated (conversation jumped topics abruptly), acknowledge it:
"I notice we've been discussing several topics. To give you the best answer, could you clarify which issue you're asking about?""""
    
    def _estimate_tokens(self, text: str) -> int:
        """Rough token estimation (1 token ≈ 4 characters)."""
        return len(text) // 4
    
    def _truncate_history(self) -> List[Dict[str, str]]:
        """Keep recent history within token budget."""
        messages = list(self.history)
        total_tokens = sum(self._estimate_tokens(m["content"]) for m in messages)
        
        # Remove oldest messages until under budget
        while total_tokens > self.max_tokens and len(messages) > 1:
            removed = messages.pop(0)
            total_tokens -= self._estimate_tokens(removed["content"])
        
        return messages
    
    def chat(self, user_message: str) -> str:
        """Process user message with managed context."""
        # Add user message to history
        self.history.append({"role": "user", "content": user_message})
        
        # Build messages for API call
        messages = [
            {"role": "system", "content": self.system_message}
        ] + self._truncate_history()
        
        # response = call_llm(messages)
        response = f"Mock response to: {user_message}"
        
        # Add assistant response to history
        self.history.append({"role": "assistant", "content": response})
        
        return response
    
    def get_context_stats(self) -> Dict[str, any]:
        """Get current context usage statistics."""
        messages = list(self.history)
        total_tokens = sum(self._estimate_tokens(m["content"]) for m in messages)
        system_tokens = self._estimate_tokens(self.system_message)
        
        return {
            "history_turns": len(messages) // 2,  # user + assistant = 1 turn
            "history_tokens": total_tokens,
            "system_tokens": system_tokens,
            "