# Customer Journey Mapping for AI Systems

## Core Concepts

Customer journey mapping in AI systems is the systematic process of modeling how users interact with AI-powered features across multiple touchpoints, capturing both the visible interaction layer and the underlying technical state transitions. Unlike traditional UX journey maps that focus on emotional states and visual design, AI journey mapping must account for model behavior, context management, error states, and the probabilistic nature of AI outputs.

### Engineering Analogy: Stateful vs. Stateless with Probabilistic Transitions

Traditional web applications follow deterministic state machines:

```python
from enum import Enum
from typing import Dict, Optional

class CheckoutState(Enum):
    CART = "cart"
    SHIPPING = "shipping"
    PAYMENT = "payment"
    CONFIRMATION = "confirmation"

class TraditionalCheckout:
    """Deterministic state machine - same input always produces same output"""
    
    def __init__(self):
        self.state = CheckoutState.CART
        self.transitions: Dict[CheckoutState, list[CheckoutState]] = {
            CheckoutState.CART: [CheckoutState.SHIPPING],
            CheckoutState.SHIPPING: [CheckoutState.PAYMENT],
            CheckoutState.PAYMENT: [CheckoutState.CONFIRMATION],
        }
    
    def transition(self, next_state: CheckoutState) -> bool:
        """Transition is valid or invalid - binary outcome"""
        if next_state in self.transitions.get(self.state, []):
            self.state = next_state
            return True
        return False
```

AI-powered systems introduce probabilistic, context-dependent transitions:

```python
from dataclasses import dataclass
from typing import List, Optional, Tuple
import time

@dataclass
class ConversationContext:
    """AI systems must track rich context, not just state"""
    messages: List[dict]
    user_intent: Optional[str]
    confidence: float
    tokens_used: int
    session_start: float
    clarifications_needed: int
    
class AICustomerJourney:
    """Probabilistic journey with context management and fallback paths"""
    
    def __init__(self, max_context_tokens: int = 4000):
        self.context = ConversationContext(
            messages=[],
            user_intent=None,
            confidence=0.0,
            tokens_used=0,
            session_start=time.time(),
            clarifications_needed=0
        )
        self.max_context_tokens = max_context_tokens
        
    def process_input(self, user_input: str) -> Tuple[str, bool, Optional[str]]:
        """
        Returns: (response, needs_clarification, detected_intent)
        Unlike deterministic systems, same input may yield different outcomes
        based on context, model state, and confidence thresholds
        """
        # Simulate intent detection with confidence scoring
        intent, confidence = self._detect_intent(user_input)
        self.context.user_intent = intent
        self.context.confidence = confidence
        
        # Decision tree based on confidence and context
        if confidence < 0.6:
            self.context.clarifications_needed += 1
            return (
                self._generate_clarification(user_input),
                True,  # needs_clarification
                None
            )
        
        # Context window management - critical for AI journeys
        if self.context.tokens_used > self.max_context_tokens * 0.8:
            self._compress_context()
        
        response = self._generate_response(user_input, intent)
        self.context.messages.append({
            "role": "user",
            "content": user_input,
            "timestamp": time.time()
        })
        
        return response, False, intent
    
    def _detect_intent(self, text: str) -> Tuple[str, float]:
        """Simulate ML-based intent detection"""
        # In production: model inference, vector similarity, etc.
        intents = {
            "refund": ["refund", "money back", "return"],
            "support": ["help", "issue", "problem"],
            "info": ["what", "how", "when"]
        }
        
        text_lower = text.lower()
        for intent, keywords in intents.items():
            if any(kw in text_lower for kw in keywords):
                # Confidence degrades with ambiguity
                confidence = 0.8 if len(text.split()) > 3 else 0.5
                return intent, confidence
        
        return "unknown", 0.3
    
    def _compress_context(self) -> None:
        """Critical for long conversations - summarize or truncate"""
        # Keep only recent messages, summarize rest
        if len(self.context.messages) > 10:
            self.context.messages = self.context.messages[-5:]
            self.context.tokens_used = len(str(self.context.messages)) // 4
    
    def _generate_clarification(self, user_input: str) -> str:
        """Fallback path when confidence is low"""
        return f"I want to help you properly. Could you provide more details about what you need?"
    
    def _generate_response(self, user_input: str, intent: str) -> str:
        """Generate contextual response based on intent"""
        responses = {
            "refund": "I can help process your refund. What's your order number?",
            "support": "I'm here to help. What issue are you experiencing?",
            "info": "I can provide that information. What specifically would you like to know?"
        }
        return responses.get(intent, "How can I assist you today?")

# Usage comparison
traditional = TraditionalCheckout()
print(traditional.transition(CheckoutState.SHIPPING))  # Always True if valid

ai_journey = AICustomerJourney()
# Same input may require different handling based on context
response1, needs_clarification1, _ = ai_journey.process_input("help")
response2, needs_clarification2, _ = ai_journey.process_input("I need help with my order refund")
print(f"Short input needs clarification: {needs_clarification1}")  # Likely True
print(f"Detailed input needs clarification: {needs_clarification2}")  # Likely False
```

### Key Insights That Change Engineering Thinking

1. **Journey mapping is debugging for user experience**: Just as you trace code execution with breakpoints, journey maps trace how users flow through AI interactions, revealing where context is lost, confidence drops, or fallback paths activate.

2. **Every AI interaction has three outcomes**: Success (high confidence, correct action), clarification needed (low confidence, request more info), and failure (misunderstood intent, wrong action). Traditional systems have two: success or error.

3. **Context is your most expensive resource**: In traditional apps, you worry about database connections or memory. In AI systems, token limits and context window management directly constrain what journeys are possible.

4. **Non-determinism requires journey forking**: You can't design a single linear journey. You must map confidence-based branches, timeout paths, and graceful degradation routes.

### Why This Matters NOW

LLM-powered features are being added to existing products at unprecedented rates. Engineers who treat AI interactions like REST endpoints (request → deterministic response) ship products with poor user experience: confusion when the AI misunderstands, frustration when context is lost mid-conversation, and abandonment when clarification loops spiral. Journey mapping exposes these issues during design, not after user complaints.

Cost management is critical: a poorly designed journey might make 10 API calls where 3 would suffice. At $0.002 per 1K tokens, inefficient journeys can cost 3x more while delivering worse experiences.

## Technical Components

### 1. State and Context Management

AI journeys require tracking both conversation state (what phase of interaction) and conversation context (history, user data, session info). Poor context management causes the most common AI journey failures.

**Technical Explanation:**

Context management involves three layers:
- **Session context**: User identity, preferences, session metadata
- **Conversation context**: Message history within token limits
- **Domain context**: Relevant business data and rules

```python
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import json

@dataclass
class Message:
    role: str  # "user", "assistant", "system"
    content: str
    timestamp: datetime
    tokens: int
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class SessionContext:
    session_id: str
    user_id: str
    started_at: datetime
    preferences: Dict[str, Any]
    domain_data: Dict[str, Any]  # Order info, account status, etc.

class ContextManager:
    """
    Manages conversation context with automatic pruning and summarization
    """
    
    def __init__(self, max_tokens: int = 4000, system_prompt_tokens: int = 200):
        self.max_tokens = max_tokens
        self.system_prompt_tokens = system_prompt_tokens
        self.available_tokens = max_tokens - system_prompt_tokens
        self.messages: List[Message] = []
        self.session: Optional[SessionContext] = None
        
    def add_message(self, role: str, content: str, metadata: Optional[Dict] = None) -> None:
        """Add message with automatic context window management"""
        tokens = self._estimate_tokens(content)
        message = Message(
            role=role,
            content=content,
            timestamp=datetime.now(),
            tokens=tokens,
            metadata=metadata or {}
        )
        
        self.messages.append(message)
        self._enforce_token_limit()
    
    def _estimate_tokens(self, text: str) -> int:
        """Rough estimation: 1 token ≈ 4 characters for English"""
        return len(text) // 4
    
    def _enforce_token_limit(self) -> None:
        """Prune old messages when approaching token limit"""
        total_tokens = sum(m.tokens for m in self.messages)
        
        if total_tokens <= self.available_tokens:
            return
        
        # Strategy: Keep first message (often contains key context) and recent messages
        if len(self.messages) <= 2:
            return  # Can't prune further
        
        # Always keep the first user message (initial request)
        first_message = self.messages[0]
        recent_messages = []
        tokens_needed = self.available_tokens - first_message.tokens
        
        # Work backwards, keeping most recent messages
        for msg in reversed(self.messages[1:]):
            if msg.tokens <= tokens_needed:
                recent_messages.insert(0, msg)
                tokens_needed -= msg.tokens
            else:
                break
        
        self.messages = [first_message] + recent_messages
    
    def get_context_for_llm(self) -> List[Dict[str, str]]:
        """Format context for LLM API call"""
        return [
            {"role": msg.role, "content": msg.content}
            for msg in self.messages
        ]
    
    def add_session_context(self, session: SessionContext) -> None:
        """Inject session context as system message"""
        self.session = session
        context_summary = self._format_session_context()
        self.add_message("system", context_summary, {"type": "session_context"})
    
    def _format_session_context(self) -> str:
        """Format session data for LLM understanding"""
        if not self.session:
            return ""
        
        parts = ["Session Context:"]
        
        if self.session.domain_data:
            parts.append(f"User Data: {json.dumps(self.session.domain_data, indent=2)}")
        
        if self.session.preferences:
            parts.append(f"Preferences: {json.dumps(self.session.preferences, indent=2)}")
        
        return "\n".join(parts)
    
    def get_summary_stats(self) -> Dict[str, Any]:
        """Debugging and monitoring helper"""
        total_tokens = sum(m.tokens for m in self.messages)
        return {
            "message_count": len(self.messages),
            "total_tokens": total_tokens,
            "available_tokens": self.available_tokens,
            "utilization": f"{(total_tokens/self.available_tokens)*100:.1f}%",
            "session_duration": (
                datetime.now() - self.session.started_at
                if self.session else None
            )
        }

# Example usage
context_mgr = ContextManager(max_tokens=4000)

# Initialize session with domain data
session = SessionContext(
    session_id="sess_123",
    user_id="user_456",
    started_at=datetime.now(),
    preferences={"language": "en", "notification_email": True},
    domain_data={"order_id": "ORD-789", "order_status": "shipped"}
)
context_mgr.add_session_context(session)

# Simulate conversation
context_mgr.add_message("user", "Where is my order?")
context_mgr.add_message("assistant", "Your order ORD-789 is currently shipped and expected to arrive in 2 days.")
context_mgr.add_message("user", "Can I change the delivery address?")

print(json.dumps(context_mgr.get_summary_stats(), indent=2, default=str))
```

**Practical Implications:**
- Context pruning is not optional—it's required for conversations longer than 5-10 exchanges
- Session context injection reduces repetitive user input ("What's your order number?")
- Token estimation must be conservative; underestimating causes API errors mid-conversation

**Real Constraints:**
- Different LLM APIs have different token limits (4K, 8K, 32K, 128K)
- Token counting varies by model; accurate counting requires model-specific tokenizers
- Context that spans multiple API calls requires persistence (database, cache)

### 2. Intent Detection and Confidence Scoring

AI journeys branch based on detected intent and confidence levels. Low confidence triggers clarification paths; high confidence enables direct action.

**Technical Explanation:**

Intent detection can use embeddings similarity, classification models, or LLM-based extraction. Confidence scoring determines which path the journey takes.

```python
from typing import List, Tuple, Optional
import numpy as np
from dataclasses import dataclass

@dataclass
class Intent:
    name: str
    confidence: float
    entities: dict
    clarification_needed: bool

class IntentRouter:
    """
    Route user inputs to appropriate handlers based on intent and confidence
    """
    
    def __init__(self, confidence_threshold: float = 0.7):
        self.confidence_threshold = confidence_threshold
        self.intent_patterns = self._initialize_patterns()
        
    def _initialize_patterns(self) -> dict:
        """Define intent patterns with example embeddings (simplified)"""
        return {
            "order_status": {
                "keywords": ["where", "status", "track", "order", "shipped"],
                "requires_entities": ["order_id"],
                "handler": "check_order_status"
            },
            "refund_request": {
                "keywords": ["refund", "return", "money back", "cancel order"],
                "requires_entities": ["order_id"],
                "handler": "process_refund"
            },
            "product_inquiry": {
                "keywords": ["what", "how", "specs", "features", "product"],
                "requires_entities": ["product_name"],
                "handler": "product_information"
            },
            "general_support": {
                "keywords": ["help", "support", "issue", "problem"],
                "requires_entities": [],
                "handler": "general_support"
            }
        }
    
    def detect_intent(self, user_input: str, context: Optional[dict] = None) -> Intent:
        """
        Detect intent with confidence scoring
        In production, this would use embeddings or a fine-tuned classifier
        """
        user_input_lower = user_input.lower()
        words = set(user_input_lower.split())
        
        intent_scores = {}
        for intent_name, pattern in self.intent_patterns.items():
            # Calculate keyword match score
            keyword_matches = sum(
                1 for kw in pattern["keywords"]
                if kw in user_input_lower
            )
            
            # Normalize by pattern keyword count
            score = keyword_matches / len(pattern["keywords"])
            
            # Boost score if context provides required entities
            if