# Multi-Turn Conversation Design

## Core Concepts

Multi-turn conversation design is the engineering discipline of managing stateful interactions between users and language models across multiple exchanges. Unlike single-shot prompts where each request is independent, multi-turn conversations require deliberate state management, context preservation, and strategic information flow across sequential interactions.

### Traditional vs. Modern State Management

Consider how you'd implement a customer support interaction:

**Traditional Approach (Web Form):**
```python
# Each request is independent, state stored server-side
class SupportTicket:
    def __init__(self):
        self.issue_type = None
        self.details = None
        self.user_info = None
    
    def step_1_issue_type(self, issue_type: str):
        self.issue_type = issue_type
        return "Please provide details..."
    
    def step_2_details(self, details: str):
        if not self.issue_type:
            raise ValueError("Must complete step 1 first")
        self.details = details
        return "Please provide contact info..."
    
    def step_3_submit(self, user_info: dict):
        if not self.details:
            raise ValueError("Must complete step 2 first")
        self.user_info = user_info
        return self.create_ticket()
```

**LLM Conversation Approach:**
```python
from typing import List, Dict
import json

class ConversationManager:
    def __init__(self, system_prompt: str):
        self.system_prompt = system_prompt
        self.messages: List[Dict[str, str]] = [
            {"role": "system", "content": system_prompt}
        ]
    
    def add_message(self, role: str, content: str) -> None:
        """Add message to conversation history"""
        self.messages.append({"role": role, "content": content})
    
    def get_response(self, user_input: str, api_call_func) -> str:
        """Add user message, get AI response, maintain history"""
        self.add_message("user", user_input)
        
        # API call with full conversation context
        response = api_call_func(self.messages)
        assistant_message = response['content']
        
        self.add_message("assistant", assistant_message)
        return assistant_message
    
    def get_context_size(self) -> int:
        """Monitor token usage"""
        return sum(len(msg['content'].split()) for msg in self.messages)

# Usage: Natural conversation flow
support_bot = ConversationManager(
    system_prompt="""You are a technical support assistant. 
    Gather: issue type, detailed description, user contact info.
    Ask one question at a time. Be concise."""
)

# Turn 1
response = support_bot.get_response(
    "My app keeps crashing", 
    mock_api_call
)
# "I'll help you troubleshoot. What type of app is it - web, mobile, or desktop?"

# Turn 2 - context from Turn 1 automatically included
response = support_bot.get_response(
    "It's a mobile app on iOS",
    mock_api_call
)
# "Got it - iOS mobile app crashing. When does it crash - on startup, 
# during specific actions, or randomly?"
```

The fundamental difference: traditional systems use explicit state machines with predefined paths, while LLM conversations use the message history itself as state, enabling flexible, natural dialogue flows.

### Key Engineering Insights

**1. Messages Are Your Database:** The conversation history is both the UI and the persistent state. Every past exchange influences future responses. This is fundamentally different from stateless APIs.

**2. Context Is a Finite Resource:** Unlike traditional databases, you have a hard token limit (typically 8K-200K tokens). Your "database" has a shrinking capacity as conversations grow.

**3. The System Prompt Is Your Architecture:** It's not just instructions—it's the persistent layer that defines behavior across all turns. Think of it as your application's core configuration that's present in every request.

### Why This Matters Now

Multi-turn conversation design has become critical because:

- **Complexity Requires Context:** Real tasks (debugging code, planning projects, analyzing data) can't be solved in single exchanges. You need iterative refinement.
- **Users Expect Continuity:** After ChatGPT, users expect AI to "remember" what they said three messages ago. Breaking that illusion destroys UX.
- **Cost Scales Linearly:** Each turn includes ALL previous context. A 10-turn conversation costs ~10x a single prompt. Poor design makes this exponentially worse.
- **State Corruption Is Silent:** Unlike database errors that throw exceptions, conversation state degrades gracefully and invisibly until outputs become nonsensical.

## Technical Components

### 1. Message History Structure

The message list is the core data structure of any multi-turn conversation:

```python
from typing import List, Dict, Literal, Optional
from dataclasses import dataclass
from datetime import datetime

Role = Literal["system", "user", "assistant"]

@dataclass
class Message:
    role: Role
    content: str
    timestamp: datetime
    tokens: Optional[int] = None
    metadata: Optional[Dict] = None

class ConversationHistory:
    """Production-ready conversation manager with validation"""
    
    def __init__(self, system_prompt: str, max_tokens: int = 8000):
        self.max_tokens = max_tokens
        self.messages: List[Message] = [
            Message(
                role="system",
                content=system_prompt,
                timestamp=datetime.now(),
                tokens=self._estimate_tokens(system_prompt)
            )
        ]
    
    def add_message(self, role: Role, content: str, metadata: Optional[Dict] = None) -> None:
        """Add message with validation and token tracking"""
        if role not in ["user", "assistant"]:
            raise ValueError(f"Invalid role: {role}")
        
        if not content.strip():
            raise ValueError("Message content cannot be empty")
        
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
        """Rough estimate: 1 token ≈ 4 characters for English"""
        return len(text) // 4
    
    def _enforce_token_limit(self) -> None:
        """Remove oldest user/assistant messages if over limit"""
        total_tokens = sum(m.tokens or 0 for m in self.messages)
        
        while total_tokens > self.max_tokens and len(self.messages) > 1:
            # Never remove system message (index 0)
            if self.messages[1].role in ["user", "assistant"]:
                removed = self.messages.pop(1)
                total_tokens -= (removed.tokens or 0)
    
    def get_api_format(self) -> List[Dict[str, str]]:
        """Convert to API-compatible format"""
        return [
            {"role": msg.role, "content": msg.content}
            for msg in self.messages
        ]
    
    def get_total_tokens(self) -> int:
        """Get current token count"""
        return sum(m.tokens or 0 for m in self.messages)
    
    def get_recent_context(self, num_exchanges: int = 5) -> str:
        """Get recent conversation for summarization"""
        recent = self.messages[-(num_exchanges * 2):]
        return "\n".join(f"{m.role}: {m.content}" for m in recent)
```

**Practical Implications:**

- Each message triple (user → assistant → user) adds context but consumes your token budget
- The system message persists forever—make it count
- Message order matters; models are biased toward recent messages
- Metadata enables debugging without polluting the conversation

**Real Constraints:**

- Token estimation is imperfect; actual tokenization varies by model
- Removing old messages loses context permanently
- Large system prompts significantly reduce available conversation space
- JSON serialization for storage requires careful handling of metadata

### 2. Context Window Management

Context windows have hard limits. Managing this budget is critical:

```python
from enum import Enum
from typing import List, Dict

class TruncationStrategy(Enum):
    SLIDING_WINDOW = "sliding"  # Keep most recent N messages
    SUMMARIZATION = "summary"    # Summarize old, keep recent verbatim
    PRIORITY_BASED = "priority"  # Keep tagged important messages

class ContextWindowManager:
    """Advanced context management with multiple strategies"""
    
    def __init__(
        self, 
        max_tokens: int,
        strategy: TruncationStrategy = TruncationStrategy.SLIDING_WINDOW,
        reserve_tokens: int = 1000  # Reserve for response
    ):
        self.max_tokens = max_tokens
        self.strategy = strategy
        self.reserve_tokens = reserve_tokens
        self.available_tokens = max_tokens - reserve_tokens
    
    def truncate_sliding_window(
        self, 
        messages: List[Message],
        system_message: Message
    ) -> List[Message]:
        """Keep system + most recent messages that fit"""
        result = [system_message]
        current_tokens = system_message.tokens or 0
        
        # Work backwards from most recent
        for msg in reversed(messages[1:]):  # Skip system message
            msg_tokens = msg.tokens or 0
            if current_tokens + msg_tokens <= self.available_tokens:
                result.insert(1, msg)  # Insert after system
                current_tokens += msg_tokens
            else:
                break
        
        return result
    
    def truncate_with_summary(
        self,
        messages: List[Message],
        system_message: Message,
        summary_func
    ) -> List[Message]:
        """Summarize old context, keep recent messages verbatim"""
        if len(messages) <= 5:  # Not enough to summarize
            return messages
        
        # Keep system + last 4 messages (2 exchanges)
        recent = messages[-4:]
        recent_tokens = sum(m.tokens or 0 for m in recent)
        
        # Check if we need summarization
        total_tokens = sum(m.tokens or 0 for m in messages)
        if total_tokens <= self.available_tokens:
            return messages
        
        # Summarize everything between system and recent
        to_summarize = messages[1:-4]
        summary_text = summary_func(to_summarize)
        
        summary_message = Message(
            role="system",
            content=f"[Conversation Summary]: {summary_text}",
            timestamp=datetime.now(),
            tokens=len(summary_text) // 4
        )
        
        return [system_message, summary_message] + recent
    
    def truncate_priority_based(
        self,
        messages: List[Message],
        system_message: Message
    ) -> List[Message]:
        """Keep messages marked as important"""
        result = [system_message]
        current_tokens = system_message.tokens or 0
        
        # First pass: add all priority messages
        priority_messages = [
            m for m in messages[1:]
            if m.metadata and m.metadata.get("priority", False)
        ]
        
        for msg in priority_messages:
            msg_tokens = msg.tokens or 0
            if current_tokens + msg_tokens <= self.available_tokens:
                result.append(msg)
                current_tokens += msg_tokens
        
        # Second pass: fill remaining space with recent messages
        for msg in reversed(messages[1:]):
            if msg not in result:
                msg_tokens = msg.tokens or 0
                if current_tokens + msg_tokens <= self.available_tokens:
                    result.insert(-len(priority_messages), msg)
                    current_tokens += msg_tokens
        
        return sorted(result, key=lambda m: m.timestamp)

# Example usage
def simple_summary(messages: List[Message]) -> str:
    """Mock summary function - in production, use LLM"""
    topics = []
    for msg in messages:
        if msg.role == "user":
            # Extract first sentence as topic
            first_sentence = msg.content.split('.')[0]
            topics.append(first_sentence)
    return f"User discussed: {'; '.join(topics[:3])}"

manager = ContextWindowManager(
    max_tokens=4000,
    strategy=TruncationStrategy.SUMMARIZATION
)
```

**Practical Implications:**

- Sliding window is simplest but loses important early context
- Summarization preserves more context but adds latency and cost
- Priority tagging requires application logic to identify important messages
- All strategies are lossy—design for graceful degradation

**Trade-offs:**

| Strategy | Pros | Cons |
|----------|------|------|
| Sliding Window | Fast, predictable, no extra costs | Loses important early context |
| Summarization | Preserves context semantically | Extra API call, added latency, compression loss |
| Priority-Based | Keeps critical info | Requires upfront classification logic |

### 3. System Prompt Engineering for Multi-Turn

System prompts in multi-turn conversations serve as persistent instructions:

```python
class SystemPromptBuilder:
    """Structured approach to system prompt design"""
    
    @staticmethod
    def build_conversational_agent(
        role: str,
        capabilities: List[str],
        constraints: List[str],
        conversation_style: Dict[str, str],
        state_management: Dict[str, any]
    ) -> str:
        """Build comprehensive system prompt"""
        
        prompt_parts = [
            f"# Role\nYou are {role}.",
            "",
            "# Capabilities",
            *[f"- {cap}" for cap in capabilities],
            "",
            "# Constraints",
            *[f"- {constraint}" for constraint in constraints],
            "",
            "# Conversation Style",
            *[f"- {key}: {value}" for key, value in conversation_style.items()],
            "",
            "# State Management",
        ]
        
        if state_management.get("track_user_info"):
            prompt_parts.append(
                "- Remember user-provided information across turns"
            )
        
        if state_management.get("summarize_periodically"):
            prompt_parts.append(
                "- After every 5 exchanges, briefly summarize progress"
            )
        
        if state_management.get("explicit_transitions"):
            prompt_parts.append(
                "- Explicitly acknowledge topic changes"
            )
        
        return "\n".join(prompt_parts)

# Example: Technical debugging assistant
system_prompt = SystemPromptBuilder.build_conversational_agent(
    role="a technical debugging assistant for Python developers",
    capabilities=[
        "Analyze error messages and stack traces",
        "Suggest debugging steps based on symptoms",
        "Explain root causes of common issues",
        "Provide code examples for fixes"
    ],
    constraints=[
        "Do not make assumptions about code you haven't seen",
        "Always ask for clarification before suggesting major refactors",
        "Limit responses to 150 words unless user asks for detail",
        "If stuck, ask for minimal reproducible example"
    ],
    conversation_style={
        "tone": "Direct and technical, skip pleasantries",
        "questions": "Ask one specific question at a time",
        "formatting": "Use markdown code blocks for all code"
    },
    state_management={
        "track_user_info": True,
        "summarize_periodically": False,
        "explicit_transitions": True
    }
)

print(system_prompt)
```

**Output:**
```
# Role
You are a technical debugging assistant for Python developers.

# Capabilities
- Analyze error messages and stack traces
- Suggest debugging steps based on symptoms
- Explain root causes of common issues
- Provide code examples for fixes

# Constraints
- Do not make assumptions about code you haven't seen
- Always ask for clarification before suggesting major refactors
- Limit responses to 150 words unless user asks for detail
- If stuck, ask for minimal reproducible example

# Conversation Style
- tone: Direct and technical, skip pleasantries
- questions: Ask one specific question at a time
- formatting: Use markdown code blocks for all code

# State Management
- Remember user-provided information across turns
- Explicitly acknowledge topic changes
```

**Practical Implications:**

- System