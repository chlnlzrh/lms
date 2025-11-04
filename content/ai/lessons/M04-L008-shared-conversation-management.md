# Shared Conversation Management

## Core Concepts

Shared conversation management is the engineering discipline of maintaining, synchronizing, and controlling access to conversation state across multiple users, sessions, or systems interacting with language models. Unlike traditional stateless APIs where each request is independent, LLM conversations are inherently stateful—each exchange builds on previous context, and managing this state becomes critical when multiple parties need visibility or control.

### Traditional vs. Modern Approach

```python
# Traditional approach: Stateless API with implicit state
class TraditionalChatAPI:
    def __init__(self, api_key: str):
        self.api_key = api_key
    
    def send_message(self, user_id: str, message: str) -> str:
        # State lives in external database, opaque to application
        # No explicit conversation object
        # Limited control over context
        response = external_api.chat(
            user=user_id,
            message=message
        )
        return response.text

# Result: No visibility into conversation history, 
# difficulty coordinating between multiple interfaces,
# unclear ownership and access control

# Modern approach: Explicit conversation state management
from typing import List, Dict, Optional
from datetime import datetime
from enum import Enum

class MessageRole(Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"

class Message:
    def __init__(self, role: MessageRole, content: str, 
                 metadata: Optional[Dict] = None):
        self.role = role
        self.content = content
        self.timestamp = datetime.utcnow()
        self.metadata = metadata or {}

class Conversation:
    def __init__(self, conversation_id: str, participants: List[str]):
        self.id = conversation_id
        self.messages: List[Message] = []
        self.participants = participants
        self.created_at = datetime.utcnow()
        self.metadata: Dict = {}
    
    def add_message(self, message: Message, author: str) -> None:
        if author not in self.participants:
            raise PermissionError(f"{author} not authorized")
        self.messages.append(message)
    
    def get_context(self, max_tokens: int = 4000) -> List[Dict]:
        # Explicit control over what context is sent to model
        context = []
        token_count = 0
        
        for msg in reversed(self.messages):
            msg_tokens = len(msg.content) // 4  # Rough estimate
            if token_count + msg_tokens > max_tokens:
                break
            context.insert(0, {
                "role": msg.role.value,
                "content": msg.content
            })
            token_count += msg_tokens
        
        return context

# Result: Explicit state, clear ownership, 
# controllable context, auditable history
```

### Key Engineering Insights

**State is a first-class concern**: In traditional web development, you might treat state as an implementation detail handled by your database layer. With LLM conversations, state management directly impacts model behavior, cost, and user experience. Every decision about what to keep, trim, or share affects the intelligence of responses.

**Synchronization is non-trivial**: When multiple users or systems interact with the same conversation, you face classic distributed systems problems: race conditions, consistency models, conflict resolution. A conversation where a human and an automated system both respond simultaneously needs clear rules.

**Context is expensive and finite**: Unlike traditional chat where message history is just data storage, LLM conversation history consumes tokens (cost) and fills limited context windows (capacity). This creates unique engineering constraints around retention, summarization, and selective inclusion.

### Why This Matters Now

The shift from single-user chatbots to multi-participant AI systems is happening rapidly. Customer support scenarios involve customers, agents, and AI assistants all contributing to the same conversation. Collaborative tools have multiple team members reviewing and continuing AI-generated work. Development environments have developers and AI pair programmers sharing conversation context across sessions and IDEs.

Poor conversation management leads to three critical failures:
1. **Context corruption**: Multiple writers creating inconsistent or contradictory conversation state
2. **Cost explosion**: Redundant storage and transmission of conversation history
3. **Security leaks**: Unauthorized access to sensitive conversation content

## Technical Components

### 1. Message Structure and Serialization

Messages are the atomic units of conversation state. Unlike simple strings, production messages need rich metadata for filtering, attribution, and processing.

```python
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any
import json
from datetime import datetime

@dataclass
class MessageMetadata:
    author_id: str
    author_type: str  # "human", "assistant", "system"
    timestamp: datetime
    token_count: Optional[int] = None
    model_version: Optional[str] = None
    processing_time_ms: Optional[float] = None
    parent_message_id: Optional[str] = None
    
@dataclass
class StructuredMessage:
    id: str
    role: str  # "system", "user", "assistant"
    content: str
    metadata: MessageMetadata
    
    def to_api_format(self) -> Dict[str, str]:
        """Convert to format expected by LLM APIs"""
        return {
            "role": self.role,
            "content": self.content
        }
    
    def to_storage_format(self) -> Dict[str, Any]:
        """Convert to format for database storage"""
        result = asdict(self)
        result['metadata']['timestamp'] = \
            result['metadata']['timestamp'].isoformat()
        return result
    
    @classmethod
    def from_storage_format(cls, data: Dict[str, Any]) -> 'StructuredMessage':
        """Reconstruct from database storage"""
        data['metadata']['timestamp'] = \
            datetime.fromisoformat(data['metadata']['timestamp'])
        return cls(
            id=data['id'],
            role=data['role'],
            content=data['content'],
            metadata=MessageMetadata(**data['metadata'])
        )

# Usage example
msg = StructuredMessage(
    id="msg_123",
    role="user",
    content="Explain quantum computing",
    metadata=MessageMetadata(
        author_id="user_456",
        author_type="human",
        timestamp=datetime.utcnow()
    )
)

# Send to LLM API (minimal format)
api_format = msg.to_api_format()

# Store in database (complete format)
storage_format = msg.to_storage_format()
```

**Practical implications**: Separating API format from storage format prevents data loss and enables rich querying. You can filter messages by author, time range, or token count without reparsing content.

**Trade-offs**: Additional structure increases storage size (typically 2-3x) and serialization overhead. For high-throughput systems processing millions of messages, consider separate hot/cold storage where only recent messages include full metadata.

### 2. Conversation State Synchronization

When multiple clients access the same conversation, you need a synchronization strategy. The choice depends on consistency requirements and scale.

```python
import threading
from typing import Dict, List, Callable
from queue import Queue
import time

class ConversationStateManager:
    """Manages shared conversation state with optimistic locking"""
    
    def __init__(self):
        self.conversations: Dict[str, Conversation] = {}
        self.version_counters: Dict[str, int] = {}
        self.locks: Dict[str, threading.Lock] = {}
        self.subscribers: Dict[str, List[Callable]] = {}
    
    def get_conversation(self, conv_id: str, 
                        expected_version: Optional[int] = None) -> tuple:
        """
        Get conversation with version for optimistic locking.
        Returns (conversation, current_version)
        """
        if conv_id not in self.conversations:
            raise KeyError(f"Conversation {conv_id} not found")
        
        current_version = self.version_counters.get(conv_id, 0)
        
        if expected_version and expected_version != current_version:
            raise ConflictError(
                f"Version mismatch: expected {expected_version}, "
                f"current {current_version}"
            )
        
        return self.conversations[conv_id], current_version
    
    def update_conversation(self, conv_id: str, 
                          message: StructuredMessage,
                          expected_version: int) -> int:
        """
        Update conversation with optimistic locking.
        Returns new version number.
        """
        lock = self.locks.setdefault(conv_id, threading.Lock())
        
        with lock:
            current_version = self.version_counters.get(conv_id, 0)
            
            if expected_version != current_version:
                raise ConflictError(
                    f"Concurrent modification detected. "
                    f"Retry with version {current_version}"
                )
            
            conv = self.conversations[conv_id]
            conv.messages.append(message)
            
            new_version = current_version + 1
            self.version_counters[conv_id] = new_version
            
            # Notify subscribers
            self._notify_subscribers(conv_id, message)
            
            return new_version
    
    def subscribe(self, conv_id: str, callback: Callable) -> None:
        """Subscribe to conversation updates"""
        if conv_id not in self.subscribers:
            self.subscribers[conv_id] = []
        self.subscribers[conv_id].append(callback)
    
    def _notify_subscribers(self, conv_id: str, 
                          message: StructuredMessage) -> None:
        """Notify all subscribers of new message"""
        for callback in self.subscribers.get(conv_id, []):
            try:
                callback(conv_id, message)
            except Exception as e:
                print(f"Subscriber notification failed: {e}")

class ConflictError(Exception):
    pass

# Usage example showing conflict resolution
manager = ConversationStateManager()

def client_update(client_id: str, conv_id: str, content: str):
    """Simulates a client updating conversation"""
    max_retries = 3
    for attempt in range(max_retries):
        try:
            conv, version = manager.get_conversation(conv_id)
            
            message = StructuredMessage(
                id=f"msg_{client_id}_{time.time()}",
                role="user",
                content=content,
                metadata=MessageMetadata(
                    author_id=client_id,
                    author_type="human",
                    timestamp=datetime.utcnow()
                )
            )
            
            new_version = manager.update_conversation(
                conv_id, message, version
            )
            print(f"Client {client_id} updated to version {new_version}")
            return
            
        except ConflictError:
            if attempt == max_retries - 1:
                raise
            time.sleep(0.1 * (2 ** attempt))  # Exponential backoff
```

**Practical implications**: Optimistic locking allows high concurrency without blocking reads. Clients detect conflicts only on write, then retry with fresh data. This works well for conversation updates which are relatively infrequent compared to reads.

**Trade-offs**: Optimistic locking can cause retry storms under high contention. If multiple clients frequently update the same conversation simultaneously, consider pessimistic locking or operational transformation (OT) techniques used in collaborative editors.

### 3. Context Window Management

Context windows are finite and expensive. Effective management requires intelligent selection and compression of conversation history.

```python
from typing import List, Tuple
import tiktoken

class ContextWindowManager:
    """Manages conversation context within token limits"""
    
    def __init__(self, model: str = "gpt-3.5-turbo", 
                 max_tokens: int = 4096):
        self.encoder = tiktoken.encoding_for_model(model)
        self.max_tokens = max_tokens
        self.reserved_tokens = 500  # Reserve for response
    
    def count_tokens(self, text: str) -> int:
        """Accurate token counting for the model"""
        return len(self.encoder.encode(text))
    
    def select_messages(self, messages: List[StructuredMessage],
                       strategy: str = "recent") -> List[Dict[str, str]]:
        """
        Select messages that fit within context window.
        
        Strategies:
        - recent: Most recent messages
        - important: Messages marked as important
        - summarized: Summarize old messages, keep recent ones full
        """
        if strategy == "recent":
            return self._select_recent(messages)
        elif strategy == "important":
            return self._select_important(messages)
        elif strategy == "summarized":
            return self._select_with_summary(messages)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
    
    def _select_recent(self, 
                      messages: List[StructuredMessage]) -> List[Dict]:
        """Keep most recent messages that fit"""
        available_tokens = self.max_tokens - self.reserved_tokens
        selected = []
        token_count = 0
        
        # Always include system messages
        system_msgs = [m for m in messages if m.role == "system"]
        for msg in system_msgs:
            tokens = self.count_tokens(msg.content)
            token_count += tokens
            selected.append(msg.to_api_format())
        
        # Add recent messages in reverse chronological order
        user_assistant_msgs = [m for m in messages 
                              if m.role in ["user", "assistant"]]
        
        for msg in reversed(user_assistant_msgs):
            tokens = self.count_tokens(msg.content)
            if token_count + tokens > available_tokens:
                break
            selected.insert(len(system_msgs), msg.to_api_format())
            token_count += tokens
        
        return selected
    
    def _select_important(self, 
                         messages: List[StructuredMessage]) -> List[Dict]:
        """Prioritize messages marked as important"""
        available_tokens = self.max_tokens - self.reserved_tokens
        token_count = 0
        
        # Sort by importance (from metadata) and recency
        def importance_score(msg: StructuredMessage) -> Tuple[int, float]:
            importance = msg.metadata.metadata.get('importance', 0)
            timestamp = msg.metadata.timestamp.timestamp()
            return (importance, timestamp)
        
        sorted_msgs = sorted(messages, key=importance_score, reverse=True)
        selected = []
        
        for msg in sorted_msgs:
            tokens = self.count_tokens(msg.content)
            if token_count + tokens > available_tokens:
                continue
            selected.append(msg.to_api_format())
            token_count += tokens
        
        # Re-sort by chronological order for model
        msg_times = {msg.content: msg.metadata.timestamp 
                    for msg in sorted_msgs}
        selected.sort(key=lambda m: msg_times.get(m['content'], 
                                                  datetime.min))
        
        return selected
    
    def _select_with_summary(self, 
                           messages: List[StructuredMessage]) -> List[Dict]:
        """
        Summarize old messages, keep recent ones full.
        This is a simplified version - production would use LLM for summary.
        """
        available_tokens = self.max_tokens - self.reserved_tokens
        recent_window = 10  # Keep last 10 messages full
        
        if len(messages) <= recent_window:
            return self._select_recent(messages)
        
        old_messages = messages[:-recent_window]
        recent_messages = messages[-recent_window:]
        
        # Create simple summary (in production, use LLM)
        summary_content = (
            f"[Summary of {len(old_messages)} earlier messages: "
            f"Discussion covered {len(set(m.metadata.author_id for m in old_messages))} "
            f"participants over {len(old_messages)} exchanges]"
        )
        
        summary_msg = {
            "role": "system",
            "content": summary_content
        }
        
        selected = [summary_msg]
        token_count = self.count_tokens(summary_content)
        
        for msg in recent_messages:
            tokens = self.count_tokens(msg.content)
            if token_count + tokens > available_tokens:
                break
            selected.append(msg.to_api_format())