# Event-Driven Architecture Design for AI Systems

## Core Concepts

Event-driven architecture (EDA) is a software design pattern where system components communicate through the production, detection, and consumption of events—discrete state changes that occur at specific points in time. Unlike request-response patterns where components directly invoke each other, EDA decouples producers from consumers through an event distribution mechanism.

For AI systems, this architectural pattern becomes critical when dealing with asynchronous inference workflows, model retraining pipelines, real-time data processing, and system observability at scale.

### Traditional vs. Event-Driven Approach

```python
# Traditional synchronous approach
from typing import Dict, Any
import time

class TraditionalAIService:
    def __init__(self, model, preprocessor, validator, logger):
        self.model = model
        self.preprocessor = preprocessor
        self.validator = validator
        self.logger = logger
    
    def process_request(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Tightly coupled, blocking calls"""
        # Each step blocks until completion
        validated = self.validator.validate(input_data)
        preprocessed = self.preprocessor.transform(validated)
        prediction = self.model.predict(preprocessed)  # Could take seconds
        self.logger.log_prediction(prediction)  # Blocks inference response
        
        return prediction
        # Problems:
        # - Client waits for logging before receiving response
        # - Validator failure blocks entire pipeline
        # - No retry logic for individual components
        # - Difficult to scale components independently
```

```python
# Event-driven approach
from typing import Dict, Any, Callable, List
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import asyncio
import json
from collections import defaultdict

class EventType(Enum):
    INFERENCE_REQUESTED = "inference.requested"
    VALIDATION_COMPLETED = "validation.completed"
    PREPROCESSING_COMPLETED = "preprocessing.completed"
    PREDICTION_COMPLETED = "prediction.completed"
    PREDICTION_FAILED = "prediction.failed"

@dataclass
class Event:
    """Immutable event representation"""
    event_id: str
    event_type: EventType
    timestamp: datetime
    data: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)
    correlation_id: str = ""  # For tracing related events
    
    def to_json(self) -> str:
        return json.dumps({
            'event_id': self.event_id,
            'event_type': self.event_type.value,
            'timestamp': self.timestamp.isoformat(),
            'data': self.data,
            'metadata': self.metadata,
            'correlation_id': self.correlation_id
        })

class EventBus:
    """Simple in-memory event bus"""
    def __init__(self):
        self._handlers: Dict[EventType, List[Callable]] = defaultdict(list)
        self._event_history: List[Event] = []
        
    def subscribe(self, event_type: EventType, handler: Callable):
        """Register handler for specific event type"""
        self._handlers[event_type].append(handler)
    
    async def publish(self, event: Event):
        """Asynchronously notify all subscribers"""
        self._event_history.append(event)
        handlers = self._handlers.get(event.event_type, [])
        
        # Fire all handlers concurrently
        await asyncio.gather(
            *[self._safe_invoke(handler, event) for handler in handlers],
            return_exceptions=True
        )
    
    async def _safe_invoke(self, handler: Callable, event: Event):
        """Invoke handler with error isolation"""
        try:
            if asyncio.iscoroutinefunction(handler):
                await handler(event)
            else:
                handler(event)
        except Exception as e:
            # Handler failures don't affect other handlers
            error_event = Event(
                event_id=f"error_{event.event_id}",
                event_type=EventType.PREDICTION_FAILED,
                timestamp=datetime.now(),
                data={'error': str(e), 'original_event': event.event_id},
                correlation_id=event.correlation_id
            )
            self._event_history.append(error_event)

class EventDrivenAIService:
    def __init__(self, event_bus: EventBus):
        self.event_bus = event_bus
        self._setup_handlers()
    
    def _setup_handlers(self):
        """Wire event handlers - components don't know about each other"""
        self.event_bus.subscribe(
            EventType.INFERENCE_REQUESTED,
            self._handle_validation
        )
        self.event_bus.subscribe(
            EventType.VALIDATION_COMPLETED,
            self._handle_preprocessing
        )
        self.event_bus.subscribe(
            EventType.PREPROCESSING_COMPLETED,
            self._handle_prediction
        )
        self.event_bus.subscribe(
            EventType.PREDICTION_COMPLETED,
            self._handle_logging  # Async, doesn't block response
        )
    
    async def _handle_validation(self, event: Event):
        """Validate input and emit result"""
        # Simulate validation
        await asyncio.sleep(0.01)
        validated_data = event.data  # Simplified
        
        await self.event_bus.publish(Event(
            event_id=f"{event.event_id}_validated",
            event_type=EventType.VALIDATION_COMPLETED,
            timestamp=datetime.now(),
            data=validated_data,
            correlation_id=event.correlation_id
        ))
    
    async def _handle_preprocessing(self, event: Event):
        """Preprocess and emit result"""
        await asyncio.sleep(0.02)
        preprocessed = {'features': event.data, 'normalized': True}
        
        await self.event_bus.publish(Event(
            event_id=f"{event.event_id}_preprocessed",
            event_type=EventType.PREPROCESSING_COMPLETED,
            timestamp=datetime.now(),
            data=preprocessed,
            correlation_id=event.correlation_id
        ))
    
    async def _handle_prediction(self, event: Event):
        """Run inference and emit result"""
        await asyncio.sleep(0.1)  # Simulate model inference
        prediction = {'class': 'positive', 'confidence': 0.94}
        
        await self.event_bus.publish(Event(
            event_id=f"{event.event_id}_predicted",
            event_type=EventType.PREDICTION_COMPLETED,
            timestamp=datetime.now(),
            data=prediction,
            correlation_id=event.correlation_id
        ))
    
    async def _handle_logging(self, event: Event):
        """Log asynchronously - doesn't affect inference latency"""
        await asyncio.sleep(0.05)
        # Log to storage, metrics, etc.
        print(f"Logged prediction: {event.correlation_id}")

# Benefits demonstrated:
# - Components independently scalable
# - Failures isolated (one handler crash doesn't affect others)
# - Easy to add new handlers (e.g., monitoring) without changing existing code
# - Async logging doesn't block inference response
# - Full event history for debugging
```

### Engineering Insights That Change Your Thinking

**1. Temporal Decoupling Eliminates Cascading Failures**: In synchronous systems, if your logging service is down, inference requests fail. With EDA, prediction events are published regardless of downstream consumer health. A dead logging service doesn't prevent predictions.

**2. Events as Audit Trail**: Every state transition becomes a queryable artifact. You get distributed tracing, debugging capabilities, and compliance documentation as architectural byproducts rather than added instrumentation.

**3. Inverse Conway's Law Opportunity**: Traditional architectures force team dependencies to match service call graphs. EDA lets teams publish events and evolve independently—the data science team can add a new model evaluation subscriber without coordinating with the inference team.

### Why This Matters Now for AI Systems

AI workloads have unique characteristics that make EDA particularly valuable:

- **Variable latency**: Model inference can take milliseconds to minutes depending on input complexity and model size
- **Async retraining**: Models need periodic retraining triggered by data drift, performance degradation, or scheduled intervals
- **Multi-stage pipelines**: Prompt routing → embedding generation → vector search → LLM inference → response validation
- **Observability requirements**: Every prediction needs metadata for debugging, bias detection, and regulatory compliance
- **Horizontal scaling**: Different pipeline stages have different resource requirements (CPU for preprocessing, GPU for inference)

## Technical Components

### 1. Event Schema Design

Event schemas define the contract between producers and consumers. Poor schema design creates versioning nightmares and tight coupling.

**Technical Explanation**: Event schemas must balance specificity (rich context for consumers) with flexibility (allowing evolution without breaking changes). The key challenge is handling schema evolution as your AI system grows.

```python
from typing import Optional, Any, Literal
from pydantic import BaseModel, Field, validator
from datetime import datetime
from enum import Enum

class SchemaVersion(str, Enum):
    V1 = "1.0"
    V2 = "2.0"

class EventMetadata(BaseModel):
    """Metadata present in all events"""
    schema_version: SchemaVersion
    event_id: str
    correlation_id: str
    causation_id: str  # ID of event that caused this one
    timestamp: datetime
    producer_service: str
    
class InferenceRequestV1(BaseModel):
    """Version 1: Basic inference request"""
    schema_version: Literal[SchemaVersion.V1] = SchemaVersion.V1
    metadata: EventMetadata
    
    # Payload
    model_name: str
    input_text: str
    max_tokens: int = 100
    temperature: float = 0.7

class InferenceRequestV2(BaseModel):
    """Version 2: Added streaming and context support"""
    schema_version: Literal[SchemaVersion.V2] = SchemaVersion.V2
    metadata: EventMetadata
    
    # Enhanced payload
    model_name: str
    input_text: str
    context: Optional[List[Dict[str, str]]] = None  # New: conversation history
    max_tokens: int = 100
    temperature: float = 0.7
    stream: bool = False  # New: streaming support
    
    @validator('context')
    def validate_context(cls, v):
        """Ensure context is well-formed"""
        if v is not None:
            for msg in v:
                if 'role' not in msg or 'content' not in msg:
                    raise ValueError("Context messages must have 'role' and 'content'")
        return v

class EventSchemaRegistry:
    """Manage multiple schema versions"""
    def __init__(self):
        self._schemas = {
            ('inference.requested', '1.0'): InferenceRequestV1,
            ('inference.requested', '2.0'): InferenceRequestV2,
        }
    
    def parse_event(self, event_type: str, raw_data: Dict[str, Any]) -> BaseModel:
        """Parse event using appropriate schema version"""
        schema_version = raw_data.get('schema_version', '1.0')
        schema_class = self._schemas.get((event_type, schema_version))
        
        if not schema_class:
            raise ValueError(f"Unknown schema: {event_type} v{schema_version}")
        
        return schema_class(**raw_data)
    
    def can_consume(self, consumer_version: str, event_version: str) -> bool:
        """Check if consumer can handle event version (forward compatibility)"""
        # V2 consumers can handle V1 events (superset)
        # V1 consumers cannot handle V2 events (missing fields)
        consumer_major = int(consumer_version.split('.')[0])
        event_major = int(event_version.split('.')[0])
        return consumer_major >= event_major

# Example: Handling multiple versions
registry = EventSchemaRegistry()

# V1 event from old producer
v1_event = {
    'schema_version': '1.0',
    'metadata': {
        'schema_version': '1.0',
        'event_id': 'evt_123',
        'correlation_id': 'corr_456',
        'causation_id': 'cause_789',
        'timestamp': datetime.now(),
        'producer_service': 'api-gateway'
    },
    'model_name': 'llama-7b',
    'input_text': 'Explain quantum computing',
    'max_tokens': 150,
    'temperature': 0.8
}

parsed_v1 = registry.parse_event('inference.requested', v1_event)
print(f"Parsed V1 event: {parsed_v1.model_name}")
```

**Practical Implications**: 
- Schema versioning allows gradual migration without breaking existing consumers
- Metadata standardization enables consistent logging and tracing across all event types
- Validation catches malformed events before they propagate through the system

**Real Constraints**:
- Schema evolution requires careful planning—removing fields is a breaking change
- Over-specified schemas create tight coupling; under-specified schemas create ambiguity
- Validation adds latency (~1-5ms per event depending on complexity)

### 2. Event Ordering and Causality

In distributed systems, events can arrive out of order. For AI pipelines, this can mean processing a prediction result before preprocessing completes.

**Technical Explanation**: Event ordering isn't guaranteed in distributed systems due to network delays, concurrent processing, and clock skew. AI systems need strategies to handle out-of-order events without corrupting pipeline state.

```python
from typing import Dict, List, Optional, Set
from dataclasses import dataclass, field
from datetime import datetime
import asyncio
from collections import defaultdict

@dataclass
class CausalityTracker:
    """Track event causality and handle out-of-order delivery"""
    # Maps correlation_id -> ordered list of events
    event_chains: Dict[str, List[Event]] = field(default_factory=lambda: defaultdict(list))
    
    # Maps correlation_id -> set of event_ids we've seen
    seen_events: Dict[str, Set[str]] = field(default_factory=lambda: defaultdict(set))
    
    # Maps correlation_id -> expected next event type
    expected_next: Dict[str, EventType] = field(default_factory=dict)
    
    # Pending events waiting for predecessors
    pending: Dict[str, List[Event]] = field(default_factory=lambda: defaultdict(list))

    def track_event(self, event: Event) -> tuple[bool, Optional[str]]:
        """
        Track event and determine if it should be processed now.
        Returns: (should_process, reason_if_not)
        """
        corr_id = event.correlation_id
        
        # Check if we've seen this event (deduplication)
        if event.event_id in self.seen_events[corr_id]:
            return False, "duplicate_event"
        
        # Record the event
        self.seen_events[corr_id].add(event.event_id)
        self.event_chains[corr_id].append(event)
        
        # Check if this event's causation_id (predecessor) has been seen
        if event.metadata.get('causation_id'):
            causation_id = event.metadata['causation_id']
            if causation_id not in self.seen_events[corr_id]:
                # Predecessor hasn't arrived yet - defer processing
                self.pending[corr_id].append(event)
                return False, f"waiting_for_{causation_id}"
        
        # Check ordering based on expected sequence
        if corr_id in self.expected_next:
            if event.event_type != self.expected_next[corr_id]:
                # Out of sequence - defer
                self.pending[corr_id].append(event)
                return False, f"out_of_sequence_expected_{self.expected_next[corr_id]}"
        
        # Update expected next event
        self._update_expected_next(corr_id, event.event_type)
        
        return True, None
    
    def _update_expected_next(self, corr_id: str, current_type: EventType):
        """Define expected event sequence"""
        sequence = {
            EventType.INFERENCE_REQUESTED: EventType.VALIDATION_COMPLETED,
            