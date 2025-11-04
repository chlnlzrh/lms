# Audit Trail Configuration for LLM Systems

## Core Concepts

Audit trails in LLM systems are structured logging mechanisms that capture the complete lifecycle of AI interactions—from input processing through model invocation to output generation. Unlike traditional application logging that focuses on system events and errors, LLM audit trails must capture non-deterministic behavior, token consumption, latency patterns, and the full context that influenced each response.

### Traditional Logging vs. LLM Audit Trails

```python
# Traditional application logging
import logging

logger = logging.getLogger(__name__)

def process_order(order_id: int) -> dict:
    logger.info(f"Processing order {order_id}")
    result = database.execute_query(order_id)
    logger.info(f"Order {order_id} completed successfully")
    return result
```

```python
# LLM audit trail (complete context capture)
import json
import time
from typing import Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
import hashlib

@dataclass
class LLMInteraction:
    interaction_id: str
    timestamp: str
    user_id: str
    session_id: str
    prompt: str
    prompt_tokens: int
    completion: str
    completion_tokens: int
    total_tokens: int
    model: str
    temperature: float
    latency_ms: int
    cost_usd: float
    prompt_hash: str
    metadata: dict[str, Any]

def log_llm_interaction(
    user_id: str,
    session_id: str,
    prompt: str,
    completion: str,
    model_config: dict,
    timing: dict,
    usage: dict
) -> str:
    interaction_id = hashlib.sha256(
        f"{user_id}{session_id}{time.time()}".encode()
    ).hexdigest()[:16]
    
    audit_entry = LLMInteraction(
        interaction_id=interaction_id,
        timestamp=datetime.utcnow().isoformat(),
        user_id=user_id,
        session_id=session_id,
        prompt=prompt,
        prompt_tokens=usage['prompt_tokens'],
        completion=completion,
        completion_tokens=usage['completion_tokens'],
        total_tokens=usage['total_tokens'],
        model=model_config['model'],
        temperature=model_config['temperature'],
        latency_ms=timing['duration_ms'],
        cost_usd=calculate_cost(usage, model_config['model']),
        prompt_hash=hashlib.sha256(prompt.encode()).hexdigest()[:16],
        metadata={
            'max_tokens': model_config.get('max_tokens'),
            'top_p': model_config.get('top_p'),
            'frequency_penalty': model_config.get('frequency_penalty')
        }
    )
    
    # Write to structured storage
    write_audit_log(audit_entry)
    return interaction_id

def calculate_cost(usage: dict, model: str) -> float:
    # Cost per 1K tokens (example rates)
    rates = {
        'gpt-4': {'input': 0.03, 'output': 0.06},
        'gpt-3.5-turbo': {'input': 0.0015, 'output': 0.002}
    }
    rate = rates.get(model, {'input': 0.001, 'output': 0.002})
    return (usage['prompt_tokens'] / 1000 * rate['input'] + 
            usage['completion_tokens'] / 1000 * rate['output'])
```

### Key Engineering Insights

**Non-determinism Requires Complete Context**: Traditional logs can reference state elsewhere. LLM audit trails must be self-contained because the same prompt with identical parameters can produce different outputs. You cannot reproduce an issue without the exact context, model version, and sampling parameters.

**Token Economics Drive Architecture**: Every token has a cost. Audit trails that don't track token consumption per user, session, and feature become cost black holes. A single missing aggregation query means you can't answer "which feature consumed 80% of our API budget last month?"

**Latency Patterns Reveal Model Issues**: LLM latency is highly variable (5x variance is normal). Aggregated metrics hide problems. You need percentile tracking per prompt pattern, not just averages. A P99 latency of 30 seconds might be acceptable for document analysis but catastrophic for chatbot responses.

### Why This Matters Now

As of 2024, LLM costs represent 40-70% of operational expenses for AI-first products. Without granular audit trails, teams discover cost overruns weeks after they occur, miss abuse patterns until thousands of dollars are wasted, and cannot debug user-reported issues because they lack the context. The shift from deterministic APIs to probabilistic models means debugging requires time-travel—you must capture everything at inference time because you cannot reproduce the exact output later.

## Technical Components

### 1. Structured Log Schema Design

The audit trail schema determines what you can analyze later. Inadequate schemas force expensive data migrations or abandon entire analysis paths.

**Technical Implementation:**

```python
from enum import Enum
from typing import Optional, List
from pydantic import BaseModel, Field
from datetime import datetime

class InteractionType(str, Enum):
    COMPLETION = "completion"
    CHAT = "chat"
    EMBEDDING = "embedding"
    FUNCTION_CALL = "function_call"

class ModelProvider(str, Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    CUSTOM = "custom"

class AuditLogEntry(BaseModel):
    # Identity & Tracing
    interaction_id: str = Field(..., description="Unique interaction identifier")
    parent_interaction_id: Optional[str] = Field(None, description="For multi-turn conversations")
    session_id: str
    user_id: str
    trace_id: Optional[str] = Field(None, description="Distributed tracing correlation")
    
    # Temporal
    timestamp: datetime
    latency_ms: int
    ttfb_ms: Optional[int] = Field(None, description="Time to first byte")
    
    # Model Configuration
    provider: ModelProvider
    model: str
    model_version: Optional[str] = None
    interaction_type: InteractionType
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    top_p: Optional[float] = None
    
    # Content (consider encryption/PII handling)
    prompt: str
    completion: str
    system_prompt: Optional[str] = None
    function_definitions: Optional[List[dict]] = None
    
    # Usage Metrics
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cost_usd: float
    
    # Quality Metrics
    finish_reason: Optional[str] = None  # "stop", "length", "content_filter"
    error_code: Optional[str] = None
    retry_count: int = 0
    
    # Context
    feature_flag: Optional[str] = None
    user_tier: Optional[str] = None  # "free", "pro", "enterprise"
    geographic_region: Optional[str] = None
    
    # Hashing for deduplication
    prompt_hash: str
    completion_hash: str
    
    class Config:
        use_enum_values = True
```

**Practical Implications:**

This schema enables critical queries:
- "Show all interactions where latency exceeded 10s for pro users"
- "Calculate cost per feature flag over the last 30 days"
- "Find all retried requests that eventually failed"
- "Identify prompt patterns that hit token limits"

**Real Constraints:**

Storage costs scale linearly with verbosity. A 4000-token interaction with full metadata consumes ~5KB. At 1M interactions/day, that's 5GB/day or 150GB/month. Compression reduces this 3-5x, but you're still looking at $2-5/month in storage costs per million interactions.

**Concrete Example:**

```python
import hashlib
from datetime import datetime

def create_audit_entry(
    user_id: str,
    session_id: str,
    prompt: str,
    response: dict,
    config: dict,
    timing: dict
) -> AuditLogEntry:
    """Create audit entry from LLM interaction"""
    
    completion = response['choices'][0]['message']['content']
    usage = response['usage']
    
    return AuditLogEntry(
        interaction_id=generate_id(),
        parent_interaction_id=config.get('parent_id'),
        session_id=session_id,
        user_id=user_id,
        trace_id=config.get('trace_id'),
        timestamp=datetime.utcnow(),
        latency_ms=timing['total_ms'],
        ttfb_ms=timing.get('ttfb_ms'),
        provider=ModelProvider.OPENAI,
        model=config['model'],
        interaction_type=InteractionType.CHAT,
        temperature=config.get('temperature', 1.0),
        max_tokens=config.get('max_tokens'),
        prompt=prompt,
        completion=completion,
        system_prompt=config.get('system_prompt'),
        prompt_tokens=usage['prompt_tokens'],
        completion_tokens=usage['completion_tokens'],
        total_tokens=usage['total_tokens'],
        cost_usd=calculate_cost(usage, config['model']),
        finish_reason=response['choices'][0]['finish_reason'],
        retry_count=timing.get('retry_count', 0),
        feature_flag=config.get('feature'),
        user_tier=get_user_tier(user_id),
        prompt_hash=hashlib.sha256(prompt.encode()).hexdigest()[:16],
        completion_hash=hashlib.sha256(completion.encode()).hexdigest()[:16]
    )

def generate_id() -> str:
    """Generate unique interaction ID"""
    return hashlib.sha256(
        f"{datetime.utcnow().isoformat()}{os.urandom(16)}".encode()
    ).hexdigest()[:16]
```

### 2. Storage Backend Selection

The storage backend determines query performance, retention capabilities, and operational costs.

**Technical Explanation:**

You need to balance three competing requirements:
1. **Write throughput**: LLM interactions generate high-volume, bursty writes
2. **Query flexibility**: Ad-hoc analysis requires column-oriented access
3. **Cost efficiency**: Long-term retention of high-cardinality data

**Practical Implementation:**

```python
from abc import ABC, abstractmethod
from typing import List, Optional
import json

class AuditStore(ABC):
    @abstractmethod
    def write(self, entry: AuditLogEntry) -> None:
        pass
    
    @abstractmethod
    def query(self, filters: dict, limit: int) -> List[AuditLogEntry]:
        pass

class PostgresAuditStore(AuditStore):
    """Good for: Strong consistency, complex queries, <100M records"""
    
    def __init__(self, connection_string: str):
        import psycopg2
        self.conn = psycopg2.connect(connection_string)
        self._create_tables()
    
    def _create_tables(self):
        schema = """
        CREATE TABLE IF NOT EXISTS llm_audit_log (
            interaction_id VARCHAR(16) PRIMARY KEY,
            timestamp TIMESTAMP NOT NULL,
            user_id VARCHAR(64) NOT NULL,
            session_id VARCHAR(64) NOT NULL,
            model VARCHAR(64) NOT NULL,
            prompt_tokens INTEGER NOT NULL,
            completion_tokens INTEGER NOT NULL,
            total_tokens INTEGER NOT NULL,
            cost_usd DECIMAL(10,6) NOT NULL,
            latency_ms INTEGER NOT NULL,
            feature_flag VARCHAR(64),
            user_tier VARCHAR(32),
            prompt TEXT,
            completion TEXT,
            metadata JSONB
        );
        
        CREATE INDEX idx_user_timestamp ON llm_audit_log(user_id, timestamp DESC);
        CREATE INDEX idx_session_timestamp ON llm_audit_log(session_id, timestamp DESC);
        CREATE INDEX idx_feature_timestamp ON llm_audit_log(feature_flag, timestamp DESC);
        CREATE INDEX idx_cost ON llm_audit_log(cost_usd DESC);
        """
        with self.conn.cursor() as cur:
            cur.execute(schema)
        self.conn.commit()
    
    def write(self, entry: AuditLogEntry) -> None:
        query = """
        INSERT INTO llm_audit_log VALUES (
            %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
        )
        """
        metadata = {
            'provider': entry.provider,
            'temperature': entry.temperature,
            'finish_reason': entry.finish_reason,
            'retry_count': entry.retry_count
        }
        
        with self.conn.cursor() as cur:
            cur.execute(query, (
                entry.interaction_id,
                entry.timestamp,
                entry.user_id,
                entry.session_id,
                entry.model,
                entry.prompt_tokens,
                entry.completion_tokens,
                entry.total_tokens,
                entry.cost_usd,
                entry.latency_ms,
                entry.feature_flag,
                entry.user_tier,
                entry.prompt,
                entry.completion,
                json.dumps(metadata)
            ))
        self.conn.commit()

class ClickHouseAuditStore(AuditStore):
    """Good for: Analytics at scale, time-series queries, >1B records"""
    
    def __init__(self, host: str, port: int = 9000):
        from clickhouse_driver import Client
        self.client = Client(host=host, port=port)
        self._create_tables()
    
    def _create_tables(self):
        schema = """
        CREATE TABLE IF NOT EXISTS llm_audit_log (
            interaction_id String,
            timestamp DateTime64(3),
            user_id String,
            session_id String,
            model String,
            prompt_tokens UInt32,
            completion_tokens UInt32,
            total_tokens UInt32,
            cost_usd Decimal(10,6),
            latency_ms UInt32,
            feature_flag String,
            user_tier String,
            prompt String,
            completion String,
            metadata String
        ) ENGINE = MergeTree()
        PARTITION BY toYYYYMM(timestamp)
        ORDER BY (timestamp, user_id, session_id)
        """
        self.client.execute(schema)
    
    def write(self, entry: AuditLogEntry) -> None:
        self.client.execute(
            "INSERT INTO llm_audit_log VALUES",
            [(
                entry.interaction_id,
                entry.timestamp,
                entry.user_id,
                entry.session_id,
                entry.model,
                entry.prompt_tokens,
                entry.completion_tokens,
                entry.total_tokens,
                float(entry.cost_usd),
                entry.latency_ms,
                entry.feature_flag or '',
                entry.user_tier or '',
                entry.prompt,
                entry.completion,
                json.dumps({'provider': entry.provider})
            )]
        )
    
    def query(self, filters: dict, limit: int = 100) -> List[dict]:
        where_clauses = []
        params = {}
        
        if 'start_time' in filters:
            where_clauses.append("timestamp >= %(start_time)s")
            params['start_time'] = filters['start_time']
        
        if 'user_id' in filters:
            where_clauses.append("user_id = %(user_id)s")
            params['user_id'] = filters['user_id']
        
        where_sql = " AND ".join(where_clauses) if where_clauses else "1=1"
        query = f"SELECT * FROM llm_audit_log WHERE {where_sql} LIMIT {limit}"
        
        return self.client.execute(query, params)
```

**Real Constraints:**

- **PostgreSQL**: Excellent for <10M interactions/month, starts degrading beyond 100M total records without partitioning
- **ClickHouse**: Handles billions of records efficiently, but eventual consistency and lack of UPDATE operations require careful design
- **S3 + Parquet**: