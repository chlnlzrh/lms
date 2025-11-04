# Schema Evolution Management for LLM Systems

## Core Concepts

Schema evolution management in LLM systems refers to the systematic handling of changes to structured data definitions—prompts, outputs, context formats, and tool interfaces—as your AI system evolves over time. Unlike traditional database schema evolution where migrations are explicit and version-controlled, LLM schema evolution involves managing the implicit contracts between natural language interfaces, structured outputs, and the downstream systems consuming them.

### Traditional vs. Modern Approach

**Traditional API Evolution:**

```python
# Version 1: Original schema
class UserProfileV1:
    def __init__(self, name: str, email: str):
        self.name = name
        self.email = email

# Version 2: Breaking change requires new endpoint
class UserProfileV2:
    def __init__(self, first_name: str, last_name: str, email: str):
        self.first_name = first_name
        self.last_name = last_name
        self.email = email

# Clients explicitly choose version
api.get("/v1/user")  # Returns V1
api.get("/v2/user")  # Returns V2
```

**LLM Schema Evolution:**

```python
from typing import Optional, Union, Literal
from pydantic import BaseModel, Field, validator
import json

class ExtractedEntityV1(BaseModel):
    """Original schema: simple extraction"""
    person_name: str
    company: str
    role: str

class ExtractedEntityV2(BaseModel):
    """Evolved schema: handles ambiguity and confidence"""
    person_name: str
    company: Optional[str] = None
    role: Optional[str] = None
    confidence: float = Field(ge=0.0, le=1.0)
    ambiguities: list[str] = Field(default_factory=list)
    
    @validator('company', 'role')
    def handle_unknown(cls, v):
        """Transform LLM's creative 'unknown' responses"""
        if v and v.lower() in ['unknown', 'not specified', 'n/a', 'unclear']:
            return None
        return v

class SchemaEvolutionHandler:
    """Handles multiple schema versions transparently"""
    
    def __init__(self):
        self.parsers = {
            'v1': self._parse_v1,
            'v2': self._parse_v2
        }
    
    def parse_llm_output(
        self, 
        raw_output: str, 
        target_version: Literal['v1', 'v2'] = 'v2'
    ) -> Union[ExtractedEntityV1, ExtractedEntityV2]:
        """Parse and adapt LLM output to requested schema version"""
        try:
            # Try parsing as latest version first
            data = json.loads(raw_output)
            v2_entity = ExtractedEntityV2(**data)
            
            if target_version == 'v1':
                # Downgrade schema for backward compatibility
                return ExtractedEntityV1(
                    person_name=v2_entity.person_name,
                    company=v2_entity.company or "Unknown",
                    role=v2_entity.role or "Unknown"
                )
            return v2_entity
            
        except Exception as e:
            # Fallback: extract using regex patterns
            return self._fallback_parse(raw_output, target_version)
    
    def _fallback_parse(self, text: str, version: str):
        """Graceful degradation when JSON parsing fails"""
        # Implementation of pattern-based extraction
        pass
```

The fundamental difference: **LLM schemas must handle semantic drift, ambiguity, and graceful degradation** that traditional schemas never encounter.

### Key Insights That Change Engineering Perspective

1. **Schemas are suggestions, not contracts**: LLMs may ignore or reinterpret your schema based on context. Your parser must be defensive.

2. **Forward compatibility is harder than backward compatibility**: Adding fields is risky—the LLM might start populating them inconsistently before you're ready.

3. **Versioning happens in prompts, not endpoints**: The same LLM endpoint produces different schemas based on system prompts, requiring version tracking in prompt templates.

4. **Schema validation is probabilistic**: Unlike database constraints that enforce 100% compliance, LLM output validation must handle partial compliance with confidence scores.

### Why This Matters NOW

As of 2024-2025, organizations are moving LLM applications from prototypes to production. The first wave deployed simple prompts; the second wave is discovering that **unmanaged schema evolution causes 60-80% of production incidents**:

- Customer-facing chatbots break when output formats drift
- Data pipelines fail silently when extraction schemas change
- A/B tests produce incomparable results across schema versions
- Model upgrades introduce subtle schema incompatibilities

## Technical Components

### 1. Schema Definition Layers

LLM systems have three schema layers that evolve independently:

**Prompt Schema (Input Contract):**

```python
from typing import Protocol, TypedDict
from datetime import datetime

class PromptTemplate(TypedDict):
    """Defines expected structure for prompt variables"""
    system_context: str
    user_query: str
    examples: list[dict[str, str]]
    constraints: list[str]
    output_format: str

class PromptSchemaV1:
    """Version 1: Basic extraction"""
    
    TEMPLATE = """Extract person and company from text.
Output JSON: {"person": "...", "company": "..."}

Text: {text}"""
    
    @staticmethod
    def render(text: str) -> str:
        return PromptSchemaV1.TEMPLATE.format(text=text)

class PromptSchemaV2:
    """Version 2: Enhanced with confidence and reasoning"""
    
    TEMPLATE = """Extract person and company from text.
For ambiguous information, indicate uncertainty.

Output JSON:
{
  "person": "...",
  "company": "..." or null,
  "confidence": 0.0-1.0,
  "reasoning": "brief explanation"
}

Text: {text}"""
    
    @staticmethod
    def render(text: str, require_reasoning: bool = True) -> str:
        template = PromptSchemaV2.TEMPLATE
        if not require_reasoning:
            # Allow clients to opt out of new field
            template = template.replace(
                '"reasoning": "brief explanation"',
                ''
            )
        return template.format(text=text)
```

**Output Schema (Response Contract):**

```python
from enum import Enum
from pydantic import BaseModel, Field

class ConfidenceLevel(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

class OutputSchemaV1(BaseModel):
    """Strict schema for structured extraction"""
    person: str
    company: str

class OutputSchemaV2(BaseModel):
    """Tolerant schema with validation"""
    person: str
    company: Optional[str]
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    reasoning: Optional[str] = None
    
    @property
    def confidence_level(self) -> ConfidenceLevel:
        if self.confidence >= 0.8:
            return ConfidenceLevel.HIGH
        elif self.confidence >= 0.5:
            return ConfidenceLevel.MEDIUM
        return ConfidenceLevel.LOW
    
    def to_v1(self) -> OutputSchemaV1:
        """Backward compatibility: downgrade to V1"""
        return OutputSchemaV1(
            person=self.person,
            company=self.company or "Unknown"
        )

class OutputSchemaV3(BaseModel):
    """Future version: structured entities"""
    entities: list[dict[str, any]]
    relationships: list[tuple[str, str, str]]
    metadata: dict[str, any]
    
    def to_v2(self) -> OutputSchemaV2:
        """Migration path from V3 to V2"""
        # Extract first person and company entities
        person = next(
            (e['name'] for e in self.entities if e['type'] == 'person'),
            "Unknown"
        )
        company = next(
            (e['name'] for e in self.entities if e['type'] == 'company'),
            None
        )
        return OutputSchemaV2(
            person=person,
            company=company,
            confidence=self.metadata.get('avg_confidence', 1.0)
        )
```

**Practical Implications:**

- Changes to prompt schema require testing output schema compatibility
- Output schema changes must maintain backward compatibility for existing clients
- Version mismatches between prompt and output schemas cause silent failures

**Trade-offs:**

- Strict schemas (V1) fail fast but require frequent updates
- Loose schemas (V2) are resilient but hide data quality issues
- Multi-version support increases code complexity by 2-3x

### 2. Version Detection and Routing

```python
import hashlib
import logging
from functools import lru_cache
from typing import Any

logger = logging.getLogger(__name__)

class SchemaVersionDetector:
    """Automatically detect and route schema versions"""
    
    def __init__(self):
        self.version_signatures = {
            'v1': {'required': ['person', 'company'], 'optional': []},
            'v2': {
                'required': ['person'], 
                'optional': ['company', 'confidence', 'reasoning']
            },
            'v3': {
                'required': ['entities', 'relationships'], 
                'optional': ['metadata']
            }
        }
    
    def detect_version(self, data: dict[str, Any]) -> str:
        """Detect schema version from parsed JSON"""
        keys = set(data.keys())
        
        # Try matching from newest to oldest
        for version in ['v3', 'v2', 'v1']:
            sig = self.version_signatures[version]
            required = set(sig['required'])
            allowed = required | set(sig['optional'])
            
            if required.issubset(keys) and keys.issubset(allowed):
                return version
        
        logger.warning(f"Unknown schema version for keys: {keys}")
        return 'unknown'
    
    @lru_cache(maxsize=1024)
    def compute_schema_hash(self, schema_str: str) -> str:
        """Cache schema hashes for performance"""
        return hashlib.sha256(schema_str.encode()).hexdigest()[:16]

class VersionedSchemaRouter:
    """Route requests to appropriate schema version handlers"""
    
    def __init__(self):
        self.handlers = {}
        self.detector = SchemaVersionDetector()
    
    def register_handler(self, version: str, handler: callable):
        """Register a schema version handler"""
        self.handlers[version] = handler
    
    def route(
        self, 
        raw_output: str, 
        target_version: Optional[str] = None
    ) -> BaseModel:
        """Parse and route to appropriate handler"""
        try:
            data = json.loads(raw_output)
        except json.JSONDecodeError:
            logger.error(f"Invalid JSON: {raw_output[:100]}")
            # Attempt to extract JSON from markdown code blocks
            data = self._extract_json_from_markdown(raw_output)
        
        detected_version = self.detector.detect_version(data)
        
        if target_version and target_version != detected_version:
            # Convert between versions
            return self._convert_version(
                data, 
                from_version=detected_version, 
                to_version=target_version
            )
        
        handler = self.handlers.get(detected_version)
        if not handler:
            raise ValueError(
                f"No handler registered for version: {detected_version}"
            )
        
        return handler(data)
    
    def _extract_json_from_markdown(self, text: str) -> dict:
        """Extract JSON from markdown code blocks"""
        import re
        pattern = r'```(?:json)?\s*(\{.*?\})\s*```'
        match = re.search(pattern, text, re.DOTALL)
        if match:
            return json.loads(match.group(1))
        raise ValueError("No JSON found in output")
    
    def _convert_version(
        self, 
        data: dict, 
        from_version: str, 
        to_version: str
    ) -> BaseModel:
        """Convert between schema versions"""
        # Build conversion graph
        conversions = {
            ('v1', 'v2'): self._v1_to_v2,
            ('v2', 'v1'): self._v2_to_v1,
            ('v3', 'v2'): self._v3_to_v2,
            ('v2', 'v3'): self._v2_to_v3,
        }
        
        conversion_key = (from_version, to_version)
        if conversion_key not in conversions:
            raise ValueError(
                f"No conversion path from {from_version} to {to_version}"
            )
        
        return conversions[conversion_key](data)
```

**Real Constraints:**

- Version detection adds 5-10ms latency per request
- Conversion between non-adjacent versions requires chaining
- Ambiguous schemas (overlapping field sets) cause false positives

### 3. Migration Strategies

```python
from datetime import datetime, timedelta
from dataclasses import dataclass
from collections import defaultdict

@dataclass
class MigrationConfig:
    """Configuration for gradual schema migration"""
    old_version: str
    new_version: str
    start_date: datetime
    end_date: datetime
    rollout_percentage: float = 0.0  # 0.0 to 1.0

class GradualMigrationManager:
    """Manage gradual rollout of schema changes"""
    
    def __init__(self):
        self.active_migrations: list[MigrationConfig] = []
        self.metrics = defaultdict(lambda: {'success': 0, 'failure': 0})
    
    def add_migration(self, config: MigrationConfig):
        """Register a new migration"""
        self.active_migrations.append(config)
    
    def should_use_new_schema(
        self, 
        user_id: str, 
        migration_name: str
    ) -> bool:
        """Determine if user should get new schema version"""
        migration = next(
            (m for m in self.active_migrations 
             if f"{m.old_version}->{m.new_version}" == migration_name),
            None
        )
        
        if not migration:
            return False
        
        now = datetime.now()
        if now < migration.start_date or now > migration.end_date:
            return False
        
        # Consistent hashing for stable user assignment
        user_hash = int(hashlib.sha256(user_id.encode()).hexdigest(), 16)
        return (user_hash % 100) / 100.0 < migration.rollout_percentage
    
    def record_result(
        self, 
        migration_name: str, 
        success: bool
    ):
        """Track migration success/failure rates"""
        key = 'success' if success else 'failure'
        self.metrics[migration_name][key] += 1
    
    def get_migration_health(self, migration_name: str) -> dict:
        """Get health metrics for migration"""
        stats = self.metrics[migration_name]
        total = stats['success'] + stats['failure']
        
        if total == 0:
            return {'health': 'unknown', 'success_rate': 0.0}
        
        success_rate = stats['success'] / total
        
        health = 'healthy' if success_rate > 0.95 else \
                 'degraded' if success_rate > 0.90 else \
                 'unhealthy'
        
        return {
            'health': health,
            'success_rate': success_rate,
            'total_requests': total
        }

class ShadowModeValidator:
    """Validate new schemas without affecting production"""
    
    def __init__(self, primary_parser, shadow_parser):
        self.primary = primary_parser
        self.shadow = shadow_parser
        self.discrepancies = []
    
    def parse(self, raw_output: str) -> Any:
        """Parse with primary