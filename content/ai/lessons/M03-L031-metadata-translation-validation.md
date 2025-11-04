# Metadata Translation & Validation for LLM Systems

## Core Concepts

**Technical Definition:** Metadata translation and validation is the systematic process of transforming structured data representations between formats while ensuring semantic consistency, type safety, and constraint adherence across LLM interaction boundaries. This encompasses schema mapping, type coercion, validation rule enforcement, and semantic preservation when data crosses between traditional typed systems and LLM text-based interfaces.

### Engineering Analogy: Type Systems Across Boundaries

Traditional API integration handles typed data within a single type system:

```python
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

@dataclass
class UserProfile:
    user_id: int
    email: str
    created_at: datetime
    preferences: dict[str, bool]
    credit_limit: Optional[float] = None

# Direct serialization, type preservation guaranteed
def api_call(profile: UserProfile) -> dict:
    return {
        "user_id": profile.user_id,
        "email": profile.email,
        "created_at": profile.created_at.isoformat(),
        "preferences": profile.preferences,
        "credit_limit": profile.credit_limit
    }
```

LLM integration breaks this type safety—you're crossing a lossy boundary:

```python
from typing import Any, TypedDict
import json
from pydantic import BaseModel, Field, validator

class UserProfileSchema(BaseModel):
    user_id: int = Field(gt=0, description="Positive integer user identifier")
    email: str = Field(regex=r'^[\w\.-]+@[\w\.-]+\.\w+$')
    created_at: str = Field(description="ISO 8601 datetime string")
    preferences: dict[str, bool]
    credit_limit: Optional[float] = Field(ge=0, le=100000)
    
    @validator('created_at')
    def validate_datetime(cls, v):
        try:
            datetime.fromisoformat(v)
            return v
        except ValueError:
            raise ValueError("Must be valid ISO 8601 datetime")

def llm_interaction_with_validation(profile: UserProfile, llm_client) -> UserProfile:
    # Translate TO LLM: structured → text with schema
    schema_doc = UserProfileSchema.schema_json(indent=2)
    profile_json = json.dumps({
        "user_id": profile.user_id,
        "email": profile.email,
        "created_at": profile.created_at.isoformat(),
        "preferences": profile.preferences,
        "credit_limit": profile.credit_limit
    })
    
    prompt = f"""
Schema for UserProfile:
{schema_doc}

Current profile:
{profile_json}

Update the user's credit limit based on their account age and preferences.
Return ONLY valid JSON matching the schema.
"""
    
    response = llm_client.complete(prompt)
    
    # Translate FROM LLM: text → structured with validation
    try:
        raw_data = json.loads(response)
        validated = UserProfileSchema(**raw_data)
        
        # Reconstruct typed object
        return UserProfile(
            user_id=validated.user_id,
            email=validated.email,
            created_at=datetime.fromisoformat(validated.created_at),
            preferences=validated.preferences,
            credit_limit=validated.credit_limit
        )
    except (json.JSONDecodeError, ValueError) as e:
        raise ValidationError(f"LLM output failed validation: {e}")
```

The critical difference: Traditional APIs maintain type safety through the entire pipeline. LLM interactions serialize everything to text, losing type information, then require aggressive validation on reconstruction.

### Key Insights That Change Engineering Approach

**Insight 1: Trust Boundaries Are Bidirectional**
You must validate both outgoing metadata (can the LLM understand this?) and incoming responses (is the LLM's output safe to use?). The outgoing validation is often overlooked—engineers assume their data is "clean," but LLMs require different constraints than databases.

**Insight 2: Schema Is Documentation AND Constraint**
In traditional systems, schemas enforce constraints. With LLMs, schemas also serve as in-context documentation. This dual purpose means your validation rules must be both machine-executable and human-readable in natural language.

**Insight 3: Partial Success Is Common**
Unlike typed APIs that fail atomically, LLM outputs often contain partially valid data. Your validation strategy must handle scenarios like: 8 of 10 fields valid, fields present but wrong types, extra fields not in schema, or nested validation failures.

### Why This Matters Now

Production LLM systems are moving from simple text generation to structured data transformation. Recent use cases include:

- **Database schema migrations**: LLMs translate between incompatible schema versions
- **Multi-system integration**: LLMs map fields between systems with different naming conventions
- **Compliance validation**: Ensuring LLM-generated data meets regulatory requirements
- **Agent workflows**: Autonomous systems where LLMs produce data consumed by other services

Without robust metadata translation and validation, these systems fail silently, producing subtly corrupted data that propagates through pipelines. The cost of a 2% error rate in a million-record migration exceeds the engineering cost of proper validation.

## Technical Components

### 1. Schema Representation & Serialization

**Technical Explanation:** LLMs understand structured data through serialized schema representations—typically JSON Schema, TypeScript interfaces, or Pydantic models converted to descriptive text. The schema must balance machine parseability with LLM comprehension.

**Practical Implications:**

```python
from typing import Literal, Annotated
from pydantic import BaseModel, Field, ConfigDict
from enum import Enum

class PaymentStatus(str, Enum):
    PENDING = "pending"
    COMPLETED = "completed"
    FAILED = "failed"
    REFUNDED = "refunded"

class PaymentRecord(BaseModel):
    model_config = ConfigDict(use_enum_values=True)
    
    transaction_id: Annotated[str, Field(
        pattern=r'^TXN-\d{10}$',
        description="Transaction ID in format TXN-XXXXXXXXXX"
    )]
    amount_cents: Annotated[int, Field(
        gt=0,
        lt=1000000000,
        description="Payment amount in cents (1-999999999)"
    )]
    status: PaymentStatus
    idempotency_key: Annotated[str, Field(
        min_length=16,
        max_length=64,
        description="Client-provided idempotency key for deduplication"
    )]
    metadata: dict[str, str] = Field(
        default_factory=dict,
        description="Arbitrary key-value pairs, all values must be strings"
    )

def schema_to_llm_context(model_class: type[BaseModel]) -> str:
    """Convert Pydantic model to LLM-friendly schema documentation."""
    schema = model_class.model_json_schema()
    
    # Flatten nested schema into readable format
    lines = [f"Schema: {schema['title']}\n"]
    
    for field_name, field_info in schema['properties'].items():
        field_type = field_info.get('type', 'unknown')
        description = field_info.get('description', '')
        
        # Extract constraints
        constraints = []
        if 'pattern' in field_info:
            constraints.append(f"pattern={field_info['pattern']}")
        if 'minimum' in field_info:
            constraints.append(f"min={field_info['minimum']}")
        if 'maximum' in field_info:
            constraints.append(f"max={field_info['maximum']}")
        if 'minLength' in field_info:
            constraints.append(f"minLength={field_info['minLength']}")
        if 'maxLength' in field_info:
            constraints.append(f"maxLength={field_info['maxLength']}")
        if 'enum' in field_info:
            constraints.append(f"enum={field_info['enum']}")
            
        constraint_str = f" [{', '.join(constraints)}]" if constraints else ""
        
        lines.append(
            f"  {field_name}: {field_type}{constraint_str}\n"
            f"    {description}"
        )
    
    return "\n".join(lines)

# Usage
context = schema_to_llm_context(PaymentRecord)
print(context)
```

**Real Constraints & Trade-offs:**
- **Verbosity vs. Clarity**: Detailed schemas consume more tokens but reduce validation failures. Benchmark shows 15-20% reduction in validation errors with explicit constraints in schema documentation, at cost of 200-300 additional tokens.
- **JSON Schema Dialect**: Different LLMs perform better with different schema styles. OpenAI models prefer JSON Schema format, while Claude performs better with TypeScript-style interfaces.

### 2. Multi-Stage Validation Pipeline

**Technical Explanation:** Validation must occur in stages: syntactic (is it parseable?), structural (does it match schema?), semantic (do values make sense?), and relational (do cross-field constraints hold?).

**Practical Implementation:**

```python
from typing import TypeVar, Generic, Callable
from dataclasses import dataclass
from enum import Enum

T = TypeVar('T')

class ValidationSeverity(Enum):
    ERROR = "error"      # Must fix
    WARNING = "warning"  # Should fix
    INFO = "info"        # Nice to have

@dataclass
class ValidationResult:
    severity: ValidationSeverity
    field_path: str
    message: str
    original_value: Any
    suggested_fix: Optional[Any] = None

class ValidationPipeline(Generic[T]):
    def __init__(self, model_class: type[T]):
        self.model_class = model_class
        self.validators: list[Callable[[dict], list[ValidationResult]]] = []
    
    def add_validator(self, validator: Callable[[dict], list[ValidationResult]]):
        self.validators.append(validator)
        return self
    
    def validate(self, raw_data: dict) -> tuple[Optional[T], list[ValidationResult]]:
        all_results = []
        
        # Stage 1: Syntactic validation (JSON structure)
        if not isinstance(raw_data, dict):
            return None, [ValidationResult(
                severity=ValidationSeverity.ERROR,
                field_path="__root__",
                message="Expected JSON object",
                original_value=raw_data
            )]
        
        # Stage 2: Structural validation (Pydantic schema)
        try:
            instance = self.model_class(**raw_data)
        except Exception as e:
            return None, [ValidationResult(
                severity=ValidationSeverity.ERROR,
                field_path="__schema__",
                message=f"Schema validation failed: {str(e)}",
                original_value=raw_data
            )]
        
        # Stage 3 & 4: Semantic and relational validation
        for validator in self.validators:
            results = validator(raw_data)
            all_results.extend(results)
        
        # Determine if errors block instantiation
        has_errors = any(r.severity == ValidationSeverity.ERROR for r in all_results)
        return (None if has_errors else instance, all_results)

# Example: Payment validation pipeline
def validate_payment_semantics(data: dict) -> list[ValidationResult]:
    results = []
    
    # Semantic: amount should be reasonable for status
    if data.get('status') == 'completed' and data.get('amount_cents', 0) == 0:
        results.append(ValidationResult(
            severity=ValidationSeverity.WARNING,
            field_path="amount_cents",
            message="Completed payment with zero amount is unusual",
            original_value=data.get('amount_cents')
        ))
    
    # Semantic: refunded status requires original transaction reference
    if data.get('status') == 'refunded' and 'original_txn' not in data.get('metadata', {}):
        results.append(ValidationResult(
            severity=ValidationSeverity.ERROR,
            field_path="metadata.original_txn",
            message="Refunded payments must reference original transaction in metadata",
            original_value=data.get('metadata'),
            suggested_fix={'original_txn': 'TXN-XXXXXXXXXX'}
        ))
    
    return results

def validate_payment_relational(data: dict) -> list[ValidationResult]:
    results = []
    
    # Relational: idempotency key should match transaction ID pattern
    txn_id = data.get('transaction_id', '')
    idem_key = data.get('idempotency_key', '')
    
    if txn_id and idem_key and not idem_key.startswith(txn_id[:8]):
        results.append(ValidationResult(
            severity=ValidationSeverity.WARNING,
            field_path="idempotency_key",
            message="Idempotency key should incorporate transaction ID prefix",
            original_value=idem_key,
            suggested_fix=f"{txn_id[:8]}-{idem_key}"
        ))
    
    return results

# Build pipeline
pipeline = ValidationPipeline(PaymentRecord)
pipeline.add_validator(validate_payment_semantics)
pipeline.add_validator(validate_payment_relational)

# Validate LLM output
llm_output = {
    "transaction_id": "TXN-1234567890",
    "amount_cents": 5000,
    "status": "completed",
    "idempotency_key": "abcdef1234567890",
    "metadata": {}
}

validated_payment, issues = pipeline.validate(llm_output)
for issue in issues:
    print(f"[{issue.severity.value}] {issue.field_path}: {issue.message}")
```

**Real Constraints & Trade-offs:**
- **Performance**: Four-stage validation adds 5-15ms latency per object. For batch processing, parallelize with `asyncio` or `multiprocessing`.
- **False Positives**: Overly strict semantic validators create alert fatigue. Tune warning thresholds based on production data—aim for <5% warning rate.

### 3. Type Coercion & Recovery Strategies

**Technical Explanation:** LLMs frequently return correct data in wrong types (integers as strings, dates as epoch timestamps instead of ISO format). Intelligent coercion attempts safe type conversion before failing validation.

**Implementation with Recovery:**

```python
from typing import get_args, get_origin, Union
from datetime import datetime, date
import re

class CoercionStrategy:
    @staticmethod
    def coerce_to_int(value: Any) -> Optional[int]:
        """Try multiple coercion strategies for integers."""
        if isinstance(value, int):
            return value
        if isinstance(value, float):
            if value.is_integer():
                return int(value)
            return None
        if isinstance(value, str):
            # Remove common formatting
            cleaned = value.strip().replace(',', '').replace('_', '')
            try:
                return int(cleaned)
            except ValueError:
                # Try float then int (handles "123.0")
                try:
                    f = float(cleaned)
                    if f.is_integer():
                        return int(f)
                except ValueError:
                    pass
        return None
    
    @staticmethod
    def coerce_to_datetime(value: Any) -> Optional[datetime]:
        """Try multiple datetime parsing strategies."""
        if isinstance(value, datetime):
            return value
        if isinstance(value, date):
            return datetime.combine(value, datetime.min.time())
        if isinstance(value, (int, float)):
            # Assume Unix timestamp
            try:
                return datetime.fromtimestamp(value)
            except (ValueError, OSError):
                pass
        if isinstance(value, str):
            # Try ISO format
            try:
                return datetime.fromisoformat(value.replace('Z', '+00:00'))
            except ValueError:
                pass
            # Try common formats
            for fmt in [
                '%Y-%m-%d %H:%M:%S',
                '%Y-%m-%d',
                '%d/%m/%Y',
                '%m/%d/%Y',
            ]:
                try:
                    return datetime.strptime(value, fmt)
                except ValueError:
                    continue
        return None
    
    @staticmethod
    def coerce_to_bool(value: Any) -> Optional[bool]:
        """Handle string booleans and numeric booleans."""
        if isinstance(value, bool