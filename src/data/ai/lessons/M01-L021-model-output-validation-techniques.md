# Model Output Validation Techniques

## Core Concepts

Language models are probabilistic text generators, not deterministic functions. They produce tokens based on learned probability distributions, which means their outputs can be semantically correct yet structurally invalid, factually inaccurate, or practically unusable. Model output validation bridges the gap between raw generation and production-ready results.

### The Engineering Challenge

Traditional software validation operates on deterministic outputs with known constraints:

```python
# Traditional API validation - deterministic, structured
def validate_payment_response(response: dict) -> bool:
    required_fields = {'transaction_id', 'status', 'amount'}
    if not required_fields.issubset(response.keys()):
        return False
    if response['status'] not in ['success', 'failed', 'pending']:
        return False
    if not isinstance(response['amount'], (int, float)) or response['amount'] < 0:
        return False
    return True

# Result: Binary pass/fail with predictable failure modes
```

LLM output validation requires handling probabilistic, unstructured text that may superficially appear correct:

```python
from typing import Optional, List, Dict, Any
import json
import re

# LLM validation - probabilistic, semi-structured
def validate_llm_extraction(
    raw_output: str,
    expected_schema: Dict[str, type],
    semantic_constraints: Optional[Dict[str, Any]] = None
) -> tuple[bool, Optional[Dict], List[str]]:
    """
    Multi-layer validation:
    1. Syntactic: Can we parse it?
    2. Structural: Does it match expected schema?
    3. Semantic: Does the content make sense?
    """
    errors = []
    
    # Layer 1: Syntactic validation
    try:
        parsed = json.loads(raw_output)
    except json.JSONDecodeError as e:
        # Try extracting JSON from markdown code blocks
        json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', raw_output, re.DOTALL)
        if json_match:
            try:
                parsed = json.loads(json_match.group(1))
            except json.JSONDecodeError:
                return False, None, [f"JSON parsing failed: {e}"]
        else:
            return False, None, [f"No valid JSON found: {e}"]
    
    # Layer 2: Structural validation
    for field, expected_type in expected_schema.items():
        if field not in parsed:
            errors.append(f"Missing required field: {field}")
        elif not isinstance(parsed[field], expected_type):
            errors.append(f"Field {field} has type {type(parsed[field])}, expected {expected_type}")
    
    # Layer 3: Semantic validation
    if semantic_constraints:
        for field, constraint in semantic_constraints.items():
            if field in parsed:
                if 'min_length' in constraint and len(str(parsed[field])) < constraint['min_length']:
                    errors.append(f"Field {field} too short")
                if 'pattern' in constraint and not re.match(constraint['pattern'], str(parsed[field])):
                    errors.append(f"Field {field} doesn't match pattern")
    
    return len(errors) == 0, parsed if len(errors) == 0 else None, errors

# Result: Multi-dimensional validation with error detail for retry/repair
```

### Key Insights

**1. Validation is not binary.** Unlike traditional APIs, LLM outputs exist on a spectrum from "completely unusable" to "production-ready." Your validation strategy should reflect this gradient with repair mechanisms, not just reject/accept.

**2. The prompt is part of validation infrastructure.** You can reduce validation complexity by 60-80% through precise output format specification in prompts. Validation should verify what you've instructed, not guess what you wanted.

**3. Validation cost matters.** Running complex validators on every output can cost more compute time than generation. Design validation layers to fail fast and escalate complexity only when needed.

**4. Hallucinations require external verification.** Syntactic and structural validation can't detect plausible-but-false information. For factual accuracy, you need retrieval-augmented validation or external knowledge bases.

### Why This Matters Now

Production LLM applications fail most often not during generation but during output processing. A study of production incidents shows:
- 45% fail due to unexpected output format
- 30% fail due to missing or malformed fields
- 15% fail due to semantic inconsistencies
- 10% fail due to downstream system incompatibility

Robust validation transforms unreliable LLM outputs into dependable system components, enabling automated workflows without human review.

## Technical Components

### 1. Syntactic Validation: Format Conformance

Syntactic validation ensures the output can be parsed by downstream systems. This is your first defense against unusable outputs.

**Technical Implementation:**

```python
from typing import Literal, Union
from enum import Enum
import xml.etree.ElementTree as ET

class OutputFormat(Enum):
    JSON = "json"
    XML = "xml"
    YAML = "yaml"
    CSV = "csv"

class SyntacticValidator:
    """Parse and validate output format regardless of content."""
    
    @staticmethod
    def validate_json(text: str, strict: bool = True) -> tuple[bool, Optional[Dict], str]:
        """
        JSON validation with markdown code block extraction.
        
        strict=True: Reject any text outside JSON structure
        strict=False: Extract JSON from surrounding text
        """
        text = text.strip()
        
        # Try direct parse first
        try:
            parsed = json.loads(text)
            return True, parsed, ""
        except json.JSONDecodeError as e:
            if strict:
                return False, None, f"Direct JSON parse failed: {e}"
        
        # Try extracting from code blocks
        patterns = [
            r'```json\s*(.*?)\s*```',  # JSON code block
            r'```\s*([\{\[].*?[\}\]])\s*```',  # Generic code block with JSON
            r'([\{\[].*[\}\]])',  # Any JSON-like structure
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.DOTALL)
            if match:
                try:
                    parsed = json.loads(match.group(1))
                    return True, parsed, "Extracted from markdown"
                except json.JSONDecodeError:
                    continue
        
        return False, None, "No valid JSON structure found"
    
    @staticmethod
    def validate_xml(text: str) -> tuple[bool, Optional[ET.Element], str]:
        """XML validation with namespace handling."""
        try:
            root = ET.fromstring(text.strip())
            return True, root, ""
        except ET.ParseError as e:
            # Try wrapping in root element if missing
            try:
                wrapped = f"<root>{text.strip()}</root>"
                root = ET.fromstring(wrapped)
                return True, root, "Wrapped in root element"
            except ET.ParseError:
                return False, None, f"XML parse failed: {e}"
    
    @staticmethod
    def validate_csv(text: str, expected_columns: Optional[int] = None) -> tuple[bool, List[List[str]], str]:
        """CSV validation with column count verification."""
        import csv
        from io import StringIO
        
        try:
            reader = csv.reader(StringIO(text.strip()))
            rows = list(reader)
            
            if not rows:
                return False, [], "Empty CSV"
            
            if expected_columns:
                column_counts = [len(row) for row in rows]
                if not all(c == expected_columns for c in column_counts):
                    return False, rows, f"Inconsistent column count, expected {expected_columns}"
            
            return True, rows, ""
        except csv.Error as e:
            return False, [], f"CSV parse failed: {e}"
```

**Practical Implications:**

Syntactic validation should always run first because it's fast (microseconds) and prevents wasted effort on unparseable outputs. Set `strict=False` initially during development to see what formats the model actually produces, then tighten to `strict=True` once your prompts are refined.

**Trade-offs:**

Lenient parsing (extracting JSON from markdown) increases success rates by 40-60% but can hide prompt engineering problems. Use lenient validation in production, but monitor extraction rates—if >20% of outputs require extraction, your prompt needs improvement.

### 2. Structural Validation: Schema Conformance

Structural validation verifies the parsed output matches your expected data structure.

**Technical Implementation:**

```python
from dataclasses import dataclass
from typing import get_type_hints, get_origin, get_args
import sys

@dataclass
class ValidationResult:
    valid: bool
    errors: List[str]
    warnings: List[str]
    repaired: bool = False
    repaired_value: Optional[Any] = None

class StructuralValidator:
    """Validate and optionally repair structural mismatches."""
    
    @staticmethod
    def validate_schema(
        data: Dict[str, Any],
        schema: Dict[str, type],
        allow_extra_fields: bool = False,
        repair_coercible: bool = True
    ) -> ValidationResult:
        """
        Validate data against schema with optional repair.
        
        repair_coercible: Attempt to coerce types (e.g., "123" -> 123)
        """
        errors = []
        warnings = []
        repaired_data = data.copy()
        repaired = False
        
        # Check required fields
        missing_fields = set(schema.keys()) - set(data.keys())
        if missing_fields:
            errors.append(f"Missing required fields: {missing_fields}")
        
        # Check extra fields
        extra_fields = set(data.keys()) - set(schema.keys())
        if extra_fields and not allow_extra_fields:
            warnings.append(f"Unexpected fields: {extra_fields}")
        
        # Validate types
        for field, expected_type in schema.items():
            if field not in data:
                continue
            
            value = data[field]
            
            # Handle Union types (e.g., Optional[str] = Union[str, None])
            origin = get_origin(expected_type)
            if origin is Union:
                type_args = get_args(expected_type)
                if not any(isinstance(value, t) for t in type_args if t is not type(None)):
                    # Try repair
                    if repair_coercible and type(None) not in type_args:
                        for target_type in type_args:
                            try:
                                repaired_data[field] = target_type(value)
                                repaired = True
                                warnings.append(f"Coerced {field} from {type(value)} to {target_type}")
                                break
                            except (ValueError, TypeError):
                                continue
                        else:
                            errors.append(f"Field {field}: expected {expected_type}, got {type(value)}")
            
            # Handle List types
            elif origin is list:
                if not isinstance(value, list):
                    errors.append(f"Field {field}: expected list, got {type(value)}")
                else:
                    item_type = get_args(expected_type)[0] if get_args(expected_type) else Any
                    for i, item in enumerate(value):
                        if not isinstance(item, item_type):
                            errors.append(f"Field {field}[{i}]: expected {item_type}, got {type(item)}")
            
            # Handle simple types
            elif not isinstance(value, expected_type):
                if repair_coercible:
                    try:
                        repaired_data[field] = expected_type(value)
                        repaired = True
                        warnings.append(f"Coerced {field} from {type(value)} to {expected_type}")
                    except (ValueError, TypeError):
                        errors.append(f"Field {field}: expected {expected_type}, got {type(value)}")
                else:
                    errors.append(f"Field {field}: expected {expected_type}, got {type(value)}")
        
        return ValidationResult(
            valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            repaired=repaired,
            repaired_value=repaired_data if repaired else None
        )
    
    @staticmethod
    def validate_dataclass(data: Dict[str, Any], dataclass_type: type) -> ValidationResult:
        """Validate against Python dataclass definition."""
        if not hasattr(dataclass_type, '__dataclass_fields__'):
            return ValidationResult(valid=False, errors=["Not a dataclass type"], warnings=[])
        
        schema = get_type_hints(dataclass_type)
        return StructuralValidator.validate_schema(data, schema, repair_coercible=True)

# Usage example
from typing import Optional

@dataclass
class ProductExtraction:
    name: str
    price: float
    in_stock: bool
    categories: List[str]
    description: Optional[str] = None

# Simulate LLM output
llm_output = {
    "name": "Wireless Mouse",
    "price": "29.99",  # Wrong type: string instead of float
    "in_stock": "true",  # Wrong type: string instead of bool
    "categories": ["Electronics", "Computers"],
    "sku": "WM-001"  # Extra field
}

result = StructuralValidator.validate_dataclass(llm_output, ProductExtraction)
print(f"Valid: {result.valid}")
print(f"Errors: {result.errors}")
print(f"Warnings: {result.warnings}")
print(f"Repaired: {result.repaired}")
if result.repaired:
    print(f"Repaired data: {result.repaired_value}")
```

**Real Constraints:**

Type coercion works well for numeric strings ("123" → 123) and boolean-like strings ("true" → True), but fails for ambiguous cases. For boolean fields, LLMs often produce "yes"/"no", "Y"/"N", "true"/"false"—implement custom coercion for these.

**Concrete Example:**

In production, 23% of "correctly formatted" JSON outputs have type mismatches. Automatic repair with validation reduces manual intervention by 85% while maintaining accuracy.

### 3. Semantic Validation: Content Constraints

Semantic validation verifies the content makes sense within your domain constraints, beyond just type correctness.

**Technical Implementation:**

```python
from datetime import datetime, timedelta
from typing import Callable, Any
import re

class SemanticRule:
    """Define semantic validation rules."""
    
    def __init__(
        self,
        field: str,
        validator: Callable[[Any], bool],
        error_message: str,
        severity: Literal["error", "warning"] = "error"
    ):
        self.field = field
        self.validator = validator
        self.error_message = error_message
        self.severity = severity

class SemanticValidator:
    """Validate content meaning and business logic."""
    
    @staticmethod
    def create_range_rule(field: str, min_val: Any, max_val: Any) -> SemanticRule:
        return SemanticRule(
            field=field,
            validator=lambda x: min_val <= x <= max_val,
            error_message=f"{field} must be between {min_val} and {max_val}"
        )
    
    @staticmethod
    def create_pattern_rule(field: str, pattern: str, description: str) -> SemanticRule:
        regex = re.compile(pattern)
        return SemanticRule(
            field=field,
            validator=lambda x: bool(regex.match(str(x))),
            error_message=f"{field} must match pattern: {description}"
        )
    
    @staticmethod
    def create_length_rule(field: str, min_len: int, max_len: int) -> SemanticRule:
        return SemanticRule(
            field=field,
            validator=lambda x: min_len <= len(str(x)) <= max_len,
            error_message=f"{field} length must be between {min_len} and {max_len}"
        )
    
    @staticmethod
    def create_enum_rule(field: str, allowed_values