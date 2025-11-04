# Schema Mapping Automation with LLMs

## Core Concepts

Schema mapping—the process of translating data from one structured format to another—has traditionally been one of the most time-consuming tasks in data engineering. Whether you're migrating databases, integrating third-party APIs, or consolidating data lakes, you've likely spent hours writing transformation logic, handling edge cases, and maintaining brittle mapping configurations.

### Traditional vs. LLM-Assisted Approach

**Traditional Schema Mapping:**

```python
# Traditional hard-coded mapping logic
def map_customer_record(source_data: dict) -> dict:
    """Map from legacy CRM to new customer database schema."""
    target = {}
    
    # Direct field mappings
    target['customer_id'] = source_data['cust_no']
    target['email'] = source_data['email_addr']
    
    # Complex transformations
    target['full_name'] = f"{source_data['fname']} {source_data['lname']}"
    
    # Conditional logic
    if source_data['cust_type'] == 'B':
        target['customer_type'] = 'business'
    elif source_data['cust_type'] == 'I':
        target['customer_type'] = 'individual'
    else:
        target['customer_type'] = 'unknown'
    
    # Date format conversion
    target['registration_date'] = datetime.strptime(
        source_data['reg_dt'], '%Y%m%d'
    ).isoformat()
    
    return target
```

This approach requires:
- Manual analysis of source and target schemas
- Hand-coding every transformation rule
- Updating code when schemas change
- Separate logic for each schema pair

**LLM-Assisted Schema Mapping:**

```python
from typing import Dict, Any
import json
from openai import OpenAI

def generate_schema_mapping(
    source_schema: Dict[str, Any],
    target_schema: Dict[str, Any],
    sample_data: Dict[str, Any],
    client: OpenAI
) -> Dict[str, str]:
    """
    Generate mapping rules between schemas using an LLM.
    Returns a mapping configuration that can be saved and reused.
    """
    prompt = f"""Given these schemas, generate a JSON mapping configuration.

SOURCE SCHEMA:
{json.dumps(source_schema, indent=2)}

TARGET SCHEMA:
{json.dumps(target_schema, indent=2)}

SAMPLE SOURCE DATA:
{json.dumps(sample_data, indent=2)}

Generate a mapping configuration as JSON with this structure:
{{
  "field_mappings": {{
    "target_field": {{
      "source_field": "source_field_name",
      "transformation": "description of any transformation needed",
      "transformation_type": "direct|concat|conditional|format"
    }}
  }}
}}

Consider field names, data types, and semantic meaning. Be specific about transformations."""

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a data engineering expert specializing in schema mapping."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.1
    )
    
    mapping_config = json.loads(response.choices[0].message.content)
    return mapping_config
```

### Key Insights That Change Your Approach

1. **From Code to Configuration**: Instead of writing transformation logic for every schema pair, you generate reusable mapping configurations that can be versioned, audited, and modified without code changes.

2. **Semantic Understanding Over Pattern Matching**: LLMs understand that "cust_no", "customer_id", and "client_identifier" likely represent the same concept, even without explicit rules. This reduces the "long tail" of edge cases you'd otherwise handle manually.

3. **Schema Evolution Becomes Manageable**: When schemas change, regenerating mappings takes minutes instead of hours of code refactoring. The LLM can identify which fields are new, removed, or renamed.

4. **Documentation as a Feature**: LLM-generated mappings naturally include explanations of transformation logic, making them self-documenting and easier for teams to review.

### Why This Matters Now

The combination of structured outputs (JSON mode), improved reasoning capabilities, and lower costs has made LLM-assisted schema mapping economically viable for production systems. You're not replacing human judgment—you're automating the mechanical translation work while keeping humans in the loop for validation and edge cases. For teams managing dozens or hundreds of data integrations, this represents a 5-10x reduction in mapping maintenance overhead.

## Technical Components

### 1. Schema Representation and Normalization

LLMs work best with structured, consistently formatted schema descriptions. The quality of your schema representation directly impacts mapping accuracy.

**Technical Implementation:**

```python
from typing import Dict, List, Optional, Literal
from dataclasses import dataclass, asdict
import json

@dataclass
class FieldSchema:
    """Normalized field schema representation."""
    name: str
    data_type: str
    nullable: bool
    description: Optional[str] = None
    constraints: Optional[Dict[str, Any]] = None
    sample_values: Optional[List[Any]] = None

@dataclass
class TableSchema:
    """Complete schema representation."""
    name: str
    fields: List[FieldSchema]
    description: Optional[str] = None
    primary_keys: Optional[List[str]] = None

def normalize_json_schema(raw_schema: Dict) -> TableSchema:
    """Convert various schema formats to normalized representation."""
    fields = []
    
    # Handle JSON Schema format
    if 'properties' in raw_schema:
        for field_name, field_def in raw_schema['properties'].items():
            fields.append(FieldSchema(
                name=field_name,
                data_type=field_def.get('type', 'string'),
                nullable=field_name not in raw_schema.get('required', []),
                description=field_def.get('description'),
                constraints=field_def.get('constraints')
            ))
    
    # Handle database schema format
    elif 'columns' in raw_schema:
        for col in raw_schema['columns']:
            fields.append(FieldSchema(
                name=col['name'],
                data_type=col['type'],
                nullable=col.get('nullable', True),
                description=col.get('comment')
            ))
    
    return TableSchema(
        name=raw_schema.get('name', 'table'),
        fields=fields,
        description=raw_schema.get('description')
    )

def schema_to_llm_format(schema: TableSchema) -> str:
    """Format schema for optimal LLM processing."""
    lines = [f"Table: {schema.name}"]
    if schema.description:
        lines.append(f"Description: {schema.description}")
    
    lines.append("\nFields:")
    for field in schema.fields:
        nullable_str = "nullable" if field.nullable else "required"
        line = f"  - {field.name} ({field.data_type}, {nullable_str})"
        if field.description:
            line += f": {field.description}"
        if field.sample_values:
            line += f" [examples: {', '.join(map(str, field.sample_values[:3]))}]"
        lines.append(line)
    
    return "\n".join(lines)
```

**Practical Implications:**

- **Sample values matter**: Including 2-3 real sample values improves mapping accuracy by 30-40%, especially for ambiguous field names
- **Descriptions are high-leverage**: A single sentence description can eliminate entire classes of mapping errors
- **Type information prevents mismatches**: Explicit data types help LLMs avoid mapping string fields to numeric fields

**Real Constraints:**

- LLMs can hallucinate field names if schemas are too long (>50 fields). Split large schemas into logical groups.
- Inconsistent naming conventions within a schema (camelCase mixed with snake_case) confuse semantic matching.

### 2. Mapping Generation with Structured Outputs

Structured outputs ensure LLMs return valid, parseable mapping configurations every time.

```python
from pydantic import BaseModel, Field
from typing import Literal, Optional
from openai import OpenAI

class FieldMapping(BaseModel):
    """Single field mapping specification."""
    source_field: Optional[str] = Field(
        description="Source field name, null if computed/constant"
    )
    transformation_type: Literal[
        "direct", "concat", "split", "conditional", 
        "format_date", "format_number", "lookup", "constant"
    ]
    transformation_spec: Optional[Dict[str, Any]] = Field(
        description="Transformation parameters"
    )
    confidence: float = Field(
        ge=0.0, le=1.0,
        description="Confidence in this mapping (0-1)"
    )
    notes: Optional[str] = Field(
        description="Explanation of mapping logic"
    )

class SchemaMappingConfig(BaseModel):
    """Complete mapping configuration."""
    mappings: Dict[str, FieldMapping]
    unmapped_source_fields: List[str]
    unmapped_target_fields: List[str]
    warnings: List[str]

def generate_mapping_with_structure(
    source_schema: TableSchema,
    target_schema: TableSchema,
    sample_data: Optional[Dict] = None,
    client: OpenAI = None
) -> SchemaMappingConfig:
    """Generate structured mapping configuration."""
    
    system_prompt = """You are a data mapping specialist. Generate precise field mappings between schemas.

Transformation types:
- direct: one-to-one field copy
- concat: combine multiple fields (spec: {"fields": [], "separator": ""})
- split: split one field (spec: {"pattern": "", "index": 0})
- conditional: if-then logic (spec: {"conditions": []})
- format_date: convert date format (spec: {"input_format": "", "output_format": ""})
- format_number: numeric conversion (spec: {"precision": 2, "type": "decimal"})
- lookup: value mapping (spec: {"mappings": {}})
- constant: fixed value (spec: {"value": ""})

Set confidence based on semantic similarity and type compatibility."""

    user_prompt = f"""Map these schemas:

SOURCE:
{schema_to_llm_format(source_schema)}

TARGET:
{schema_to_llm_format(target_schema)}
"""

    if sample_data:
        user_prompt += f"\n\nSAMPLE DATA:\n{json.dumps(sample_data, indent=2)}"

    response = client.beta.chat.completions.parse(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        response_format=SchemaMappingConfig,
        temperature=0.1
    )
    
    return response.choices[0].message.parsed
```

**Practical Implications:**

- **Confidence scores enable human-in-the-loop**: Review mappings with confidence <0.7 before deploying
- **Structured specs make execution deterministic**: Transformation specs provide exact parameters for data processing
- **Unmapped fields flag schema mismatches**: Immediately identifies missing or incompatible fields

**Trade-offs:**

- Structured outputs add ~200-300ms latency but eliminate parsing errors
- More specific transformation types improve accuracy but require more comprehensive execution logic
- Pydantic validation catches schema errors but adds dependency weight

### 3. Mapping Execution Engine

Generated mappings must be executed reliably on actual data.

```python
from datetime import datetime
from typing import Any, Dict
import re

class MappingExecutor:
    """Execute mapping configurations on data."""
    
    def __init__(self, mapping_config: SchemaMappingConfig):
        self.config = mapping_config
        self.execution_stats = {
            "total_records": 0,
            "successful_records": 0,
            "field_errors": {}
        }
    
    def execute(self, source_record: Dict[str, Any]) -> Dict[str, Any]:
        """Transform a single source record to target format."""
        target_record = {}
        
        for target_field, mapping in self.config.mappings.items():
            try:
                value = self._execute_mapping(source_record, mapping)
                target_record[target_field] = value
            except Exception as e:
                # Track errors by field
                if target_field not in self.execution_stats["field_errors"]:
                    self.execution_stats["field_errors"][target_field] = []
                self.execution_stats["field_errors"][target_field].append(str(e))
                
                # Set null for failed mappings
                target_record[target_field] = None
        
        self.execution_stats["total_records"] += 1
        if not any(target_record[f] is None for f in target_record):
            self.execution_stats["successful_records"] += 1
        
        return target_record
    
    def _execute_mapping(
        self, 
        source_record: Dict[str, Any], 
        mapping: FieldMapping
    ) -> Any:
        """Execute a single field mapping."""
        
        if mapping.transformation_type == "direct":
            return source_record.get(mapping.source_field)
        
        elif mapping.transformation_type == "concat":
            spec = mapping.transformation_spec
            fields = spec["fields"]
            separator = spec.get("separator", " ")
            values = [str(source_record.get(f, "")) for f in fields]
            return separator.join(v for v in values if v)
        
        elif mapping.transformation_type == "split":
            spec = mapping.transformation_spec
            value = source_record.get(mapping.source_field, "")
            pattern = spec.get("pattern", r"\s+")
            index = spec.get("index", 0)
            parts = re.split(pattern, str(value))
            return parts[index] if index < len(parts) else None
        
        elif mapping.transformation_type == "format_date":
            spec = mapping.transformation_spec
            value = source_record.get(mapping.source_field)
            if not value:
                return None
            
            # Parse input format
            input_fmt = spec.get("input_format", "%Y%m%d")
            output_fmt = spec.get("output_format", "%Y-%m-%d")
            
            dt = datetime.strptime(str(value), input_fmt)
            return dt.strftime(output_fmt)
        
        elif mapping.transformation_type == "lookup":
            spec = mapping.transformation_spec
            value = source_record.get(mapping.source_field)
            mappings = spec.get("mappings", {})
            default = spec.get("default", None)
            return mappings.get(str(value), default)
        
        elif mapping.transformation_type == "constant":
            spec = mapping.transformation_spec
            return spec.get("value")
        
        elif mapping.transformation_type == "conditional":
            spec = mapping.transformation_spec
            conditions = spec.get("conditions", [])
            
            for condition in conditions:
                field = condition["field"]
                operator = condition["operator"]
                compare_value = condition["value"]
                result = condition["result"]
                
                source_value = source_record.get(field)
                
                if operator == "equals" and source_value == compare_value:
                    return result
                elif operator == "contains" and compare_value in str(source_value):
                    return result
                elif operator == "gt" and source_value > compare_value:
                    return result
            
            return spec.get("default", None)
        
        else:
            raise ValueError(f"Unknown transformation type: {mapping.transformation_type}")
    
    def get_error_report(self) -> Dict[str, Any]:
        """Generate execution statistics and error report."""
        success_rate = (
            self.execution_stats["successful_records"] / 
            self.execution_stats["total_records"]
            if self.execution_stats["total_records"] > 0
            else 0
        )
        
        return {
            "success_rate": success_rate,
            "total_records": self.execution_stats["total_records"],
            "successful_records": self.execution_stats["successful_records"],
            "field_errors": {
                field: {
                    "count": len(errors),
                    "sample_errors": errors[:5]
                }
                for field, errors