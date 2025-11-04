# Database Schema Design & ERD Generation with LLMs

## Core Concepts

Database schema design traditionally requires deep domain knowledge, normalization expertise, and iterative refinement through stakeholder interviews. Large Language Models fundamentally change this workflow by transforming natural language requirements into structured schemas while applying design patterns learned from millions of examples.

### Traditional vs. LLM-Assisted Approach

**Traditional Approach:**
```python
# Manual schema design process
# 1. Stakeholder interviews (days/weeks)
# 2. Manual ERD sketching
# 3. Normalization analysis
# 4. SQL generation
# 5. Multiple review cycles

class TraditionalSchemaDesign:
    def __init__(self):
        self.stakeholder_notes = []
        self.entities = []
        self.relationships = []
    
    def conduct_interviews(self, stakeholders):
        # Manual, time-consuming process
        for stakeholder in stakeholders:
            self.stakeholder_notes.append(
                self.interview(stakeholder)  # Hours per stakeholder
            )
    
    def design_schema(self):
        # Expert applies normalization rules manually
        # High cognitive load, error-prone
        for note in self.stakeholder_notes:
            entities = self.extract_entities_manually(note)
            self.entities.extend(entities)
        return self.normalize_and_create_erd()  # Hours of work
```

**LLM-Assisted Approach:**
```python
from anthropic import Anthropic
from typing import Dict, List
import json

class LLMSchemaDesigner:
    def __init__(self, api_key: str):
        self.client = Anthropic(api_key=api_key)
        
    def design_schema(self, requirements: str) -> Dict:
        """
        Convert natural language requirements to schema in minutes.
        Automatically applies normalization and best practices.
        """
        prompt = f"""Analyze these requirements and generate a database schema.

Requirements:
{requirements}

Provide:
1. Entities with attributes and data types
2. Relationships with cardinality
3. Constraints (PK, FK, unique, not null)
4. Indexes for common queries
5. Brief justification for design decisions

Format as JSON with this structure:
{{
  "entities": [{{"name": "...", "attributes": [...], "primary_key": "..."}}],
  "relationships": [{{"from": "...", "to": "...", "type": "...", "cardinality": "..."}}],
  "indexes": [...],
  "design_rationale": "..."
}}"""

        response = self.client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=4000,
            messages=[{"role": "user", "content": prompt}]
        )
        
        return json.loads(response.content[0].text)
```

### Key Engineering Insights

**1. LLMs as Pattern Matchers, Not Database Theorists**

LLMs excel at recognizing common schema patterns from training data but may miss subtle domain-specific constraints. They've seen thousands of e-commerce, social media, and SaaS schemas, making them excellent starting points. However, they operate on statistical patterns, not formal database theory.

**2. Iterative Refinement Over Perfect First Pass**

The power isn't in generating perfect schemas immediately—it's in dramatically accelerating iteration cycles from hours to seconds. You can test multiple normalization strategies, explore denormalization trade-offs, and adjust cardinalities instantly.

**3. Domain Context is the Bottleneck**

The limiting factor shifts from technical schema design to articulating domain requirements clearly. An LLM can instantly apply 3NF normalization, but it needs you to specify business rules like "orders can't be modified after shipment" or "users can belong to multiple organizations with different roles."

### Why This Matters Now

Schema design bottlenecks kill project momentum. A two-week schema design phase compresses to two hours, but more critically, stakeholders can see and critique concrete schemas immediately rather than abstract ERD diagrams. This tight feedback loop catches requirement misunderstandings early when they're cheap to fix.

Modern applications demand rapid iteration. Microservices need isolated data models, feature flags require schema changes, and A/B tests spawn database variations. LLM-assisted design makes schema evolution a lightweight operation rather than a heavyweight architectural decision.

## Technical Components

### 1. Structured Output Extraction

LLMs generate unstructured text by default. Reliable schema generation requires enforcing JSON structure while preserving the model's reasoning capabilities.

**Technical Explanation:**

Structured outputs use constrained decoding or JSON schema validation to ensure LLM responses match specified formats. This prevents parsing failures and ensures consistent output structure for downstream tooling.

**Practical Implementation:**

```python
from anthropic import Anthropic
from pydantic import BaseModel, Field
from typing import List, Literal
import json

# Define strict schema using Pydantic
class Attribute(BaseModel):
    name: str
    data_type: str
    nullable: bool
    default: str | None = None
    constraints: List[str] = Field(default_factory=list)

class Entity(BaseModel):
    name: str
    attributes: List[Attribute]
    primary_key: List[str]

class Relationship(BaseModel):
    from_entity: str
    to_entity: str
    relationship_type: Literal["one-to-one", "one-to-many", "many-to-many"]
    from_cardinality: str
    to_cardinality: str

class DatabaseSchema(BaseModel):
    entities: List[Entity]
    relationships: List[Relationship]
    indexes: List[Dict[str, str]]

def generate_structured_schema(requirements: str, api_key: str) -> DatabaseSchema:
    """Generate schema with guaranteed structure."""
    client = Anthropic(api_key=api_key)
    
    schema_json = DatabaseSchema.model_json_schema()
    
    prompt = f"""Generate a database schema for these requirements:

{requirements}

Output valid JSON matching this exact schema:
{json.dumps(schema_json, indent=2)}

Ensure all entities have primary keys and relationships specify cardinality."""

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=4000,
        messages=[{"role": "user", "content": prompt}]
    )
    
    # Parse and validate against Pydantic model
    schema_dict = json.loads(response.content[0].text)
    return DatabaseSchema(**schema_dict)
```

**Constraints & Trade-offs:**

Strict schemas reduce creative solutions. An overly rigid structure might prevent the LLM from suggesting innovative designs like JSONB columns for flexible attributes. Balance structure with flexibility by making optional fields genuinely optional.

### 2. Multi-Turn Schema Refinement

Initial schemas often miss critical constraints or make incorrect assumptions. Multi-turn conversations allow probing specific aspects without regenerating the entire schema.

**Technical Explanation:**

Maintain conversation context to iteratively refine schemas. Each turn builds on previous context, allowing targeted questions about specific entities, relationships, or constraints.

**Practical Implementation:**

```python
from typing import List, Dict, Any

class SchemaRefiner:
    def __init__(self, api_key: str):
        self.client = Anthropic(api_key=api_key)
        self.conversation_history: List[Dict[str, str]] = []
        self.current_schema: Dict[str, Any] = {}
    
    def initial_design(self, requirements: str) -> Dict[str, Any]:
        """Generate initial schema."""
        prompt = f"""Design a database schema for:

{requirements}

Output as JSON with entities, relationships, and constraints."""
        
        self.conversation_history.append({
            "role": "user",
            "content": prompt
        })
        
        response = self.client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=4000,
            messages=self.conversation_history
        )
        
        schema_text = response.content[0].text
        self.conversation_history.append({
            "role": "assistant",
            "content": schema_text
        })
        
        self.current_schema = json.loads(schema_text)
        return self.current_schema
    
    def refine(self, refinement_request: str) -> Dict[str, Any]:
        """Refine specific aspects of the schema."""
        self.conversation_history.append({
            "role": "user",
            "content": refinement_request
        })
        
        response = self.client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=4000,
            messages=self.conversation_history
        )
        
        updated_schema_text = response.content[0].text
        self.conversation_history.append({
            "role": "assistant",
            "content": updated_schema_text
        })
        
        self.current_schema = json.loads(updated_schema_text)
        return self.current_schema
    
    def add_constraint(self, entity: str, constraint: str) -> Dict[str, Any]:
        """Add specific constraint to entity."""
        request = f"""Update the {entity} entity to include this constraint:
{constraint}

Return the complete updated schema as JSON."""
        return self.refine(request)
```

**Real Constraints:**

Conversation history consumes context window tokens. After 5-10 refinement turns, you may need to summarize the conversation or start fresh with the refined schema as the new baseline. Monitor token usage and condense history when approaching limits.

### 3. SQL Generation from Schema Definitions

Schema definitions must translate to executable SQL DDL statements. This requires handling database-specific syntax, constraint ordering, and dependency resolution.

**Technical Explanation:**

Different databases (PostgreSQL, MySQL, SQL Server) have varying syntax for constraints, data types, and indexes. LLMs can generate dialect-specific SQL when provided with the target database context.

**Practical Implementation:**

```python
from enum import Enum

class DatabaseDialect(Enum):
    POSTGRESQL = "postgresql"
    MYSQL = "mysql"
    SQLSERVER = "sqlserver"

def generate_sql_ddl(
    schema: Dict[str, Any],
    dialect: DatabaseDialect,
    api_key: str
) -> str:
    """Generate executable SQL DDL from schema definition."""
    client = Anthropic(api_key=api_key)
    
    dialect_hints = {
        DatabaseDialect.POSTGRESQL: """
- Use SERIAL for auto-increment
- Use TEXT instead of VARCHAR(MAX)
- Use JSONB for JSON columns
- Create indexes with CREATE INDEX CONCURRENTLY when possible
""",
        DatabaseDialect.MYSQL: """
- Use AUTO_INCREMENT for auto-increment
- Specify ENGINE=InnoDB
- Use VARCHAR with explicit length
- JSON type available in 5.7+
""",
        DatabaseDialect.SQLSERVER: """
- Use IDENTITY for auto-increment
- Use NVARCHAR for Unicode support
- Use schema prefix (dbo.)
- Clustered index on primary key
"""
    }
    
    prompt = f"""Generate SQL DDL statements for this schema:

{json.dumps(schema, indent=2)}

Target database: {dialect.value}
{dialect_hints[dialect]}

Requirements:
1. Create tables in correct dependency order
2. Include all constraints (PK, FK, unique, not null)
3. Add indexes for foreign keys and common queries
4. Include comments explaining design decisions
5. Make script idempotent (DROP IF EXISTS)

Output complete, executable SQL."""

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=6000,
        messages=[{"role": "user", "content": prompt}]
    )
    
    return response.content[0].text
```

**Practical Implications:**

Generated SQL often requires review for production use. LLMs may use deprecated syntax, miss database-specific optimizations (like PostgreSQL partial indexes), or generate suboptimal constraint ordering. Always review and test against a development database before applying to production.

### 4. Visual ERD Generation

Entity-Relationship Diagrams communicate schema structure to stakeholders. LLMs can generate ERDs in text-based formats like Mermaid or PlantUML that render to visual diagrams.

**Technical Explanation:**

Mermaid and PlantUML use text syntax to define diagrams. LLMs excel at generating these formats, which can be rendered in documentation tools, Markdown viewers, or dedicated diagram applications.

**Practical Implementation:**

```python
def generate_erd_diagram(
    schema: Dict[str, Any],
    format: Literal["mermaid", "plantuml"],
    api_key: str
) -> str:
    """Generate visual ERD in text-based diagram format."""
    client = Anthropic(api_key=api_key)
    
    format_examples = {
        "mermaid": """
erDiagram
    CUSTOMER ||--o{ ORDER : places
    ORDER ||--|{ LINE-ITEM : contains
    CUSTOMER {
        string id PK
        string name
        string email UK
    }
""",
        "plantuml": """
@startuml
entity "Customer" as customer {
    *id : int <<PK>>
    --
    name : varchar(100)
    email : varchar(255) <<UK>>
}
@enduml
"""
    }
    
    prompt = f"""Generate an ERD diagram for this schema:

{json.dumps(schema, indent=2)}

Format: {format}

Example syntax:
{format_examples[format]}

Include:
- All entities and attributes
- Relationships with cardinality
- Primary keys (PK) and foreign keys (FK)
- Unique constraints (UK)
- Clear entity positioning for readability"""

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=4000,
        messages=[{"role": "user", "content": prompt}]
    )
    
    return response.content[0].text

# Usage example with rendering
def save_mermaid_diagram(diagram: str, output_path: str):
    """Save Mermaid diagram as HTML for viewing."""
    html_template = f"""
<!DOCTYPE html>
<html>
<head>
    <script src="https://cdn.jsdelivr.net/npm/mermaid/dist/mermaid.min.js"></script>
    <script>mermaid.initialize({{startOnLoad:true}});</script>
</head>
<body>
    <div class="mermaid">
{diagram}
    </div>
</body>
</html>
"""
    with open(output_path, 'w') as f:
        f.write(html_template)
```

**Trade-offs:**

Text-based diagrams excel at version control and automation but offer less layout control than visual tools. Complex schemas may require manual layout adjustments. For schemas with 10+ entities, consider generating multiple focused diagrams rather than one comprehensive ERD.

### 5. Schema Validation and Anti-Pattern Detection

LLMs can review schemas for common anti-patterns, normalization issues, and design flaws before implementation.

**Technical Explanation:**

By prompting LLMs with established database design principles, you can create an automated reviewer that identifies issues like missing indexes, circular dependencies, or denormalization without clear justification.

**Practical Implementation:**

```python
from typing import List

class SchemaValidator:
    def __init__(self, api_key: str):
        self.client = Anthropic(api_key=api_key)
    
    def validate_schema(self, schema: Dict[str, Any]) -> List[Dict[str, str]]:
        """Identify design issues and anti-patterns."""
        validation_prompt = f"""Review this database schema for issues:

{json.dumps(schema, indent=2)}

Check for:
1. Normalization violations (1NF, 2NF, 3NF)
2. Missing indexes on foreign keys
3. Circular dependencies
4. Overly generic names (data, info, value)
5. Missing constraints (not null on required fields)
6. Many-to-many without junction tables
7. Large VARCHAR fields that should be TEXT
8. Missing created_at/updated_at audit fields
9. Composite primary keys without clear justification
10. Missing unique constraints on natural keys

For each issue found, provide:
- Severity (critical/warning/suggestion)
- Entity/relationship affected
- Description of the issue
- Recommended fix

Output as JSON array:
[{{"severity": "...", "location": "...", "issue": "...", "fix": "..."}}]"""

        response = self.client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=4000,
            messages