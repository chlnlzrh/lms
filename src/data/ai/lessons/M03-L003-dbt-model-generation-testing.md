# dbt Model Generation & Testing with LLMs

## Core Concepts

### Technical Definition

Using Large Language Models to generate SQL transformations and testing logic for dbt (data build tool) projects involves leveraging language models to translate natural language requirements into validated, production-ready SQL code and associated test definitions. This differs from traditional code generation by understanding both SQL semantics and dbt's specific framework conventions—including materialization strategies, dependencies, and testing patterns.

### Engineering Analogy: Code Comparison

**Traditional Approach:**
```python
# Manual dbt model creation - 30-45 minutes per model
"""
1. Analyst writes requirements in ticket
2. Engineer interprets requirements
3. Engineer writes SQL with dbt Jinja
4. Engineer writes schema.yml tests
5. Engineer documents columns manually
6. Back-and-forth on business logic
"""

# Result: models/marts/fct_orders.sql
{{
  config(
    materialized='incremental',
    unique_key='order_id'
  )
}}

select
    order_id,
    customer_id,
    order_date,
    -- Engineer had to guess column names from source
    -- No automatic documentation
    -- Tests written separately in YAML
from {{ ref('stg_orders') }}
```

**LLM-Assisted Approach:**
```python
# LLM-assisted generation - 5-10 minutes per model
from typing import Dict, List
import anthropic
import re

def generate_dbt_model(
    requirement: str,
    source_schemas: Dict[str, List[Dict[str, str]]],
    model_type: str = "incremental"
) -> Dict[str, str]:
    """
    Generate complete dbt model with tests and documentation.
    
    Args:
        requirement: Natural language model requirement
        source_schemas: Dict of source tables with column metadata
        model_type: Materialization strategy
    
    Returns:
        Dict with 'sql', 'schema', and 'documentation' keys
    """
    client = anthropic.Anthropic()
    
    prompt = f"""Generate a production-ready dbt model with the following:

REQUIREMENT:
{requirement}

AVAILABLE SOURCES:
{format_schema_context(source_schemas)}

OUTPUT FORMAT:
1. SQL model with dbt Jinja
2. schema.yml with tests
3. Column-level documentation

CONSTRAINTS:
- Use {model_type} materialization
- Include appropriate unique/not_null tests
- Add relationship tests for foreign keys
- Document all business logic in descriptions
"""
    
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=4000,
        messages=[{"role": "user", "content": prompt}]
    )
    
    return parse_dbt_output(response.content[0].text)

# Result: Complete model + tests + docs in minutes
# - Accurate column names from schema context
# - Appropriate tests for data quality
# - Documentation explaining transformations
```

### Key Insights That Change Engineering Thinking

**1. Schema Context Eliminates Ambiguity**

Traditional approaches require engineers to manually explore source tables, leading to errors and rework. LLMs can process complete schema metadata—column names, types, constraints, existing documentation—to generate accurate references and appropriate type casting without trial-and-error.

**2. Test Generation Follows Patterns**

dbt tests follow predictable patterns based on column types and business context. An LLM trained on sufficient examples can infer that `email` columns need format validation, `amount` columns need non-negative checks, and foreign keys need relationship tests—patterns that engineers often forget or apply inconsistently.

**3. Incremental Logic Is Algorithmically Complex**

Writing correct incremental models with merge logic, late-arriving data handling, and efficient filtering requires deep dbt knowledge. LLMs encode these patterns better than most engineers remember them, especially for edge cases like handling updates vs. inserts or implementing proper lookback windows.

### Why This Matters NOW

**Velocity Without Quality Sacrifice:** Data teams face constant pressure to model more data faster. Manual SQL writing bottlenecks pipelines, but naive code generation creates technical debt. LLMs now understand enough about SQL semantics and dbt conventions to generate production-quality code that passes review.

**Knowledge Distribution:** dbt best practices live in documentation, Discourse posts, and senior engineers' heads. LLMs trained on this corpus democratize expertise, allowing mid-level engineers to implement advanced patterns (slowly changing dimensions, event sessionization) without extensive mentorship.

**Testing Coverage Gap:** Most teams under-test their transformations due to time pressure. Automated test generation based on schema analysis and business rules closes this gap, catching data quality issues before they reach dashboards.

## Technical Components

### 1. Schema Context Extraction and Formatting

**Technical Explanation:**

Schema context provides the LLM with structured metadata about source tables, enabling accurate column references and appropriate SQL operations. This requires extracting information from dbt's internal catalog, introspecting databases, or parsing existing schema YAML files.

**Practical Implementation:**

```python
from typing import Dict, List, Optional
import yaml
import json
from dataclasses import dataclass, asdict

@dataclass
class ColumnMetadata:
    name: str
    data_type: str
    description: Optional[str] = None
    tests: List[str] = None
    
    def __post_init__(self):
        if self.tests is None:
            self.tests = []

@dataclass
class TableMetadata:
    name: str
    database: str
    schema: str
    columns: List[ColumnMetadata]
    row_count: Optional[int] = None

def extract_schema_from_dbt_catalog(catalog_path: str) -> Dict[str, TableMetadata]:
    """
    Parse dbt's catalog.json to extract table and column metadata.
    
    Returns structured schema information for context building.
    """
    with open(catalog_path, 'r') as f:
        catalog = json.load(f)
    
    tables = {}
    for node_id, node_data in catalog['sources'].items():
        columns = [
            ColumnMetadata(
                name=col_name,
                data_type=col_data['type'],
                description=col_data.get('comment')
            )
            for col_name, col_data in node_data['columns'].items()
        ]
        
        table = TableMetadata(
            name=node_data['name'],
            database=node_data['database'],
            schema=node_data['schema'],
            columns=columns,
            row_count=node_data['metadata'].get('row_count')
        )
        tables[node_id] = table
    
    return tables

def format_schema_for_llm(tables: Dict[str, TableMetadata]) -> str:
    """
    Format schema metadata for optimal LLM comprehension.
    
    Uses consistent structure that LLMs parse reliably.
    """
    context = []
    for table_id, table in tables.items():
        table_section = f"""
TABLE: {table.schema}.{table.name}
COLUMNS:
"""
        for col in table.columns:
            col_line = f"  - {col.name} ({col.data_type})"
            if col.description:
                col_line += f": {col.description}"
            table_section += col_line + "\n"
        
        if table.row_count:
            table_section += f"APPROXIMATE ROWS: {table.row_count:,}\n"
        
        context.append(table_section)
    
    return "\n".join(context)
```

**Real Constraints and Trade-offs:**

- **Catalog Size:** Large dbt projects have hundreds of models. Including all schemas exceeds context windows. Solution: Filter to relevant upstream models based on dependency graph or use semantic search to retrieve only related tables.

- **Stale Metadata:** Catalog reflects last run. Schema changes between runs cause incorrect generation. Solution: Implement validation step that checks generated SQL against current database schema before deployment.

- **Type Mapping Ambiguity:** Same logical type (e.g., timestamp) has different representations across warehouses (TIMESTAMP vs DATETIME vs TIMESTAMP_NTZ). Solution: Include warehouse-specific type information in context.

### 2. Prompt Engineering for SQL Generation

**Technical Explanation:**

Effective prompts balance specificity (clear requirements) with flexibility (allowing LLM to apply SQL best practices). The prompt structure must guide the model toward dbt-specific patterns while avoiding over-specification that leads to rigid, unoptimized code.

**Practical Implementation:**

```python
from typing import Literal
from textwrap import dedent

def build_model_generation_prompt(
    requirement: str,
    schema_context: str,
    materialization: Literal["table", "view", "incremental", "ephemeral"],
    testing_level: Literal["basic", "comprehensive", "minimal"] = "comprehensive"
) -> str:
    """
    Construct prompt with appropriate constraints and examples.
    """
    
    base_prompt = dedent(f"""
    You are generating production dbt SQL. Follow these requirements:
    
    BUSINESS REQUIREMENT:
    {requirement}
    
    AVAILABLE SOURCES:
    {schema_context}
    
    MATERIALIZATION: {materialization}
    
    REQUIRED SQL PATTERNS:
    - Use ref() for dbt models, source() for raw tables
    - Apply appropriate WHERE filters before JOINs
    - Use explicit column lists (no SELECT *)
    - Cast data types at read time for consistency
    - Use CTEs for logical separation
    - Add comments for complex business logic
    """)
    
    if materialization == "incremental":
        base_prompt += dedent("""
        
        INCREMENTAL REQUIREMENTS:
        - Use is_incremental() macro
        - Define unique_key in config
        - Filter to recent records using lookback window
        - Handle late-arriving data appropriately
        - Example pattern:
          {% if is_incremental() %}
            where event_timestamp > (select max(event_timestamp) from {{ this }})
          {% endif %}
        """)
    
    testing_specs = {
        "basic": "unique and not_null tests for primary keys",
        "comprehensive": "unique, not_null, relationships, and accepted_values tests",
        "minimal": "not_null tests only for critical columns"
    }
    
    base_prompt += dedent(f"""
    
    TESTING REQUIREMENTS: {testing_specs[testing_level]}
    
    OUTPUT FORMAT:
    ```sql
    -- SQL model code here
    ```
    
    ```yaml
    # schema.yml with tests and documentation
    ```
    
    Explain key design decisions in 2-3 sentences.
    """)
    
    return base_prompt
```

**Real Constraints and Trade-offs:**

- **Prompt Length vs. Specificity:** Detailed constraints improve output quality but consume tokens. At 200+ token prompts, you sacrifice context space for source schemas. Solution: Use few-shot examples instead of explicit rules for common patterns.

- **Warehouse-Specific SQL:** Snowflake, BigQuery, Redshift have different functions (DATEDIFF vs DATE_DIFF, LISTAGG vs STRING_AGG). Solution: Include warehouse identifier in prompt and provide dialect-specific function references.

- **Example Pollution:** Including example SQL in prompts can cause copy-paste artifacts. Solution: Use structural examples (CTE organization) rather than complete SQL examples.

### 3. Test Generation Based on Schema Analysis

**Technical Explanation:**

Intelligent test generation analyzes column names, data types, and existing patterns to infer appropriate dbt tests. This goes beyond generic "not_null" tests to implement domain-specific validations based on semantic understanding of data.

**Practical Implementation:**

```python
import re
from typing import List, Dict, Set

class TestInferenceEngine:
    """
    Infer appropriate dbt tests based on column metadata and patterns.
    """
    
    # Pattern-based test rules
    COLUMN_PATTERNS = {
        r'.*_id$': ['unique', 'not_null'],
        r'.*_email$': ['not_null'],  # Add custom email test
        r'.*_date$': ['not_null'],
        r'.*_amount$': ['not_null'],  # Add non-negative test
        r'.*_status$': [],  # Will add accepted_values
        r'.*_percentage$': [],  # Add range test 0-100
    }
    
    TYPE_TESTS = {
        'varchar': [],
        'integer': [],
        'numeric': [],
        'timestamp': ['not_null'],
        'boolean': ['not_null'],
    }
    
    def __init__(self, foreign_key_relationships: Dict[str, str] = None):
        """
        Args:
            foreign_key_relationships: Map of 'column_name' -> 'ref_table.ref_column'
        """
        self.fk_relationships = foreign_key_relationships or {}
    
    def infer_tests(
        self,
        column: ColumnMetadata,
        is_primary_key: bool = False,
        sample_values: List[str] = None
    ) -> List[Dict[str, any]]:
        """
        Generate test definitions for a column.
        
        Returns list of test dictionaries ready for schema.yml
        """
        tests = []
        
        # Primary key tests
        if is_primary_key:
            tests.append({'unique': None})
            tests.append({'not_null': None})
            return tests
        
        # Pattern-based tests
        for pattern, pattern_tests in self.COLUMN_PATTERNS.items():
            if re.match(pattern, column.name, re.IGNORECASE):
                tests.extend([{test: None} for test in pattern_tests])
        
        # Type-based tests
        base_type = column.data_type.lower().split('(')[0]
        if base_type in self.TYPE_TESTS:
            tests.extend([{test: None} for test in self.TYPE_TESTS[base_type]])
        
        # Foreign key relationship tests
        if column.name in self.fk_relationships:
            ref = self.fk_relationships[column.name]
            ref_table, ref_column = ref.split('.')
            tests.append({
                'relationships': {
                    'to': f"ref('{ref_table}')",
                    'field': ref_column
                }
            })
        
        # Infer accepted_values from samples
        if sample_values and '_status' in column.name.lower():
            unique_values = set(sample_values)
            if len(unique_values) <= 10:  # Small cardinality
                tests.append({
                    'accepted_values': {
                        'values': sorted(unique_values)
                    }
                })
        
        # Domain-specific tests
        if '_amount' in column.name.lower():
            tests.append({
                'dbt_utils.expression_is_true': {
                    'expression': f"{column.name} >= 0"
                }
            })
        
        if '_percentage' in column.name.lower():
            tests.append({
                'dbt_utils.expression_is_true': {
                    'expression': f"{column.name} >= 0 AND {column.name} <= 100"
                }
            })
        
        # Deduplicate
        seen = set()
        unique_tests = []
        for test in tests:
            test_key = str(test)
            if test_key not in seen:
                seen.add(test_key)
                unique_tests.append(test)
        
        return unique_tests

def generate_schema_yaml(
    model_name: str,
    columns: List[ColumnMetadata],
    test_engine: TestInferenceEngine,
    primary_key: str = None
) -> str:
    """
    Generate complete schema.yml with inferred tests.
    """
    schema = {
        'version': 2,
        'models': [{
            'name': model_name,
            'description': f"{{ doc('{model_name}') }}",
            'columns': []
        }]
    }
    
    for col in columns:
        col_def = {
            'name': col.name,
            'description': col.description or f"{col.name} column",
        }
        
        # Infer tests
        tests = test_engine.infer_tests(
            col,
            is_primary_key=(col.name == primary_key)
        )
        
        if tests:
            col_def['tests'] = tests
        
        schema['models'][0]['columns'].append(col_def)
    
    return yaml.dump(schema, default_flow_style=False, sort_keys=False)
```

**Real Constraints and Trade-offs:**

- **False Positives:**