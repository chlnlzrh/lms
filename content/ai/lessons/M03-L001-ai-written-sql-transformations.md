# AI-Written SQL Transformations

## Core Concepts

### What Are AI-Written SQL Transformations?

AI-written SQL transformations use large language models to generate, modify, or translate SQL queries based on natural language descriptions, existing code patterns, or semi-structured requirements. Unlike traditional code generation tools that rely on templates or rule-based systems, LLM-based approaches understand context, infer intent, and can handle ambiguous requirements while producing syntactically correct and semantically meaningful SQL.

This isn't about replacing SQL expertise—it's about accelerating the translation between business logic and database operations, especially for repetitive transformations, dialect conversions, and exploratory data work.

### Traditional vs. Modern Approach

**Traditional SQL Generation:**

```python
# Template-based approach with rigid structure
def generate_report_query(table: str, columns: list[str], 
                         filters: dict[str, str]) -> str:
    """Rule-based SQL generation - brittle and limited"""
    select_clause = ", ".join(columns)
    where_clauses = [f"{k} = '{v}'" for k, v in filters.items()]
    where_clause = " AND ".join(where_clauses) if where_clauses else "1=1"
    
    return f"SELECT {select_clause} FROM {table} WHERE {where_clause}"

# Usage - must fit exact template structure
query = generate_report_query(
    table="users",
    columns=["user_id", "email"],
    filters={"status": "active"}
)
# Output: SELECT user_id, email FROM users WHERE status = 'active'

# Limitations:
# - Can't handle JOIN logic
# - No aggregate functions
# - No subqueries or CTEs
# - Rigid input structure
```

**LLM-Based Approach:**

```python
from typing import Optional
import anthropic
import os

def generate_sql_with_llm(
    requirement: str,
    schema_context: str,
    dialect: str = "postgresql"
) -> tuple[str, str]:
    """
    Generate SQL using LLM with schema awareness and dialect support
    Returns: (sql_query, explanation)
    """
    client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
    
    prompt = f"""Generate a {dialect} SQL query for this requirement:

{requirement}

Database schema context:
{schema_context}

Return:
1. The SQL query (optimized and following best practices)
2. Brief explanation of key decisions

Format as:
SQL:
<your query>

EXPLANATION:
<your explanation>
"""
    
    message = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=1500,
        messages=[{"role": "user", "content": prompt}]
    )
    
    response = message.content[0].text
    
    # Parse response
    parts = response.split("EXPLANATION:")
    sql = parts[0].replace("SQL:", "").strip()
    explanation = parts[1].strip() if len(parts) > 1 else ""
    
    return sql, explanation

# Usage - flexible natural language
schema = """
users (user_id, email, status, created_at, country)
orders (order_id, user_id, amount, order_date)
"""

requirement = """
Get the top 10 countries by total order value in 2024, 
but only include active users. Show country, total revenue, 
and user count. Sort by revenue descending.
"""

sql, explanation = generate_sql_with_llm(requirement, schema)

# Output (example):
# SQL:
# SELECT 
#     u.country,
#     SUM(o.amount) as total_revenue,
#     COUNT(DISTINCT u.user_id) as user_count
# FROM users u
# INNER JOIN orders o ON u.user_id = o.user_id
# WHERE u.status = 'active'
#     AND EXTRACT(YEAR FROM o.order_date) = 2024
# GROUP BY u.country
# ORDER BY total_revenue DESC
# LIMIT 10;
#
# EXPLANATION:
# - Uses INNER JOIN to ensure only users with orders
# - Filters on status before aggregation for performance
# - EXTRACT for year filtering (dialect-specific)
# - COUNT(DISTINCT) ensures accurate user count with multiple orders
```

The LLM approach handles complex requirements, understands relationships, applies best practices, and adapts to different SQL dialects—all from natural language descriptions.

### Key Engineering Insights

**1. Context is the Constraint**

Unlike traditional code generation where schema is embedded in code, LLM-based SQL generation is only as good as the schema context you provide. The model doesn't have access to your database—it relies entirely on the information in your prompt. This means schema documentation becomes a first-class engineering artifact.

**2. Verification is Non-Negotiable**

LLMs can generate syntactically valid SQL that's semantically wrong (joins on incorrect columns, wrong aggregation logic, missing filters). Every generated query needs validation—either through testing against known results or expert review. Treat LLM output as a first draft, not production code.

**3. Patterns Emerge Through Examples**

LLMs excel at pattern recognition. Providing 1-2 example queries in your organization's style dramatically improves output quality. This is especially valuable for enforcing conventions (naming, formatting, optimization patterns) without building custom tooling.

### Why This Matters Now

**Dialect Fragmentation**: Organizations increasingly use multiple database systems (Postgres for OLTP, BigQuery for analytics, Snowflake for data warehousing). Manual dialect translation is error-prone and time-consuming. LLMs handle dialect-specific syntax naturally.

**Data Team Bottlenecks**: Analytics teams spend 40-60% of time writing variations of similar SQL queries. AI-assisted SQL generation allows junior engineers to handle routine transformations while senior engineers focus on complex optimization and architecture.

**Legacy Migration Acceleration**: Companies have millions of lines of legacy SQL (Oracle, SQL Server) that need translation to modern platforms. LLM-based transformation can accelerate migration projects from years to months when combined with proper validation.

## Technical Components

### 1. Prompt Structure for SQL Generation

The quality of generated SQL depends heavily on how you structure the generation request. A well-formed prompt includes schema context, requirements clarity, and output constraints.

**Technical Explanation:**

Effective SQL generation prompts follow a consistent structure:
- **Schema Context**: Table definitions, relationships, and constraints
- **Requirement Specification**: Clear, unambiguous description of desired output
- **Constraints**: Performance requirements, dialect, conventions
- **Examples (Optional)**: Sample queries demonstrating style/patterns

**Practical Implementation:**

```python
from dataclasses import dataclass
from typing import Optional

@dataclass
class SQLGenerationRequest:
    """Structured request for SQL generation"""
    requirement: str
    schema_context: str
    dialect: str = "postgresql"
    performance_constraints: Optional[str] = None
    example_patterns: Optional[list[str]] = None
    output_format: str = "formatted"  # 'formatted' or 'compact'

def build_sql_prompt(request: SQLGenerationRequest) -> str:
    """
    Construct optimized prompt for SQL generation
    """
    prompt_parts = [
        f"Generate {request.dialect} SQL for this requirement:",
        f"\nREQUIREMENT:\n{request.requirement}",
        f"\nSCHEMA:\n{request.schema_context}"
    ]
    
    if request.performance_constraints:
        prompt_parts.append(
            f"\nPERFORMANCE REQUIREMENTS:\n{request.performance_constraints}"
        )
    
    if request.example_patterns:
        examples = "\n\n".join(request.example_patterns)
        prompt_parts.append(
            f"\nFOLLOW THESE STYLE PATTERNS:\n{examples}"
        )
    
    prompt_parts.extend([
        "\nGENERATE:",
        "1. SQL query (optimized, production-ready)",
        "2. Index recommendations if applicable",
        "3. Any assumptions made"
    ])
    
    return "\n".join(prompt_parts)

# Usage example
request = SQLGenerationRequest(
    requirement="Find users who made their first purchase in the last 30 days",
    schema_context="""
    users (id, email, created_at)
    orders (id, user_id, created_at, amount)
    """,
    dialect="postgresql",
    performance_constraints="Table has 10M+ orders, must use indexes efficiently",
    example_patterns=[
        "-- Company style: use CTEs for readability\nWITH recent AS (...) SELECT ..."
    ]
)

prompt = build_sql_prompt(request)
```

**Real Constraints:**

- **Token Limits**: Large schemas (100+ tables) may exceed context windows. Solution: filter to relevant tables only.
- **Ambiguity Handling**: LLMs will make assumptions when requirements are unclear. Always request that assumptions be stated explicitly.
- **Consistency**: Same prompt may yield different queries across runs. Use temperature=0 for reproducibility in production.

### 2. Schema Context Engineering

Schema context is the most critical input for SQL generation quality. How you represent schema information directly impacts correctness and optimization.

**Technical Explanation:**

LLMs need to understand:
- Table structure (columns, types, constraints)
- Relationships (foreign keys, cardinality)
- Data characteristics (nullable, indexed, typical values)
- Business semantics (what the data represents)

**Practical Implementation:**

```python
from typing import NamedTuple

class ColumnInfo(NamedTuple):
    name: str
    type: str
    nullable: bool
    indexed: bool
    description: str = ""

class TableSchema(NamedTuple):
    name: str
    columns: list[ColumnInfo]
    primary_key: list[str]
    foreign_keys: dict[str, str]  # column -> referenced_table.column
    row_count_estimate: Optional[int] = None

def format_schema_context(schemas: list[TableSchema], 
                          include_stats: bool = True) -> str:
    """
    Format schema information for LLM consumption
    """
    context_parts = []
    
    for schema in schemas:
        # Table header with statistics
        table_info = f"Table: {schema.name}"
        if include_stats and schema.row_count_estimate:
            table_info += f" (~{schema.row_count_estimate:,} rows)"
        context_parts.append(table_info)
        
        # Columns with rich metadata
        for col in schema.columns:
            col_def = f"  - {col.name}: {col.type}"
            
            attributes = []
            if not col.nullable:
                attributes.append("NOT NULL")
            if col.indexed:
                attributes.append("INDEXED")
            if col.name in schema.primary_key:
                attributes.append("PRIMARY KEY")
            
            if attributes:
                col_def += f" [{', '.join(attributes)}]"
            
            if col.description:
                col_def += f"\n    // {col.description}"
            
            context_parts.append(col_def)
        
        # Relationships
        if schema.foreign_keys:
            context_parts.append("  Relationships:")
            for fk_col, ref in schema.foreign_keys.items():
                context_parts.append(f"    - {fk_col} -> {ref}")
        
        context_parts.append("")  # Blank line between tables
    
    return "\n".join(context_parts)

# Example usage
users_schema = TableSchema(
    name="users",
    columns=[
        ColumnInfo("id", "INTEGER", False, True, "Unique user identifier"),
        ColumnInfo("email", "VARCHAR(255)", False, True, "User email (unique)"),
        ColumnInfo("created_at", "TIMESTAMP", False, True, "Account creation time"),
        ColumnInfo("country_code", "CHAR(2)", True, False, "ISO country code"),
    ],
    primary_key=["id"],
    foreign_keys={},
    row_count_estimate=5_000_000
)

orders_schema = TableSchema(
    name="orders",
    columns=[
        ColumnInfo("id", "INTEGER", False, True),
        ColumnInfo("user_id", "INTEGER", False, True, "FK to users.id"),
        ColumnInfo("created_at", "TIMESTAMP", False, True),
        ColumnInfo("amount", "DECIMAL(10,2)", False, False),
        ColumnInfo("status", "VARCHAR(20)", False, True),
    ],
    primary_key=["id"],
    foreign_keys={"user_id": "users.id"},
    row_count_estimate=15_000_000
)

context = format_schema_context([users_schema, orders_schema])
print(context)
```

**Real Constraints:**

- **Selective Context**: For databases with 100+ tables, provide only relevant tables. Use dependency analysis to include related tables automatically.
- **Metadata Quality**: Poor descriptions lead to incorrect assumptions. Example: A column named `status` without description might be used incorrectly.
- **Scale Awareness**: Including row counts helps LLMs suggest appropriate optimization strategies (indexes, partitioning).

### 3. Query Validation and Testing

Generated SQL must be validated before execution, especially in production contexts. Validation ranges from syntax checking to semantic correctness verification.

**Technical Explanation:**

Multi-layer validation approach:
1. **Syntax Validation**: Parse SQL to ensure it's valid for target dialect
2. **Semantic Validation**: Verify columns exist, joins are logical
3. **Performance Validation**: Check for missing indexes, full table scans
4. **Result Validation**: Compare against known results or expectations

**Practical Implementation:**

```python
import sqlparse
from sqlparse.sql import IdentifierList, Identifier, Where
from sqlparse.tokens import Keyword, DML
from typing import Set, List
import re

class SQLValidator:
    """Validate generated SQL queries"""
    
    def __init__(self, schema_map: dict[str, TableSchema]):
        self.schema_map = schema_map
    
    def validate_syntax(self, sql: str) -> tuple[bool, str]:
        """Check if SQL is syntactically valid"""
        try:
            parsed = sqlparse.parse(sql)
            if not parsed:
                return False, "Empty or invalid SQL"
            
            # Check for common syntax issues
            formatted = sqlparse.format(sql, reindent=True)
            if "ERROR" in formatted.upper():
                return False, "Syntax error detected"
            
            return True, "Syntax valid"
        except Exception as e:
            return False, f"Parse error: {str(e)}"
    
    def extract_table_references(self, sql: str) -> Set[str]:
        """Extract all table names referenced in query"""
        parsed = sqlparse.parse(sql)[0]
        tables = set()
        
        from_seen = False
        for token in parsed.tokens:
            if from_seen and isinstance(token, IdentifierList):
                for identifier in token.get_identifiers():
                    tables.add(str(identifier.get_real_name()).lower())
            elif from_seen and isinstance(token, Identifier):
                tables.add(str(token.get_real_name()).lower())
            
            if token.ttype is Keyword and token.value.upper() == 'FROM':
                from_seen = True
        
        return tables
    
    def validate_schema_compliance(self, sql: str) -> tuple[bool, List[str]]:
        """Check if query references valid tables and columns"""
        issues = []
        
        # Extract referenced tables
        tables = self.extract_table_references(sql)
        
        # Check all tables exist
        for table in tables:
            if table not in self.schema_map:
                issues.append(f"Unknown table: {table}")
        
        # Extract column references (simplified - production would be more robust)
        sql_upper = sql.upper()
        select_match = re.search(r'SELECT\s+(.*?)\s+FROM', sql_upper, re.DOTALL)
        if select_match:
            select_clause = select_match.group(1)
            # Basic column extraction (would need enhancement for aliases, functions)
            columns = [c.strip() for c in select_clause.split(',')]
            
            # Validate columns exist in referenced tables
            for col in columns:
                if '.' in col:  # Qualified column reference
                    table, column = col.split('.', 1)
                    table = table.lower().strip()
                    column = column.lower().strip()