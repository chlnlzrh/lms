# Natural Language SQL Agents: Engineering Database Interfaces with LLMs

## Core Concepts

Natural Language SQL Agents are systems that translate human language queries into executable SQL statements, execute them against databases, and return results in natural language. Unlike traditional database interfaces requiring SQL expertise, these agents enable natural language interaction with structured data while maintaining the precision and performance of SQL.

### Traditional vs. Agent-Based Architecture

**Traditional approach:**

```python
from typing import List, Dict, Any
import sqlite3

class TraditionalDatabaseInterface:
    """Manual SQL construction requires complete schema knowledge"""
    
    def __init__(self, db_path: str):
        self.conn = sqlite3.connect(db_path)
    
    def get_top_customers(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Hardcoded query - one function per business question"""
        query = """
            SELECT c.customer_id, c.name, SUM(o.total) as revenue
            FROM customers c
            JOIN orders o ON c.customer_id = o.customer_id
            WHERE o.order_date >= date('now', '-1 year')
            GROUP BY c.customer_id, c.name
            ORDER BY revenue DESC
            LIMIT ?
        """
        cursor = self.conn.execute(query, (limit,))
        return [dict(zip([d[0] for d in cursor.description], row)) 
                for row in cursor.fetchall()]
    
    def get_product_performance(self, category: str) -> List[Dict[str, Any]]:
        """Another hardcoded query for different question"""
        query = """
            SELECT p.product_name, COUNT(oi.order_id) as order_count,
                   SUM(oi.quantity * oi.price) as revenue
            FROM products p
            JOIN order_items oi ON p.product_id = oi.product_id
            WHERE p.category = ?
            GROUP BY p.product_id, p.product_name
            ORDER BY revenue DESC
        """
        cursor = self.conn.execute(query, (category,))
        return [dict(zip([d[0] for d in cursor.description], row)) 
                for row in cursor.fetchall()]
```

**Agent-based approach:**

```python
from typing import List, Dict, Any, Optional
import sqlite3
import json
from anthropic import Anthropic

class NaturalLanguageSQLAgent:
    """Single interface handles arbitrary business questions"""
    
    def __init__(self, db_path: str, api_key: str):
        self.conn = sqlite3.connect(db_path)
        self.client = Anthropic(api_key=api_key)
        self.schema = self._extract_schema()
    
    def _extract_schema(self) -> str:
        """Generate schema documentation from database"""
        cursor = self.conn.execute("""
            SELECT sql FROM sqlite_master 
            WHERE type='table' AND name NOT LIKE 'sqlite_%'
        """)
        tables = [row[0] for row in cursor.fetchall()]
        return "\n\n".join(tables)
    
    def query(self, natural_language_query: str) -> Dict[str, Any]:
        """Handle any business question without hardcoding"""
        
        # Generate SQL from natural language
        sql_query = self._generate_sql(natural_language_query)
        
        # Execute with safety checks
        results = self._execute_safely(sql_query)
        
        # Format results naturally
        natural_response = self._format_response(
            natural_language_query, sql_query, results
        )
        
        return {
            "query": natural_language_query,
            "sql": sql_query,
            "results": results,
            "explanation": natural_response
        }
    
    def _generate_sql(self, question: str) -> str:
        """Use LLM to generate SQL from natural language"""
        prompt = f"""You are a SQL expert. Generate a SQL query for this question.

Database Schema:
{self.schema}

Question: {question}

Return ONLY the SQL query, no explanation. Ensure the query is safe (read-only)."""

        response = self.client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}]
        )
        
        sql = response.content[0].text.strip()
        # Remove markdown code blocks if present
        sql = sql.replace("```sql", "").replace("```", "").strip()
        return sql
    
    def _execute_safely(self, sql: str) -> List[Dict[str, Any]]:
        """Execute SQL with safety constraints"""
        # Validate read-only
        forbidden_keywords = ['INSERT', 'UPDATE', 'DELETE', 'DROP', 'ALTER', 'CREATE']
        if any(keyword in sql.upper() for keyword in forbidden_keywords):
            raise ValueError(f"Query contains forbidden operations: {sql}")
        
        cursor = self.conn.execute(sql)
        columns = [d[0] for d in cursor.description]
        return [dict(zip(columns, row)) for row in cursor.fetchall()]
    
    def _format_response(self, question: str, sql: str, 
                        results: List[Dict[str, Any]]) -> str:
        """Convert structured results to natural language"""
        prompt = f"""Provide a clear answer to the user's question based on the data.

Question: {question}
SQL Query: {sql}
Results: {json.dumps(results[:5], indent=2)}
Total rows: {len(results)}

Provide a concise, natural language answer."""

        response = self.client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=512,
            messages=[{"role": "user", "content": prompt}]
        )
        
        return response.content[0].text.strip()


# Usage comparison
if __name__ == "__main__":
    # Traditional: Need new function for every question type
    trad = TraditionalDatabaseInterface("sales.db")
    customers = trad.get_top_customers(10)  # Only works for this specific query
    
    # Agent: Single interface for all questions
    agent = NaturalLanguageSQLAgent("sales.db", "your-api-key")
    
    # Same question as traditional
    result1 = agent.query("Who are our top 10 customers by revenue in the last year?")
    
    # New question - no code changes needed
    result2 = agent.query("What percentage of orders were returned in Q4?")
    
    # Complex ad-hoc analysis
    result3 = agent.query(
        "Compare average order value by customer segment, "
        "excluding orders with discounts over 20%"
    )
```

### Why Natural Language SQL Agents Matter Now

**1. Dynamic Business Intelligence:** Traditional BI dashboards require weeks of development for each new metric. SQL agents enable analysts to ask questions directly without engineering bottlenecks.

**2. Democratized Data Access:** Non-technical stakeholders can query databases without learning SQL syntax or waiting for data team availability. This reduces query backlog by 60-80% in production deployments.

**3. Adaptive Schema Handling:** As database schemas evolve, agents adapt automatically. Traditional hardcoded queries break with schema changes, requiring manual updates across potentially hundreds of functions.

**4. Cost-Effective Scaling:** Instead of maintaining extensive custom API layers, agents provide a single interface that handles arbitrary queries, reducing codebase complexity by orders of magnitude.

### Key Engineering Insight

The power shift is from **compile-time query definition** (writing functions for every possible question) to **runtime query generation** (letting LLMs construct queries on-demand). This mirrors the broader shift from imperative to declarative programming: you specify **what** you want, not **how** to get it.

## Technical Components

### 1. Schema Representation and Context Management

The schema representation determines what the LLM "knows" about your database structure. Poor schema representation is the #1 cause of incorrect SQL generation.

**Technical Implementation:**

```python
from typing import Dict, List, Optional
from dataclasses import dataclass
import sqlite3

@dataclass
class ColumnInfo:
    name: str
    type: str
    nullable: bool
    description: Optional[str] = None

@dataclass
class TableInfo:
    name: str
    columns: List[ColumnInfo]
    description: Optional[str] = None
    sample_values: Optional[Dict[str, List[Any]]] = None

class SchemaManager:
    """Advanced schema extraction with semantic annotations"""
    
    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn
        self.annotations = self._load_annotations()
    
    def _load_annotations(self) -> Dict[str, Dict]:
        """Load human-written descriptions for ambiguous schema elements"""
        # In production, load from config file or metadata table
        return {
            "customers.segment": {
                "description": "Customer tier: 'enterprise', 'mid-market', 'smb'",
                "samples": ["enterprise", "mid-market", "smb"]
            },
            "orders.status": {
                "description": "Order state: 'pending', 'shipped', 'delivered', 'cancelled'",
                "samples": ["pending", "shipped", "delivered", "cancelled"]
            }
        }
    
    def get_enriched_schema(self, include_samples: bool = True) -> str:
        """Generate LLM-optimized schema representation"""
        tables = self._get_all_tables()
        schema_parts = []
        
        for table in tables:
            table_info = self._get_table_info(table, include_samples)
            schema_parts.append(self._format_table_for_llm(table_info))
        
        return "\n\n".join(schema_parts)
    
    def _get_all_tables(self) -> List[str]:
        cursor = self.conn.execute("""
            SELECT name FROM sqlite_master 
            WHERE type='table' AND name NOT LIKE 'sqlite_%'
            ORDER BY name
        """)
        return [row[0] for row in cursor.fetchall()]
    
    def _get_table_info(self, table: str, include_samples: bool) -> TableInfo:
        """Extract comprehensive table metadata"""
        # Get column information
        cursor = self.conn.execute(f"PRAGMA table_info({table})")
        columns = []
        
        for row in cursor.fetchall():
            col_name = row[1]
            col_type = row[2]
            nullable = not row[3]
            
            # Check for annotation
            annotation_key = f"{table}.{col_name}"
            description = self.annotations.get(annotation_key, {}).get("description")
            
            columns.append(ColumnInfo(col_name, col_type, nullable, description))
        
        # Get sample values for better context
        sample_values = None
        if include_samples:
            sample_values = self._get_sample_values(table, columns)
        
        return TableInfo(table, columns, sample_values=sample_values)
    
    def _get_sample_values(self, table: str, 
                           columns: List[ColumnInfo]) -> Dict[str, List[Any]]:
        """Extract representative sample values"""
        samples = {}
        
        for col in columns[:5]:  # Limit to first 5 columns to control context size
            try:
                cursor = self.conn.execute(f"""
                    SELECT DISTINCT {col.name} 
                    FROM {table} 
                    WHERE {col.name} IS NOT NULL 
                    LIMIT 3
                """)
                samples[col.name] = [row[0] for row in cursor.fetchall()]
            except sqlite3.Error:
                continue
        
        return samples
    
    def _format_table_for_llm(self, table_info: TableInfo) -> str:
        """Format table info for optimal LLM comprehension"""
        lines = [f"Table: {table_info.name}"]
        
        if table_info.description:
            lines.append(f"Description: {table_info.description}")
        
        lines.append("Columns:")
        for col in table_info.columns:
            col_line = f"  - {col.name} ({col.type})"
            if not col.nullable:
                col_line += " NOT NULL"
            if col.description:
                col_line += f" -- {col.description}"
            
            # Add sample values if available
            if table_info.sample_values and col.name in table_info.sample_values:
                samples = table_info.sample_values[col.name]
                col_line += f" [examples: {', '.join(map(str, samples))}]"
            
            lines.append(col_line)
        
        return "\n".join(lines)
```

**Practical Implications:**

- **Context Window Management:** Full schema + samples can consume 5-10K tokens for medium databases. For large schemas, implement table filtering based on query intent.
- **Sample Values Critical:** Including 2-3 sample values per column reduces SQL generation errors by ~40% for columns with enum-like values.
- **Annotation Investment:** Spending 30 minutes adding descriptions to ambiguous columns (status codes, abbreviations) prevents days of debugging incorrect queries.

### 2. SQL Generation with Validation Pipeline

Raw LLM output requires validation before execution. A robust validation pipeline prevents errors and security issues.

**Technical Implementation:**

```python
from typing import Optional, List, Tuple
import re
import sqlparse
from anthropic import Anthropic

class SQLGenerator:
    """Generate and validate SQL with multi-stage checking"""
    
    def __init__(self, client: Anthropic, schema: str):
        self.client = client
        self.schema = schema
    
    def generate_validated_sql(self, question: str) -> Tuple[str, List[str]]:
        """Generate SQL with validation, return (sql, warnings)"""
        
        # Stage 1: Generate SQL
        raw_sql = self._generate_sql(question)
        
        # Stage 2: Syntax validation
        if not self._validate_syntax(raw_sql):
            # Attempt self-correction
            raw_sql = self._self_correct(question, raw_sql)
        
        # Stage 3: Safety validation
        warnings = self._validate_safety(raw_sql)
        
        # Stage 4: Semantic validation
        semantic_warnings = self._validate_semantics(raw_sql, question)
        warnings.extend(semantic_warnings)
        
        return raw_sql, warnings
    
    def _generate_sql(self, question: str) -> str:
        """Initial SQL generation with structured prompt"""
        
        system_prompt = """You are an expert SQL generator. Follow these rules:
1. Generate ONLY SELECT queries (read-only)
2. Use explicit JOIN conditions, never implicit joins
3. Always use table aliases for multi-table queries
4. Use LIMIT clauses for potentially large result sets
5. Return ONLY the SQL query, no explanation"""

        user_prompt = f"""Database Schema:
{self.schema}

Generate SQL for: {question}

SQL Query:"""

        response = self.client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=1024,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}]
        )
        
        sql = response.content[0].text.strip()
        # Clean formatting
        sql = sql.replace("```sql", "").replace("```", "").strip()
        return sql
    
    def _validate_syntax(self, sql: str) -> bool:
        """Check SQL syntax validity"""
        try:
            parsed = sqlparse.parse(sql)
            if not parsed:
                return False
            
            # Check it's a single statement
            if len(parsed) != 1:
                return False
            
            # Verify it's a SELECT statement
            stmt = parsed[0]
            if stmt.get_type() != 'SELECT':
                return False
            
            return True
        except Exception:
            return False
    
    def _validate_safety(self, sql: str) -> List[str]:
        """Check for dangerous patterns"""
        warnings = []
        sql_upper = sql.upper()
        
        # Check for write operations
        forbidden = ['INSERT', 'UPDATE', 'DELETE', 'DROP', 'ALTER', 'CREATE', 
                    'TRUNCATE', 'REPLACE', '