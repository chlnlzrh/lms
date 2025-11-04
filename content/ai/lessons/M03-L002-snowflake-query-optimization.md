# Snowflake Query Optimization for AI/ML Workloads

## Core Concepts

Snowflake query optimization is the practice of structuring SQL queries, data layouts, and warehouse configurations to minimize compute costs and latency in cloud data warehouse environments. Unlike traditional databases where optimization focuses on disk I/O and index tuning, Snowflake optimization centers on three key dimensions: **data clustering** (how data is physically organized in micro-partitions), **query compilation** (how SQL is translated to distributed execution plans), and **warehouse sizing** (matching compute resources to workload characteristics).

### Traditional vs. Modern Approach

**Traditional Database Optimization:**
```python
# Traditional approach: Index-centric optimization
# You'd create indexes, analyze execution plans, tune buffer pools

import psycopg2

conn = psycopg2.connect("dbname=analytics")
cursor = conn.cursor()

# Create indexes manually
cursor.execute("""
    CREATE INDEX idx_user_created ON users(created_at);
    CREATE INDEX idx_user_country ON users(country);
""")

# Query relies on index selection by optimizer
cursor.execute("""
    SELECT country, COUNT(*) 
    FROM users 
    WHERE created_at >= '2024-01-01'
    GROUP BY country
""")
# Performance depends on index maintenance, statistics freshness
# Manual VACUUM, ANALYZE required regularly
```

**Modern Snowflake Approach:**
```python
# Snowflake approach: Clustering and pruning-centric
import snowflake.connector

conn = snowflake.connector.connect(
    user='engineer',
    password='***',
    account='xy12345',
    warehouse='COMPUTE_WH',
    database='ANALYTICS'
)

cursor = conn.cursor()

# No manual indexes - use clustering keys for common filters
cursor.execute("""
    ALTER TABLE users CLUSTER BY (created_at, country);
""")

# Same query benefits from automatic micro-partition pruning
cursor.execute("""
    SELECT country, COUNT(*) 
    FROM users 
    WHERE created_at >= '2024-01-01'
    GROUP BY country
""")

# Query profile shows partitions scanned vs. total
# 42 partitions scanned out of 1,200 total (96.5% pruning efficiency)
```

The fundamental shift: **from index management to data organization**. Snowflake automatically maintains micro-partitions (16MB compressed chunks) with metadata about min/max values, distinct counts, and null counts. Your job is to organize data so queries can skip entire partitions based on filter predicates.

### Why This Matters NOW

For AI/ML engineers working with Snowflake, query optimization directly impacts three critical metrics:

1. **Feature engineering pipeline costs**: A 10x optimization in feature extraction queries can mean $50K/month savings in warehouse credits
2. **Model training latency**: Reducing data preparation from 45 minutes to 4 minutes enables rapid experimentation cycles
3. **Inference latency**: Real-time feature lookups need sub-second response times, achievable only with proper optimization

The compression ratio and automatic scaling that make Snowflake attractive for ML workloads also create unique performance traps. A poorly written query might scan billions of rows across thousands of micro-partitions when it only needs thousands of rows from a handful of partitions.

## Technical Components

### 1. Micro-Partition Pruning

Micro-partitions are Snowflake's fundamental storage unit—immutable, compressed files containing 50,000-500,000 rows. Each partition stores metadata including min/max values for all columns. When you issue a query with a WHERE clause, Snowflake's optimizer examines partition metadata to determine which partitions could possibly contain matching rows.

**Technical Implication:** Pruning happens at compile time, before any data is read. A query filtering on `WHERE timestamp >= '2024-01-01'` will only scan partitions whose max timestamp is >= 2024-01-01. If your data is randomly ordered by timestamp, you'll scan most partitions. If timestamp is monotonically increasing during inserts, you'll scan a tiny fraction.

**Real Constraints:**
- Pruning effectiveness degrades as data becomes fragmented through updates/deletes
- String columns prune less effectively than numeric/date columns
- Multi-column filters require proper clustering key order

**Concrete Example:**

```python
import snowflake.connector
import time
from typing import Dict, Any

def analyze_query_pruning(conn: snowflake.connector.SnowflakeConnection, 
                          query: str) -> Dict[str, Any]:
    """
    Execute query and extract micro-partition pruning statistics.
    """
    cursor = conn.cursor()
    
    # Get query ID for profiling
    cursor.execute(query)
    query_id = cursor.sfqid
    
    # Wait for query to complete
    time.sleep(2)
    
    # Extract pruning stats from query profile
    cursor.execute(f"""
        SELECT 
            execution_status,
            total_elapsed_time / 1000.0 as seconds_elapsed,
            bytes_scanned,
            bytes_written,
            rows_produced,
            compilation_time / 1000.0 as compile_seconds,
            execution_time / 1000.0 as execute_seconds,
            partitions_scanned,
            partitions_total
        FROM TABLE(INFORMATION_SCHEMA.QUERY_HISTORY_BY_SESSION())
        WHERE query_id = '{query_id}'
    """)
    
    result = cursor.fetchone()
    
    if result and result[7] is not None:
        pruning_efficiency = (1 - result[7] / result[8]) * 100 if result[8] > 0 else 0
        
        return {
            'query_id': query_id,
            'elapsed_seconds': result[1],
            'bytes_scanned_gb': result[2] / (1024**3),
            'rows_produced': result[4],
            'partitions_scanned': result[7],
            'partitions_total': result[8],
            'pruning_efficiency_pct': pruning_efficiency
        }
    
    return {}

# Compare pruning with different query patterns
conn = snowflake.connector.connect(
    user='engineer',
    account='xy12345',
    warehouse='COMPUTE_WH'
)

# Bad: Random UUID filter - no pruning possible
bad_query = """
    SELECT * FROM events 
    WHERE user_id = 'abc-123-def'  -- UUID has no natural ordering
    LIMIT 1000
"""

stats_bad = analyze_query_pruning(conn, bad_query)
print(f"Bad query: {stats_bad['partitions_scanned']}/{stats_bad['partitions_total']} "
      f"partitions ({stats_bad['pruning_efficiency_pct']:.1f}% pruned)")

# Good: Time-range filter on clustered column
good_query = """
    SELECT * FROM events 
    WHERE event_timestamp >= '2024-01-15'
      AND event_timestamp < '2024-01-16'
    LIMIT 1000
"""

stats_good = analyze_query_pruning(conn, good_query)
print(f"Good query: {stats_good['partitions_scanned']}/{stats_good['partitions_total']} "
      f"partitions ({stats_good['pruning_efficiency_pct']:.1f}% pruned)")

# Output:
# Bad query: 1847/1850 partitions (0.2% pruned)
# Good query: 12/1850 partitions (99.4% pruned)
```

### 2. Clustering Keys and Data Layout

Clustering keys tell Snowflake to physically co-locate rows with similar values. Unlike indexes, clustering is not enforced—it's a suggestion that affects how data is organized during DML operations. Snowflake automatically maintains clustering through background re-clustering, but this has cost implications.

**Technical Explanation:** When you define `CLUSTER BY (col1, col2)`, Snowflake sorts data within micro-partitions by col1, then col2. This creates natural ordering that enables pruning. The clustering depth metric (0-100+) indicates how well-maintained clustering is—lower is better, with 0-4 considered well-clustered.

**Practical Implications:**
- First column in clustering key has highest impact on pruning
- Clustering maintenance consumes warehouse credits automatically
- High-cardinality columns (millions of distinct values) cluster poorly

**Trade-offs:**
- More clustering keys = better query pruning but higher maintenance costs
- Frequent updates/inserts degrade clustering over time
- Over-clustering can actually increase costs without query benefits

```python
from typing import List, Tuple
import snowflake.connector

def evaluate_clustering_candidates(
    conn: snowflake.connector.SnowflakeConnection,
    table: str,
    candidate_columns: List[str]
) -> List[Tuple[str, float, int]]:
    """
    Analyze which columns would make effective clustering keys
    by examining cardinality and common query patterns.
    """
    cursor = conn.cursor()
    results = []
    
    for column in candidate_columns:
        # Check cardinality
        cursor.execute(f"""
            SELECT 
                COUNT(DISTINCT {column}) as distinct_vals,
                COUNT(*) as total_rows,
                COUNT(DISTINCT {column}) * 100.0 / COUNT(*) as cardinality_pct
            FROM {table}
        """)
        
        distinct_vals, total_rows, cardinality_pct = cursor.fetchone()
        
        # Lower cardinality % = better clustering candidate
        # Ideal: 0.01% - 10% (sweet spot for pruning)
        score = 100 - min(cardinality_pct, 100)
        
        results.append((column, score, distinct_vals))
    
    # Sort by score descending
    results.sort(key=lambda x: x[1], reverse=True)
    
    return results

# Example usage
conn = snowflake.connector.connect(
    user='engineer',
    account='xy12345',
    warehouse='COMPUTE_WH',
    database='ML_FEATURES'
)

candidates = ['event_timestamp', 'user_id', 'country_code', 'event_type']
scores = evaluate_clustering_candidates(conn, 'user_events', candidates)

print("Clustering Key Recommendations:")
print(f"{'Column':<20} {'Score':<10} {'Distinct Values':<20}")
print("-" * 50)
for col, score, distinct in scores:
    print(f"{col:<20} {score:<10.1f} {distinct:<20,}")

# Output:
# Column               Score      Distinct Values     
# --------------------------------------------------
# country_code         98.5       195                 
# event_type           96.2       42                  
# event_timestamp      45.8       8,472,103           
# user_id              2.1        124,583,992
```

### 3. Query Result Caching

Snowflake caches query results for 24 hours. If you execute the exact same query (byte-for-byte identical SQL), the result is returned instantly from cache with zero compute cost. This is distinct from the data cache (also called warehouse cache) which caches raw micro-partitions in SSD storage attached to warehouse nodes.

**Technical Explanation:** Result cache is global across all warehouses and users. Cache keys include the SQL text, query context (database, schema, role), and table versions. Any DML on source tables invalidates affected cache entries.

**Practical Implications for ML Workloads:**
- Feature extraction queries running repeatedly can be free after first execution
- Parameterized queries (different WHERE values) don't hit result cache
- Large result sets (>10GB) are not cached

**Common Failure Mode:**
Engineers write dynamic SQL with timestamps or GUIDs that change on every execution, preventing cache hits:

```python
import snowflake.connector
from datetime import datetime

conn = snowflake.connector.connect(
    user='engineer',
    account='xy12345',
    warehouse='COMPUTE_WH'
)

cursor = conn.cursor()

# BAD: Query includes current timestamp - never hits cache
bad_query = f"""
    SELECT 
        user_id,
        -- This comment includes run time: {datetime.now()}
        COUNT(*) as event_count
    FROM events
    WHERE event_date = '2024-01-15'
    GROUP BY user_id
"""

cursor.execute(bad_query)
# Cost: Full warehouse execution every time

# GOOD: Static query - hits cache on subsequent runs
good_query = """
    SELECT 
        user_id,
        COUNT(*) as event_count
    FROM events
    WHERE event_date = '2024-01-15'
    GROUP BY user_id
"""

cursor.execute(good_query)
# Cost: Full execution first time, zero cost for 24 hours after

# Check if query used cached results
cursor.execute(f"""
    SELECT 
        query_id,
        query_text,
        execution_status,
        total_elapsed_time,
        bytes_scanned,
        query_load_percent  -- 0 means cache hit
    FROM TABLE(INFORMATION_SCHEMA.QUERY_HISTORY_BY_SESSION())
    ORDER BY start_time DESC
    LIMIT 5
""")

for row in cursor.fetchall():
    cache_hit = "CACHE HIT" if row[5] == 0 else "CACHE MISS"
    print(f"{cache_hit}: {row[1][:50]}... ({row[3]}ms)")
```

### 4. Warehouse Sizing and Scaling

Warehouses come in T-shirt sizes (XS to 6XL), each doubling compute resources and cost. Choosing the right size involves understanding your query characteristics: CPU-bound (aggregations, joins) vs. I/O-bound (large scans), and concurrency requirements.

**Technical Explanation:** 
- XS = 1 cluster with 1 server (8 virtual cores)
- S = 1 cluster with 2 servers (16 cores)
- Each size up doubles both compute power and cost per second
- Multi-cluster warehouses (1-10 clusters) auto-scale based on query queue depth

**Real Constraints:**
- Smaller warehouses have less data cache (SSD attached to nodes)
- Very large queries may be slower on small warehouses due to memory constraints
- Scaling up provides more resources to a single query; scaling out provides more concurrency

```python
import snowflake.connector
from typing import Dict, List
import statistics

def benchmark_warehouse_sizes(
    conn: snowflake.connector.SnowflakeConnection,
    query: str,
    sizes: List[str] = ['XSMALL', 'SMALL', 'MEDIUM', 'LARGE']
) -> Dict[str, Dict[str, float]]:
    """
    Run same query on different warehouse sizes to find optimal cost/performance.
    Returns execution time and credit consumption for each size.
    """
    cursor = conn.cursor()
    results = {}
    
    for size in sizes:
        warehouse_name = f"BENCH_{size}"
        
        # Create/alter warehouse to specific size
        cursor.execute(f"""
            CREATE WAREHOUSE IF NOT EXISTS {warehouse_name}
            WITH WAREHOUSE_SIZE = '{size}'
            AUTO_SUSPEND = 60
            AUTO_RESUME = TRUE
        """)
        
        cursor.execute(f"USE WAREHOUSE {warehouse_name}")
        
        # Run query 3 times to account for cache warmup
        execution_times = []
        for run in range(3):
            # Clear result cache by adding unique comment
            unique_query = f"-- Run {run}\n{query}"
            
            start = time.time()
            cursor.execute(unique_query)
            cursor.fetchall()  # Ensure full execution
            elapsed = time.time() - start
            execution_times.append(elapsed)
        
        # Use median time (eliminates outliers)
        median_time = statistics.median(execution_times)
        
        # Get credit consumption (cost)
        # Credits per hour by size: XS=1, S=2, M=4, L=8
        credits_per_hour = 2 ** (sizes.index(size))
        credits_consumed = (median_time / 3600) * credits_per_hour
        
        results[size] = {
            'median_seconds': median_time,
            'credits_consumed': credits_consumed,
            'cost_efficiency': credits_consumed / median_time  # Lower is better
        }
        
        # Cleanup
        cursor.execute(f"DROP WAREHOUSE {warehouse_name}")
    
    return results

# Example: Find optimal warehouse for feature aggregation query
conn = snowflake.connector.connect(
    user='engineer',