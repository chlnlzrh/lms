# Codebase Awareness: Engineering Context-Aware AI Systems

## Core Concepts

**Technical Definition:** Codebase awareness is the capability of an AI system to maintain and leverage structured knowledge about a software project's architecture, dependencies, conventions, and implementation patterns to generate contextually appropriate suggestions and solutions.

Think of it as the difference between a contractor who walks onto a job site cold versus one who studied the blueprints, building codes, existing infrastructure, and team practices before starting work.

### Traditional vs. Modern Approaches

**Traditional approach:**
```python
# Developer manually provides context in every prompt
prompt = """
Write a function to fetch user data.
"""
# AI has no knowledge of:
# - existing HTTP client patterns
# - error handling conventions
# - logging standards
# - type definitions
# - authentication flow
```

**Context-aware approach:**
```python
# System maintains structured codebase knowledge
codebase_context = {
    "http_client": "httpx with retry middleware",
    "error_handling": "custom exceptions from app.exceptions",
    "auth": "JWT tokens via get_auth_header()",
    "logging": "structlog with correlation_id",
    "user_model": "app.models.User (Pydantic v2)"
}

prompt = """
Write a function to fetch user data.
Context: {codebase_context}
"""
# AI generates code matching existing patterns
```

### Key Engineering Insights

**1. Context is exponentially more valuable than raw capability.** A moderately capable model with accurate codebase context outperforms a highly capable model operating blindly. The difference isn't 10-20%—it's often 5-10x in terms of suggestions that actually merge without modification.

**2. Codebase awareness isn't binary—it's dimensional.** You need awareness across multiple layers: syntactic (formatting), semantic (what code does), architectural (how components interact), and conventional (team patterns). Missing any layer produces brittle results.

**3. Stale context is worse than no context.** When an AI suggests patterns from deprecated code or references deleted modules, it erodes trust faster than generic suggestions. Context freshness becomes a first-class engineering concern.

### Why This Matters Now

Modern codebases have reached a complexity threshold where context gathering dominates development time. Engineers spend 60-70% of their time reading code versus writing it. AI systems that can compress this context acquisition phase from hours to seconds fundamentally change the development economics.

Additionally, the token context windows of current models (128K-200K tokens) can now fit substantial portions of real codebases. For the first time, we can practically provide enough context for useful, project-specific assistance rather than generic suggestions.

## Technical Components

### 1. Context Extraction & Indexing

**Technical Explanation:** Before an AI can use codebase context, you need structured extraction of relevant information. This involves parsing source files, building dependency graphs, and creating searchable indexes of patterns, definitions, and relationships.

**Practical Implementation:**

```python
from pathlib import Path
from typing import Dict, List, Set
import ast
import json

class CodebaseIndexer:
    def __init__(self, root_path: Path):
        self.root_path = root_path
        self.index: Dict[str, any] = {
            "functions": {},
            "classes": {},
            "imports": {},
            "patterns": []
        }
    
    def index_python_file(self, file_path: Path) -> None:
        """Extract structured information from Python file."""
        content = file_path.read_text()
        tree = ast.parse(content)
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                self.index["functions"][node.name] = {
                    "file": str(file_path),
                    "args": [arg.arg for arg in node.args.args],
                    "returns": ast.unparse(node.returns) if node.returns else None,
                    "decorators": [ast.unparse(d) for d in node.decorator_list],
                    "docstring": ast.get_docstring(node)
                }
            
            elif isinstance(node, ast.ClassDef):
                methods = [n.name for n in node.body if isinstance(n, ast.FunctionDef)]
                self.index["classes"][node.name] = {
                    "file": str(file_path),
                    "methods": methods,
                    "bases": [ast.unparse(base) for base in node.bases]
                }
            
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    self.index["imports"][alias.name] = str(file_path)
            
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                for alias in node.names:
                    full_name = f"{module}.{alias.name}"
                    self.index["imports"][full_name] = str(file_path)
    
    def build_index(self) -> Dict:
        """Index entire codebase."""
        python_files = self.root_path.rglob("*.py")
        
        for file_path in python_files:
            if "venv" in file_path.parts or "__pycache__" in file_path.parts:
                continue
            try:
                self.index_python_file(file_path)
            except SyntaxError:
                print(f"Skipping {file_path}: syntax error")
        
        return self.index
    
    def find_related_code(self, query: str, context_type: str = "function") -> List[Dict]:
        """Find code related to query."""
        results = []
        search_space = self.index.get(context_type + "s", {})
        
        for name, metadata in search_space.items():
            if query.lower() in name.lower():
                results.append({"name": name, **metadata})
        
        return results
```

**Trade-offs:**
- **Depth vs. Speed:** Full AST parsing provides rich information but takes seconds per file. String-based regex matching is 100x faster but misses nuance.
- **Memory vs. Query Speed:** In-memory indexes enable sub-millisecond lookups but consume 50-200MB per 10K files. Disk-based indexes reduce memory but add 10-50ms query latency.
- **Update Frequency:** Real-time indexing on every file save catches changes instantly but can consume 5-10% CPU. Periodic indexing (every 5 minutes) is unnoticeable but creates context lag.

**Real Constraint:** For codebases >100K files, full re-indexing becomes prohibitive (30+ minutes). You need incremental indexing that only processes changed files, which adds complexity in tracking dependencies and invalidating cached indexes.

### 2. Context Selection & Ranking

**Technical Explanation:** With a codebase index, the challenge shifts to selecting which context to provide. Token budgets are finite—you can't send 50K lines of code with every request. You need ranking algorithms to prioritize the most relevant context.

**Practical Implementation:**

```python
from typing import List, Dict, Tuple
import numpy as np
from dataclasses import dataclass
from collections import defaultdict

@dataclass
class ContextItem:
    content: str
    relevance_score: float
    token_count: int
    category: str  # 'definition', 'example', 'dependency', 'pattern'

class ContextSelector:
    def __init__(self, token_budget: int = 8000):
        self.token_budget = token_budget
        self.category_weights = {
            'definition': 1.0,
            'example': 0.7,
            'dependency': 0.9,
            'pattern': 0.6
        }
    
    def estimate_tokens(self, text: str) -> int:
        """Rough token estimation (4 chars per token average)."""
        return len(text) // 4
    
    def score_relevance(self, item: str, query: str, metadata: Dict) -> float:
        """Score context relevance using multiple signals."""
        score = 0.0
        
        # Lexical overlap
        query_terms = set(query.lower().split())
        item_terms = set(item.lower().split())
        overlap = len(query_terms & item_terms)
        score += overlap * 2.0
        
        # Recency (prefer recently modified files)
        if 'modified_days_ago' in metadata:
            recency_score = 1.0 / (1.0 + metadata['modified_days_ago'] / 30.0)
            score += recency_score
        
        # Usage frequency (prefer commonly used patterns)
        if 'reference_count' in metadata:
            score += np.log1p(metadata['reference_count']) * 0.5
        
        # Import depth (prefer direct dependencies)
        if 'import_depth' in metadata:
            score += (3 - min(metadata['import_depth'], 3)) * 0.3
        
        return score
    
    def select_context(
        self, 
        query: str, 
        available_context: List[Tuple[str, Dict]]
    ) -> List[ContextItem]:
        """Select optimal context within token budget."""
        
        # Score all available context
        scored_items = []
        for content, metadata in available_context:
            tokens = self.estimate_tokens(content)
            base_score = self.score_relevance(content, query, metadata)
            category = metadata.get('category', 'example')
            weighted_score = base_score * self.category_weights[category]
            
            scored_items.append(ContextItem(
                content=content,
                relevance_score=weighted_score,
                token_count=tokens,
                category=category
            ))
        
        # Sort by relevance
        scored_items.sort(key=lambda x: x.relevance_score, reverse=True)
        
        # Greedy selection with diversity
        selected = []
        used_tokens = 0
        category_counts = defaultdict(int)
        
        for item in scored_items:
            # Enforce category diversity (no more than 50% from one category)
            if category_counts[item.category] >= len(selected) * 0.5:
                continue
            
            if used_tokens + item.token_count <= self.token_budget:
                selected.append(item)
                used_tokens += item.token_count
                category_counts[item.category] += 1
            
            if used_tokens > self.token_budget * 0.9:
                break
        
        return selected
    
    def format_context(self, selected: List[ContextItem]) -> str:
        """Format selected context for prompt inclusion."""
        sections = defaultdict(list)
        
        for item in selected:
            sections[item.category].append(item.content)
        
        formatted = "=== CODEBASE CONTEXT ===\n\n"
        
        for category in ['definition', 'dependency', 'example', 'pattern']:
            if sections[category]:
                formatted += f"## {category.upper()}S:\n"
                formatted += "\n---\n".join(sections[category])
                formatted += "\n\n"
        
        return formatted
```

**Trade-offs:**
- **Precision vs. Recall:** Strict relevance thresholds reduce noise but risk missing critical context. Relaxed thresholds provide comprehensive context but dilute signal.
- **Static vs. Dynamic Ranking:** Pre-computed relevance scores enable instant selection but miss query-specific nuances. Dynamic scoring considers query context but adds 50-200ms latency.

**Real Constraint:** Relevance scoring is domain-specific. Generic text similarity metrics fail for code—`handle_user_auth` and `authenticate_user` are semantically identical but lexically dissimilar. You need embeddings trained on code, which adds infrastructure complexity.

### 3. Context Freshness & Invalidation

**Technical Explanation:** Codebase context must stay synchronized with actual code. This requires tracking file changes, invalidating stale indexes, and propagating updates through dependent context.

**Practical Implementation:**

```python
from pathlib import Path
from typing import Dict, Set
import hashlib
import time
import json

class ContextCache:
    def __init__(self, cache_dir: Path):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(exist_ok=True)
        self.file_hashes: Dict[str, str] = {}
        self.dependency_graph: Dict[str, Set[str]] = {}
        self.load_cache_state()
    
    def compute_file_hash(self, file_path: Path) -> str:
        """Compute hash of file content."""
        content = file_path.read_bytes()
        return hashlib.sha256(content).hexdigest()
    
    def load_cache_state(self) -> None:
        """Load cached file hashes and dependencies."""
        state_file = self.cache_dir / "cache_state.json"
        if state_file.exists():
            data = json.loads(state_file.read_text())
            self.file_hashes = data.get("file_hashes", {})
            self.dependency_graph = {
                k: set(v) for k, v in data.get("dependency_graph", {}).items()
            }
    
    def save_cache_state(self) -> None:
        """Persist cache state to disk."""
        state_file = self.cache_dir / "cache_state.json"
        data = {
            "file_hashes": self.file_hashes,
            "dependency_graph": {k: list(v) for k, v in self.dependency_graph.items()}
        }
        state_file.write_text(json.dumps(data, indent=2))
    
    def track_dependencies(self, file_path: str, imports: List[str]) -> None:
        """Record which files depend on which modules."""
        if file_path not in self.dependency_graph:
            self.dependency_graph[file_path] = set()
        self.dependency_graph[file_path].update(imports)
    
    def find_affected_files(self, changed_file: str) -> Set[str]:
        """Find all files affected by a change."""
        affected = {changed_file}
        to_check = {changed_file}
        
        while to_check:
            current = to_check.pop()
            # Find files that import current file
            for file_path, dependencies in self.dependency_graph.items():
                if current in dependencies and file_path not in affected:
                    affected.add(file_path)
                    to_check.add(file_path)
        
        return affected
    
    def check_and_invalidate(self, file_path: Path) -> Set[str]:
        """Check if file changed and invalidate affected context."""
        file_str = str(file_path)
        current_hash = self.compute_file_hash(file_path)
        
        if file_str in self.file_hashes:
            if self.file_hashes[file_str] != current_hash:
                # File changed - find affected context
                affected = self.find_affected_files(file_str)
                
                # Invalidate cached context for all affected files
                for affected_file in affected:
                    cache_file = self.cache_dir / f"{hashlib.md5(affected_file.encode()).hexdigest()}.json"
                    if cache_file.exists():
                        cache_file.unlink()
                
                # Update hash
                self.file_hashes[file_str] = current_hash
                self.save_cache_state()
                
                return affected
        else:
            # New file
            self.file_hashes[file_str] = current_hash
            self.save_cache_state()
        
        return set()
    
    def get_cached_context(self, query: str) -> Dict | None:
        """Retrieve cached context if valid."""
        query_hash = hashlib.md5(query.encode()).hexdigest()
        cache_file = self.cache_dir / f"query_{query_hash}.json"
        
        if cache_file.exists():
            data = json.loads(cache_file.read_text())
            # Check if cache is stale (>5 minutes old)
            if time.time() - data['timestamp'] < 300:
                return data['context']
        
        return None
    
    def cache_context(self, query: str, context: Dict) -> None:
        """Cache context for query."""
        query_hash = hashlib.md5(query.encode()).hexdigest()
        cache_file = self.cache_