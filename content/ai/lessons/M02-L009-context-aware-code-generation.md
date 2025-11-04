# Context-Aware Code Generation

## Core Concepts

Context-aware code generation represents a fundamental shift from template-based code synthesis to dynamic generation that understands project structure, coding patterns, dependencies, and architectural constraints. Unlike traditional code generators that apply fixed templates regardless of surrounding code, context-aware systems analyze your existing codebase to generate code that matches your style, respects your architecture, and integrates seamlessly with existing components.

### Traditional vs. Context-Aware Generation

```python
# Traditional template-based generation
# Input: "Create a user class"
# Output: Generic template, always the same

class User:
    def __init__(self, name, email):
        self.name = name
        self.email = email
    
    def get_name(self):
        return self.name

# Context-aware generation
# Analyzes existing codebase and finds:
# - You use dataclasses throughout
# - You have a BaseEntity with standard fields
# - You use type hints consistently
# - You have a validation pattern with pydantic

from dataclasses import dataclass
from datetime import datetime
from typing import Optional
from .base import BaseEntity
from .validators import EmailValidator

@dataclass
class User(BaseEntity):
    name: str
    email: str
    created_at: datetime = datetime.utcnow()
    last_login: Optional[datetime] = None
    
    def __post_init__(self):
        EmailValidator.validate(self.email)
        super().__post_init__()
```

The key insight: Context-aware generation treats your codebase as a living specification. Instead of asking "what code should I generate for this task?", it asks "what code would a developer familiar with this codebase write for this task?"

### Why This Matters Now

Three converging factors make context-aware code generation critical:

1. **Token context windows** have expanded from 2K to 200K+ tokens, allowing models to consume entire codebases as context
2. **Retrieval-augmented generation (RAG)** enables efficient selection of relevant code examples without exhausting context limits
3. **Fine-tuning and embedding models** can learn project-specific patterns that pure prompt engineering cannot capture

This isn't about replacing developers—it's about eliminating the mechanical translation from "I know what needs to exist" to "I've typed all the boilerplate." The cognitive work remains; the manual transcription vanishes.

## Technical Components

### 1. Context Extraction and Representation

Context extraction transforms unstructured code into structured information the model can reason about. This involves three layers:

**Syntactic Layer:** Parse trees, ASTs, symbol tables
**Semantic Layer:** Type relationships, dependency graphs, data flow
**Pragmatic Layer:** Naming conventions, architectural patterns, idioms

```python
from typing import Dict, List, Set
import ast
from pathlib import Path

class CodebaseContextExtractor:
    def __init__(self, root_path: Path):
        self.root_path = root_path
        self.class_definitions: Dict[str, ast.ClassDef] = {}
        self.function_signatures: Dict[str, ast.FunctionDef] = {}
        self.import_patterns: List[str] = []
        
    def extract_project_patterns(self) -> Dict[str, any]:
        """Extract reusable patterns from codebase."""
        patterns = {
            "class_style": self._detect_class_style(),
            "error_handling": self._detect_error_patterns(),
            "naming_conventions": self._analyze_naming(),
            "common_imports": self._get_import_frequency(),
        }
        return patterns
    
    def _detect_class_style(self) -> str:
        """Determine if codebase uses dataclasses, attrs, or traditional classes."""
        style_votes = {"dataclass": 0, "attrs": 0, "traditional": 0}
        
        for py_file in self.root_path.rglob("*.py"):
            tree = ast.parse(py_file.read_text())
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    if any(d.id == "dataclass" for d in node.decorator_list 
                           if isinstance(d, ast.Name)):
                        style_votes["dataclass"] += 1
                    elif any("attr" in ast.unparse(d) for d in node.decorator_list):
                        style_votes["attrs"] += 1
                    else:
                        style_votes["traditional"] += 1
        
        return max(style_votes, key=style_votes.get)
    
    def _detect_error_patterns(self) -> Dict[str, int]:
        """Identify how errors are handled across codebase."""
        patterns = {
            "custom_exceptions": 0,
            "return_none": 0,
            "raise_builtin": 0,
            "result_type": 0,
        }
        
        for py_file in self.root_path.rglob("*.py"):
            content = py_file.read_text()
            if "raise Custom" in content or "raise Application" in content:
                patterns["custom_exceptions"] += 1
            if "return None" in content and "except" in content:
                patterns["return_none"] += 1
            if "Result[" in content or "Option[" in content:
                patterns["result_type"] += 1
                
        return patterns
```

**Practical Implications:** When generating code, the model references these patterns to make consistent choices. If your codebase uses custom exceptions 80% of the time, generated code should follow suit.

**Trade-offs:** Deep analysis requires time and compute. For interactive generation, cache extracted patterns and update incrementally. Full re-analysis might run nightly or on significant commits.

### 2. Context Window Management

Modern models offer large context windows, but context is a limited resource requiring strategic allocation:

```python
from typing import List, Tuple
from dataclasses import dataclass
import tiktoken

@dataclass
class CodeContext:
    content: str
    relevance_score: float
    token_count: int
    context_type: str  # "definition", "example", "documentation"

class ContextWindowManager:
    def __init__(self, model_name: str = "gpt-4", max_tokens: int = 8000):
        self.encoding = tiktoken.encoding_for_model(model_name)
        self.max_tokens = max_tokens
        self.reserved_tokens = 1000  # Reserve for response
        
    def count_tokens(self, text: str) -> int:
        return len(self.encoding.encode(text))
    
    def select_context(
        self, 
        available_contexts: List[CodeContext],
        mandatory_contexts: List[CodeContext],
        generation_task: str
    ) -> List[CodeContext]:
        """
        Select optimal context mix within token budget.
        
        Priority:
        1. Mandatory contexts (interface definitions, base classes)
        2. High-relevance examples
        3. Documentation/comments
        """
        budget = self.max_tokens - self.reserved_tokens
        
        # Mandatory contexts first
        selected = []
        used_tokens = 0
        
        for ctx in mandatory_contexts:
            if used_tokens + ctx.token_count > budget:
                # Truncate if necessary
                available = budget - used_tokens
                truncated = self._truncate_context(ctx, available)
                selected.append(truncated)
                break
            selected.append(ctx)
            used_tokens += ctx.token_count
        
        # Fill remaining budget with highest relevance contexts
        remaining = budget - used_tokens
        optional = sorted(
            available_contexts, 
            key=lambda x: x.relevance_score / x.token_count,  # Efficiency score
            reverse=True
        )
        
        for ctx in optional:
            if ctx.token_count <= remaining:
                selected.append(ctx)
                remaining -= ctx.token_count
                
        return selected
    
    def _truncate_context(self, ctx: CodeContext, token_limit: int) -> CodeContext:
        """Intelligently truncate context to fit token limit."""
        if ctx.context_type == "definition":
            # Keep signatures, remove implementations
            lines = ctx.content.split("\n")
            truncated = []
            for line in lines:
                if self.count_tokens("\n".join(truncated + [line])) > token_limit:
                    break
                truncated.append(line)
            return CodeContext(
                content="\n".join(truncated),
                relevance_score=ctx.relevance_score,
                token_count=self.count_tokens("\n".join(truncated)),
                context_type=ctx.context_type
            )
        return ctx
```

**Practical Implications:** Context selection dramatically impacts output quality. A definition without usage examples produces generic code. Examples without definitions produce code that might not integrate correctly.

**Constraints:** Token counting isn't free—cache counts for static content. Relevance scoring requires either embeddings (compute cost) or heuristics (accuracy trade-off).

### 3. Retrieval Mechanisms for Code

When your codebase exceeds context limits, retrieval determines what the model sees:

```python
from typing import List, Tuple
import numpy as np
from pathlib import Path

class CodeRetriever:
    def __init__(self, embedding_model):
        self.embedding_model = embedding_model
        self.code_chunks: List[Tuple[str, np.ndarray, Dict]] = []
        
    def index_codebase(self, root_path: Path):
        """Create searchable index of code chunks."""
        for py_file in root_path.rglob("*.py"):
            chunks = self._chunk_file(py_file)
            for chunk, metadata in chunks:
                embedding = self.embedding_model.embed(chunk)
                self.code_chunks.append((chunk, embedding, metadata))
    
    def _chunk_file(self, file_path: Path) -> List[Tuple[str, Dict]]:
        """
        Chunk code at logical boundaries (class/function level).
        Preserve context by including docstrings and signatures.
        """
        tree = ast.parse(file_path.read_text())
        chunks = []
        
        for node in ast.walk(tree):
            if isinstance(node, (ast.ClassDef, ast.FunctionDef)):
                chunk_text = ast.unparse(node)
                metadata = {
                    "file": str(file_path),
                    "type": type(node).__name__,
                    "name": node.name,
                    "line_start": node.lineno,
                }
                chunks.append((chunk_text, metadata))
                
        return chunks
    
    def retrieve_relevant_code(
        self, 
        query: str, 
        top_k: int = 5,
        filters: Dict[str, any] = None
    ) -> List[Tuple[str, float, Dict]]:
        """
        Retrieve most relevant code chunks for generation task.
        
        Hybrid approach:
        - Semantic similarity (embeddings)
        - Structural similarity (AST patterns)
        - Dependency proximity (import graph)
        """
        query_embedding = self.embedding_model.embed(query)
        
        scored_chunks = []
        for chunk, embedding, metadata in self.code_chunks:
            # Apply filters
            if filters and not self._matches_filters(metadata, filters):
                continue
                
            # Semantic similarity
            similarity = np.dot(query_embedding, embedding)
            
            # Boost score based on metadata
            if self._is_structural_match(query, metadata):
                similarity *= 1.5
                
            scored_chunks.append((chunk, similarity, metadata))
        
        # Sort by score and return top_k
        scored_chunks.sort(key=lambda x: x[1], reverse=True)
        return scored_chunks[:top_k]
    
    def _matches_filters(self, metadata: Dict, filters: Dict) -> bool:
        """Check if chunk matches filter criteria."""
        for key, value in filters.items():
            if metadata.get(key) != value:
                return False
        return True
    
    def _is_structural_match(self, query: str, metadata: Dict) -> bool:
        """Boost chunks that match structural patterns in query."""
        query_lower = query.lower()
        if "class" in query_lower and metadata["type"] == "ClassDef":
            return True
        if "function" in query_lower and metadata["type"] == "FunctionDef":
            return True
        return False
```

**Practical Implications:** Retrieval quality determines whether the model sees relevant examples. Poor retrieval means the model generates code based on general knowledge, not your specific patterns.

**Trade-offs:** Embedding every code chunk is expensive upfront but enables fast retrieval. Alternatively, use simpler keyword/AST matching for speed at the cost of semantic understanding.

### 4. Prompt Construction for Code Generation

The prompt architecture determines how context translates to output:

```python
from typing import List, Optional
from dataclasses import dataclass

@dataclass
class GenerationRequest:
    task_description: str
    target_file: Optional[Path] = None
    target_function: Optional[str] = None
    style_constraints: Optional[Dict] = None

class CodeGenerationPromptBuilder:
    def __init__(self, context_manager: ContextWindowManager):
        self.context_manager = context_manager
        
    def build_prompt(
        self,
        request: GenerationRequest,
        relevant_code: List[CodeContext],
        project_patterns: Dict[str, any]
    ) -> str:
        """
        Construct prompt with optimal context ordering.
        
        Structure:
        1. System context (project patterns)
        2. Relevant definitions (interfaces, base classes)
        3. Usage examples
        4. Task specification
        5. Output format instructions
        """
        
        sections = []
        
        # Section 1: Project patterns
        sections.append(self._format_project_context(project_patterns))
        
        # Section 2: Required definitions
        definitions = [ctx for ctx in relevant_code if ctx.context_type == "definition"]
        if definitions:
            sections.append("## Relevant Definitions\n")
            sections.extend([ctx.content for ctx in definitions])
        
        # Section 3: Usage examples
        examples = [ctx for ctx in relevant_code if ctx.context_type == "example"]
        if examples:
            sections.append("## Usage Examples\n")
            sections.extend([ctx.content for ctx in examples[:3]])  # Limit examples
        
        # Section 4: Task specification
        sections.append(f"## Task\n{request.task_description}\n")
        
        # Section 5: Output format
        sections.append(self._format_output_instructions(request))
        
        return "\n\n".join(sections)
    
    def _format_project_context(self, patterns: Dict[str, any]) -> str:
        """Format project patterns as clear instructions."""
        context = "## Project Context\n\n"
        
        if patterns.get("class_style") == "dataclass":
            context += "- Use @dataclass for data structures\n"
        
        error_patterns = patterns.get("error_handling", {})
        if error_patterns.get("custom_exceptions", 0) > 5:
            context += "- Raise custom exceptions (not generic Exception)\n"
        
        if patterns.get("naming_conventions"):
            context += f"- Follow naming: {patterns['naming_conventions']}\n"
            
        return context
    
    def _format_output_instructions(self, request: GenerationRequest) -> str:
        """Specify exactly what format the output should take."""
        instructions = "## Output Requirements\n\n"
        instructions += "Generate complete, runnable code with:\n"
        instructions += "- All necessary imports\n"
        instructions += "- Type hints for function signatures\n"
        instructions += "- Docstrings for public functions/classes\n"
        instructions += "- Error handling matching project patterns\n"
        
        if request.target_file:
            instructions += f"\nCode should fit in: {request.target_file}\n"
            
        return instructions
```

**Practical Implications:** Prompt structure affects consistency more than most developers expect. Place critical constraints early and repeat them in output instructions.

**Constraints:** Longer prompts consume more tokens and increase latency. Monitor which sections actually improve output quality—remove those that don't.

### 5. Validation and Integration

Generated code must integrate seamlessly with existing code:

```python
import subprocess
from typing import Optional, List, Dict

class GeneratedCodeValidator:
    def