# Auto-Generated API Documentation with LLMs

## Core Concepts

Auto-generated API documentation using LLMs transforms code parsing and template filling into intelligent documentation synthesis. Instead of extracting function signatures and docstrings into formatted HTML, LLMs analyze code structure, infer intent, generate usage examples, and produce documentation that explains *why* alongside *what*.

### Traditional vs. LLM-Powered Approach

```python
# Traditional: Sphinx/JSDoc approach
def calculate_discount(base_price: float, customer_tier: str, 
                       promo_code: Optional[str] = None) -> float:
    """
    Calculate discount for a customer.
    
    Args:
        base_price: Original price
        customer_tier: Customer tier level
        promo_code: Optional promotional code
        
    Returns:
        Final discounted price
    """
    # Implementation...
    pass

# Output: Literal docstring rendering with type info
```

```python
# LLM-Powered: Understanding context and relationships
from typing import Optional
import anthropic

def generate_api_docs(source_code: str, context: dict) -> str:
    """Generate comprehensive documentation from code analysis."""
    client = anthropic.Anthropic()
    
    prompt = f"""Analyze this function and generate documentation that:
1. Explains the business logic and use cases
2. Shows realistic usage examples
3. Describes edge cases and error conditions
4. Links to related functions

Code:
{source_code}

Related context:
{context}

Generate markdown documentation with examples."""
    
    response = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=2000,
        messages=[{"role": "user", "content": prompt}]
    )
    
    return response.content[0].text

# Output: Contextual explanation with examples, edge cases, 
# relationships to other APIs, and usage patterns
```

The traditional approach extracts what's explicitly written. The LLM approach *understands* the code's purpose, generates realistic examples, identifies edge cases developers forgot to document, and explains relationships between components.

### Key Insights

**Context is everything**: LLMs excel when given repository structure, related functions, test files, and existing documentation. A function in isolation gets generic docs; a function with its test suite and calling code gets documentation that explains actual usage patterns.

**Documentation debt becomes manageable**: Instead of retroactively writing docs for 200 undocumented endpoints, you can generate first-draft documentation in minutes, then refine only what matters. The LLM catches inconsistencies between implementation and intent.

**Living documentation**: Code changes invalidate docs instantly. LLMs can regenerate affected documentation automatically, comparing old and new implementations to highlight what changed for users.

### Why This Matters Now

Modern APIs ship with thousands of endpoints. Manual documentation creates bottlenecks: engineers hate writing it, docs go stale immediately, and quality varies wildly. LLMs don't replace technical writers—they eliminate the grinding work of translating code to prose, letting humans focus on architecture decisions, design rationale, and complex workflows.

OpenAPI specs define structure but lack narrative. Generated docs can explain *when* to use `/users/{id}/preferences` versus `/preferences?user_id={id}`, not just that both exist. For internal APIs, this documentation quality directly impacts development velocity.

## Technical Components

### 1. Code Context Extraction

Before asking an LLM to document anything, you need structured context. The LLM needs to understand not just one function but its ecosystem.

**Technical Explanation**: Context extraction involves parsing source code into an abstract syntax tree (AST), identifying relationships (calls, imports, inheritance), and collecting metadata (type hints, decorators, test coverage). This structured data becomes the foundation for prompts.

```python
import ast
from pathlib import Path
from typing import Dict, List, Set

class CodeContextExtractor:
    def __init__(self, repo_path: Path):
        self.repo_path = repo_path
        self.function_calls: Dict[str, Set[str]] = {}
        self.imports: Dict[str, List[str]] = {}
        
    def extract_function_context(self, file_path: Path, 
                                 function_name: str) -> Dict:
        """Extract comprehensive context for a single function."""
        with open(file_path, 'r') as f:
            source = f.read()
            
        tree = ast.parse(source)
        
        context = {
            'source_code': '',
            'dependencies': [],
            'callers': [],
            'test_examples': [],
            'related_functions': []
        }
        
        # Find the target function
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == function_name:
                context['source_code'] = ast.get_source_segment(source, node)
                context['dependencies'] = self._extract_calls(node)
                context['decorators'] = [d.id for d in node.decorator_list 
                                        if isinstance(d, ast.Name)]
                
        # Find callers across repository
        context['callers'] = self._find_callers(function_name)
        
        # Find associated tests
        test_file = self._find_test_file(file_path)
        if test_file:
            context['test_examples'] = self._extract_test_cases(
                test_file, function_name
            )
            
        return context
    
    def _extract_calls(self, node: ast.FunctionDef) -> List[str]:
        """Extract function calls within this function."""
        calls = []
        for child in ast.walk(node):
            if isinstance(child, ast.Call):
                if isinstance(child.func, ast.Name):
                    calls.append(child.func.id)
                elif isinstance(child.func, ast.Attribute):
                    calls.append(f"{child.func.value.id}.{child.func.attr}")
        return calls
    
    def _find_callers(self, function_name: str) -> List[Dict]:
        """Find where this function is called across the codebase."""
        callers = []
        for py_file in self.repo_path.rglob("*.py"):
            try:
                with open(py_file, 'r') as f:
                    source = f.read()
                if function_name in source:
                    # Parse to verify it's actually a call
                    tree = ast.parse(source)
                    for node in ast.walk(tree):
                        if isinstance(node, ast.Call):
                            if (isinstance(node.func, ast.Name) and 
                                node.func.id == function_name):
                                callers.append({
                                    'file': str(py_file),
                                    'context': ast.get_source_segment(
                                        source, node
                                    )
                                })
            except:
                continue
        return callers
    
    def _find_test_file(self, source_file: Path) -> Optional[Path]:
        """Locate corresponding test file."""
        test_patterns = [
            source_file.parent / f"test_{source_file.name}",
            source_file.parent / "tests" / f"test_{source_file.name}",
            self.repo_path / "tests" / source_file.name
        ]
        for pattern in test_patterns:
            if pattern.exists():
                return pattern
        return None
    
    def _extract_test_cases(self, test_file: Path, 
                           function_name: str) -> List[str]:
        """Extract test cases that exercise this function."""
        with open(test_file, 'r') as f:
            source = f.read()
            
        tree = ast.parse(source)
        test_cases = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                if node.name.startswith('test_'):
                    # Check if this test calls our function
                    for child in ast.walk(node):
                        if isinstance(child, ast.Call):
                            if (isinstance(child.func, ast.Name) and 
                                child.func.id == function_name):
                                test_cases.append(
                                    ast.get_source_segment(source, node)
                                )
                                break
        return test_cases
```

**Practical Implications**: Rich context produces dramatically better documentation. A function documented with its test cases generates examples that actually work. Knowing callers helps explain common usage patterns.

**Constraints**: AST parsing only works for syntactically valid code. Dynamic languages like Python have runtime behavior that static analysis misses. Cross-repository dependencies require additional tooling.

### 2. Prompt Engineering for Documentation

Generic "document this function" prompts produce generic documentation. Effective prompts structure the task, provide examples, and specify format.

**Technical Explanation**: Documentation prompts need clear roles, structured inputs, and output specifications. The LLM should understand it's generating developer-facing content, not end-user tutorials.

```python
from typing import Dict, List
import anthropic

class DocumentationGenerator:
    def __init__(self, api_key: str):
        self.client = anthropic.Anthropic(api_key=api_key)
        
    def generate_function_docs(self, context: Dict) -> str:
        """Generate comprehensive function documentation."""
        
        prompt = self._build_prompt(context)
        
        response = self.client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=3000,
            temperature=0.3,  # Lower temperature for consistency
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )
        
        return response.content[0].text
    
    def _build_prompt(self, context: Dict) -> str:
        """Construct structured prompt with all context."""
        
        # Start with role and task definition
        prompt = """You are a technical documentation expert. Generate comprehensive 
API documentation for the following function. Focus on practical usage for 
developers integrating this API.

"""
        
        # Include the actual source code
        prompt += f"""## Function Source Code
```python
{context['source_code']}
```

"""
        
        # Add test examples if available
        if context.get('test_examples'):
            prompt += "## Test Examples\n"
            for test in context['test_examples'][:3]:  # Limit to 3 examples
                prompt += f"```python\n{test}\n```\n\n"
        
        # Add usage context
        if context.get('callers'):
            prompt += "## Real Usage Examples from Codebase\n"
            for caller in context['callers'][:3]:
                prompt += f"File: {caller['file']}\n```python\n{caller['context']}\n```\n\n"
        
        # Add dependencies for relationship context
        if context.get('dependencies'):
            prompt += f"## Called Functions\n{', '.join(context['dependencies'])}\n\n"
        
        # Specify output format
        prompt += """## Generate Documentation With:

1. **Overview**: One paragraph explaining what this function does and why you'd use it

2. **Parameters**: For each parameter:
   - Type and constraints
   - What it controls
   - Valid values/ranges
   - Common pitfalls

3. **Return Value**: 
   - Type and structure
   - Success/error cases
   - What to do with the result

4. **Usage Examples**: 
   - Basic usage (happy path)
   - Advanced usage with options
   - Error handling

5. **Edge Cases & Errors**:
   - Invalid inputs
   - Error messages and meanings
   - How to handle failures

6. **Related APIs**: Functions commonly used with this one

Format as markdown. Be specific and practical."""
        
        return prompt
    
    def generate_endpoint_docs(self, endpoint_spec: Dict, 
                               implementation: str,
                               example_requests: List[Dict]) -> str:
        """Generate REST API endpoint documentation."""
        
        prompt = f"""Generate REST API documentation for this endpoint.

## Endpoint Specification
- Method: {endpoint_spec['method']}
- Path: {endpoint_spec['path']}
- Auth: {endpoint_spec.get('auth', 'Not specified')}

## Implementation
```python
{implementation}
```

## Example Requests from Logs
"""
        for req in example_requests[:5]:
            prompt += f"""
Request:
```json
{req.get('request')}
```
Response:
```json
{req.get('response')}
```
"""
        
        prompt += """
Generate documentation including:
1. Endpoint purpose and use cases
2. Request format with all parameters explained
3. Response format with field descriptions
4. Status codes and error responses
5. Rate limits and quotas (if applicable)
6. Complete curl and Python examples
7. Common integration patterns

Be specific about authentication, required headers, and error handling."""
        
        response = self.client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=4000,
            temperature=0.3,
            messages=[{"role": "user", "content": prompt}]
        )
        
        return response.content[0].text
```

**Practical Implications**: Well-structured prompts with real usage examples generate documentation that answers actual developer questions. Including test cases means examples that compile and run.

**Trade-offs**: Longer prompts cost more and slower generation. Balance context richness against token costs—prioritize recent, representative examples over exhaustive coverage.

### 3. Output Formatting and Integration

Generated documentation needs consistent formatting and integration into existing documentation systems.

**Technical Explanation**: LLMs produce text; you need structured output that fits your documentation pipeline (Markdown, OpenAPI, Docusaurus, etc.). Post-processing validates format, adds metadata, and handles edge cases.

```python
import re
from typing import Dict, Optional
from pathlib import Path

class DocumentationFormatter:
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(exist_ok=True)
        
    def format_and_save(self, raw_docs: str, 
                       metadata: Dict,
                       output_format: str = "markdown") -> Path:
        """Format generated docs and save to appropriate location."""
        
        if output_format == "markdown":
            formatted = self._format_markdown(raw_docs, metadata)
            file_ext = ".md"
        elif output_format == "openapi":
            formatted = self._format_openapi(raw_docs, metadata)
            file_ext = ".yaml"
        else:
            raise ValueError(f"Unsupported format: {output_format}")
        
        # Generate filename from function/endpoint name
        filename = self._generate_filename(metadata) + file_ext
        output_path = self.output_dir / filename
        
        with open(output_path, 'w') as f:
            f.write(formatted)
            
        return output_path
    
    def _format_markdown(self, raw_docs: str, metadata: Dict) -> str:
        """Format as markdown with frontmatter."""
        
        # Add frontmatter for static site generators
        frontmatter = f"""---
title: {metadata.get('title', 'API Documentation')}
category: {metadata.get('category', 'API')}
generated: {metadata.get('generated_at', 'unknown')}
source: {metadata.get('source_file', 'unknown')}
---

"""
        
        # Clean up LLM output
        cleaned = self._clean_markdown(raw_docs)
        
        # Ensure code blocks have language specified
        cleaned = self._fix_code_blocks(cleaned)
        
        # Add navigation links if part of multi-page docs
        if metadata.get('related_pages'):
            navigation = "\n## Related Documentation\n\n"
            for page in metadata['related_pages']:
                navigation += f"- [{page['title']}]({page['path']})\n"
            cleaned += navigation
        
        return frontmatter + cleaned
    
    def _clean_markdown(self, text: str) -> str:
        """Clean up common LLM markdown issues."""
        
        # Remove excessive newlines
        text = re.sub(r'\n{4,}', '\n\n\n', text)
        
        # Ensure headers have proper spacing
        text = re.sub(r'(#+[^\n]+)\n([^\n])', r'\1\n\n\2', text)
        
        # Fix bullet point inconsistencies
        text = re.sub(r'\n-([^\s])', r'\n- \1', text)