# Onboarding Documentation Automation

## Core Concepts

### Technical Definition

Onboarding documentation automation uses large language models to dynamically generate, update, and personalize technical documentation based on user context, role requirements, and system state. Unlike static documentation that requires manual maintenance across versions, automated systems treat documentation as computed output—generated from source code, system metadata, configuration files, and usage patterns.

### Traditional vs. Modern Approach

**Traditional approach:**

```python
# Manual documentation maintenance
class UserOnboarding:
    def __init__(self):
        # Static markdown files maintained separately
        self.docs = {
            'setup': 'docs/setup.md',
            'api_guide': 'docs/api.md',
            'deployment': 'docs/deploy.md'
        }
    
    def get_documentation(self, user_role: str) -> str:
        # Return same docs regardless of context
        with open(self.docs['setup'], 'r') as f:
            return f.read()
    
    # Problems:
    # - Documentation drifts from codebase
    # - Same content for all users/contexts
    # - Manual updates across 50+ files
    # - No awareness of user's actual environment
```

**Modern LLM-powered approach:**

```python
from typing import Dict, List, Optional
import anthropic
import inspect
import json

class DynamicOnboarding:
    def __init__(self, api_key: str):
        self.client = anthropic.Anthropic(api_key=api_key)
        
    def generate_documentation(
        self,
        user_context: Dict[str, any],
        code_artifacts: List[str],
        existing_setup: Optional[Dict] = None
    ) -> str:
        """Generate contextual documentation from actual system state."""
        
        # Extract current system information
        system_info = self._analyze_codebase(code_artifacts)
        
        prompt = f"""Generate onboarding documentation for a {user_context['role']} 
with {user_context['experience_level']} experience.

Current System State:
{json.dumps(system_info, indent=2)}

User's Existing Setup:
{json.dumps(existing_setup or {}, indent=2)}

Requirements:
1. Skip steps already completed
2. Highlight changes since last documented version
3. Include environment-specific commands for {user_context.get('os', 'unix')}
4. Focus on {user_context.get('primary_goal', 'general setup')}

Generate step-by-step documentation with actual commands they can copy-paste."""

        message = self.client.messages.create(
            model="claude-sonnet-4-0",
            max_tokens=4096,
            messages=[{"role": "user", "content": prompt}]
        )
        
        return message.content[0].text
    
    def _analyze_codebase(self, artifacts: List[str]) -> Dict:
        """Extract documentation-relevant metadata from code."""
        info = {
            'dependencies': self._extract_dependencies(),
            'api_endpoints': self._extract_endpoints(),
            'config_requirements': self._extract_config(),
            'version': self._get_current_version()
        }
        return info
```

### Key Engineering Insights

**1. Documentation as Computed Output**

Traditional documentation is a separate artifact that requires synchronization. Modern approach treats it as a view computed from source truth—similar to how database views are computed from tables. When code changes, documentation automatically reflects those changes.

**2. Context-Aware Content Generation**

The same codebase requires different documentation for different contexts:
- Frontend developer needs API endpoint details
- DevOps engineer needs deployment configuration
- New hire needs conceptual overview first
- Experienced developer needs only what's changed

LLMs excel at transforming the same source material into appropriately filtered and formatted content.

**3. Progressive Disclosure Through Interaction**

Instead of overwhelming users with comprehensive documentation, LLM systems can implement progressive disclosure—providing information just-in-time based on user questions and observed confusion points.

### Why This Matters Now

**Technical debt reduction:** Engineering teams spend 15-25% of time maintaining documentation. This compounds as codebases grow—a 100K LOC project might have 500+ documentation pages requiring manual sync.

**Onboarding velocity:** New engineers typically need 2-4 weeks to become productive. Context-aware documentation can reduce this by 40-60% by eliminating time spent on irrelevant information and outdated instructions.

**API surface area explosion:** Modern systems expose hundreds of endpoints, configuration options, and integration patterns. Static documentation cannot efficiently serve all user contexts without becoming overwhelming.

## Technical Components

### Component 1: Source Material Extraction

**Technical Explanation**

Source material extraction transforms code, configuration, and metadata into structured formats suitable for LLM processing. This involves parsing abstract syntax trees (AST), analyzing dependency graphs, extracting type signatures, and correlating documentation comments with implementation.

**Practical Implementation**

```python
import ast
from typing import Dict, List, Any
from pathlib import Path
import yaml

class SourceMaterialExtractor:
    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        
    def extract_api_surface(self, filepath: str) -> Dict[str, Any]:
        """Extract public API from Python module."""
        with open(filepath, 'r') as f:
            tree = ast.parse(f.read())
        
        api_surface = {
            'classes': [],
            'functions': [],
            'constants': []
        }
        
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                if not node.name.startswith('_'):
                    api_surface['classes'].append({
                        'name': node.name,
                        'methods': [m.name for m in node.body 
                                   if isinstance(m, ast.FunctionDef) 
                                   and not m.name.startswith('_')],
                        'docstring': ast.get_docstring(node),
                        'init_params': self._extract_init_params(node)
                    })
            
            elif isinstance(node, ast.FunctionDef):
                if not node.name.startswith('_') and node.col_offset == 0:
                    api_surface['functions'].append({
                        'name': node.name,
                        'params': [arg.arg for arg in node.args.args],
                        'docstring': ast.get_docstring(node),
                        'return_annotation': self._get_annotation(node.returns)
                    })
        
        return api_surface
    
    def extract_configuration_schema(self, config_file: str) -> Dict:
        """Extract configuration requirements."""
        with open(config_file, 'r') as f:
            if config_file.endswith('.yaml') or config_file.endswith('.yml'):
                config = yaml.safe_load(f)
            else:
                config = json.load(f)
        
        # Build schema with descriptions
        schema = self._infer_schema(config)
        return schema
    
    def extract_dependencies(self) -> Dict[str, str]:
        """Extract dependency versions and purposes."""
        requirements = self.project_root / 'requirements.txt'
        dependencies = {}
        
        if requirements.exists():
            with open(requirements, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        if '==' in line:
                            pkg, version = line.split('==')
                            dependencies[pkg.strip()] = version.strip()
                        else:
                            dependencies[line] = 'latest'
        
        return dependencies
    
    def _extract_init_params(self, class_node: ast.ClassDef) -> List[Dict]:
        """Extract __init__ parameters with type hints."""
        for node in class_node.body:
            if isinstance(node, ast.FunctionDef) and node.name == '__init__':
                return [{
                    'name': arg.arg,
                    'annotation': self._get_annotation(arg.annotation)
                } for arg in node.args.args if arg.arg != 'self']
        return []
    
    def _get_annotation(self, annotation) -> str:
        """Convert AST annotation to string."""
        if annotation is None:
            return 'Any'
        return ast.unparse(annotation)
    
    def _infer_schema(self, config: Dict, prefix: str = '') -> Dict:
        """Recursively infer configuration schema."""
        schema = {}
        for key, value in config.items():
            full_key = f"{prefix}.{key}" if prefix else key
            schema[full_key] = {
                'type': type(value).__name__,
                'required': True,
                'example': value
            }
            if isinstance(value, dict):
                schema.update(self._infer_schema(value, full_key))
        return schema
```

**Real Constraints**

- AST parsing fails on dynamically generated code
- Type hints may be incomplete or absent in older codebases
- Configuration schemas often implicit rather than explicit
- Extraction time scales linearly with codebase size

**Trade-offs**

Static analysis provides guaranteed accuracy but limited coverage. Runtime introspection captures dynamic behavior but requires execution environment. Hybrid approaches balance coverage and safety.

### Component 2: Context Assembly

**Technical Explanation**

Context assembly combines extracted source material with user-specific information to create LLM prompts. This involves template composition, token budget management, relevance filtering, and maintaining consistency across multi-turn interactions.

**Practical Implementation**

```python
from dataclasses import dataclass
from typing import List, Optional
import tiktoken

@dataclass
class UserContext:
    role: str
    experience_level: str  # 'junior', 'mid', 'senior'
    primary_goal: str
    os: str
    completed_steps: List[str]
    pain_points: List[str]

class ContextAssembler:
    def __init__(self, model: str = "claude-sonnet-4-0"):
        self.encoder = tiktoken.encoding_for_model("gpt-4")  # Approximation
        self.max_context_tokens = 180000
        self.target_output_tokens = 4000
        self.max_input_tokens = self.max_context_tokens - self.target_output_tokens
        
    def assemble_prompt(
        self,
        user_context: UserContext,
        source_materials: Dict[str, Any],
        task_type: str
    ) -> str:
        """Assemble contextually-appropriate prompt within token budget."""
        
        # Prioritize information based on user context
        prioritized_materials = self._prioritize_materials(
            source_materials, 
            user_context
        )
        
        # Build prompt sections in priority order
        sections = []
        token_count = 0
        
        # Critical context (always included)
        core_prompt = self._build_core_prompt(user_context, task_type)
        core_tokens = self._count_tokens(core_prompt)
        sections.append(core_prompt)
        token_count += core_tokens
        
        # Add prioritized materials until token budget exhausted
        for material_type, content in prioritized_materials:
            content_str = self._format_material(material_type, content)
            content_tokens = self._count_tokens(content_str)
            
            if token_count + content_tokens < self.max_input_tokens:
                sections.append(content_str)
                token_count += content_tokens
            else:
                # Try to include truncated version
                truncated = self._truncate_to_fit(
                    content_str, 
                    self.max_input_tokens - token_count
                )
                if truncated:
                    sections.append(truncated)
                break
        
        return "\n\n".join(sections)
    
    def _prioritize_materials(
        self, 
        materials: Dict[str, Any],
        user_context: UserContext
    ) -> List[tuple]:
        """Order materials by relevance to user context."""
        
        priority_map = {
            'junior': ['quickstart', 'api_surface', 'examples', 'config', 'architecture'],
            'mid': ['api_surface', 'config', 'examples', 'architecture', 'deployment'],
            'senior': ['architecture', 'config', 'deployment', 'api_surface', 'examples']
        }
        
        ordered_keys = priority_map.get(user_context.experience_level, priority_map['mid'])
        
        # Filter out completed steps
        filtered_materials = []
        for key in ordered_keys:
            if key in materials:
                # Skip if user already completed related steps
                if not any(step in key for step in user_context.completed_steps):
                    filtered_materials.append((key, materials[key]))
        
        return filtered_materials
    
    def _build_core_prompt(self, user_context: UserContext, task_type: str) -> str:
        """Build essential prompt instructions."""
        
        experience_instructions = {
            'junior': 'Provide detailed explanations with examples. Explain technical terms.',
            'mid': 'Balance explanation with actionable steps. Assume basic terminology knowledge.',
            'senior': 'Focus on architecture decisions and trade-offs. Minimal hand-holding.'
        }
        
        return f"""You are generating {task_type} for a {user_context.role} with {user_context.experience_level} experience.

Primary Goal: {user_context.primary_goal}
Operating System: {user_context.os}
Already Completed: {', '.join(user_context.completed_steps) or 'None'}
Known Pain Points: {', '.join(user_context.pain_points) or 'None'}

Instructions:
- {experience_instructions[user_context.experience_level]}
- Skip completed steps
- Use {user_context.os}-specific commands
- Address known pain points
- Include copy-pasteable commands
- Explain expected output for verification
"""
    
    def _format_material(self, material_type: str, content: Any) -> str:
        """Format extracted material for LLM consumption."""
        if material_type == 'api_surface':
            return f"## API Reference\n\n```python\n{json.dumps(content, indent=2)}\n```"
        elif material_type == 'config':
            return f"## Configuration Schema\n\n```yaml\n{yaml.dump(content, default_flow_style=False)}\n```"
        else:
            return f"## {material_type.title()}\n\n{json.dumps(content, indent=2)}"
    
    def _count_tokens(self, text: str) -> int:
        """Approximate token count."""
        return len(self.encoder.encode(text))
    
    def _truncate_to_fit(self, text: str, remaining_tokens: int) -> Optional[str]:
        """Intelligently truncate content to fit token budget."""
        if remaining_tokens < 100:  # Minimum useful size
            return None
        
        # Truncate at approximately remaining_tokens
        approx_chars = remaining_tokens * 4  # Rough approximation
        truncated = text[:approx_chars]
        
        # Find last complete section
        last_newline = truncated.rfind('\n\n')
        if last_newline > 0:
            truncated = truncated[:last_newline]
        
        return truncated + "\n\n[... truncated for brevity ...]"
```

**Real Constraints**

Token budget management is critical—exceeding limits causes request failures. Priority ranking requires understanding user context deeply. Multi-turn conversations accumulate context, requiring aggressive pruning strategies.

**Trade-offs**

Including more context improves accuracy but increases cost and latency. Aggressive filtering reduces cost but risks missing crucial information. Dynamic prioritization adds complexity but significantly improves relevance.

### Component 3: Quality Validation

**Technical Explanation**

Generated documentation requires validation before presentation. This includes factual accuracy checking (comparing generated content against source), completeness verification (ensuring all critical topics covered), and technical correctness (validating code examples actually work).

**Practical Implementation**

```python
import subprocess
import tempfile
from typing import List, Dict, Tuple

class DocumentationValidator:
    def __init__(self, source_materials: Dict[str, Any]):
        self.source_materials = source_materials
        
    def validate_generated_docs(self, generated_content: str) -> Dict[str, Any]:
        """Run comprehensive validation checks."""
        
        results = {
            'vali