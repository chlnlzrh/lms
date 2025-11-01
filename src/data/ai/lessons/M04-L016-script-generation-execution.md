# Script Generation & Execution with LLMs

## Core Concepts

**Technical Definition:** Script generation with LLMs is the process of using language models to produce executable code based on natural language specifications, followed by programmatic execution of that code within a controlled environment. Unlike traditional code generation tools that rely on templates or AST manipulation, LLM-based script generation treats code as structured text, leveraging statistical patterns learned from vast code repositories to produce syntactically valid and contextually appropriate programs.

### Engineering Analogy: Template Systems vs. LLM Generation

Traditional approaches to automated script generation rely on predefined templates with parameter substitution:

```python
# Traditional: Template-based script generation
class TraditionalScriptGenerator:
    def __init__(self):
        self.templates = {
            'file_processor': '''
import os
def process_files(directory: str, extension: str):
    for filename in os.listdir(directory):
        if filename.endswith(extension):
            # Process {action}
            pass
''',
            'api_caller': '''
import requests
def call_api(url: str, params: dict):
    response = requests.get(url, params=params)
    return response.json()
'''
        }
    
    def generate(self, template_name: str, **kwargs) -> str:
        template = self.templates.get(template_name, "")
        return template.format(**kwargs)

# Usage is rigid and limited
generator = TraditionalScriptGenerator()
script = generator.generate('file_processor', action='print filename')
# Can only generate predefined patterns
```

LLM-based generation treats the entire specification as context:

```python
# Modern: LLM-based script generation
import anthropic
import subprocess
import tempfile
from typing import Optional, Dict, Any
from pathlib import Path

class LLMScriptGenerator:
    def __init__(self, api_key: str):
        self.client = anthropic.Anthropic(api_key=api_key)
    
    def generate(self, specification: str, language: str = "python") -> str:
        """Generate executable script from natural language specification."""
        prompt = f"""Generate a complete, production-ready {language} script that:
{specification}

Requirements:
- Include all necessary imports
- Add type hints where applicable
- Include basic error handling
- Add docstrings for functions
- Make the script executable with proper main guard

Output ONLY the code, no explanations."""

        message = self.client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=4096,
            messages=[{"role": "user", "content": prompt}]
        )
        
        return message.content[0].text
    
    def execute_safely(
        self, 
        code: str, 
        timeout: int = 30,
        allowed_imports: Optional[set] = None
    ) -> Dict[str, Any]:
        """Execute generated code in isolated environment with safety checks."""
        if allowed_imports:
            # Basic static analysis for import validation
            imports = self._extract_imports(code)
            forbidden = imports - allowed_imports
            if forbidden:
                return {
                    "success": False,
                    "error": f"Forbidden imports: {forbidden}",
                    "stdout": "",
                    "stderr": ""
                }
        
        # Execute in temporary file for better isolation
        with tempfile.NamedTemporaryFile(
            mode='w', 
            suffix='.py', 
            delete=False
        ) as f:
            f.write(code)
            temp_path = f.name
        
        try:
            result = subprocess.run(
                ['python', temp_path],
                capture_output=True,
                text=True,
                timeout=timeout
            )
            return {
                "success": result.returncode == 0,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "returncode": result.returncode
            }
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "error": f"Execution exceeded {timeout}s timeout",
                "stdout": "",
                "stderr": ""
            }
        finally:
            Path(temp_path).unlink(missing_ok=True)
    
    def _extract_imports(self, code: str) -> set:
        """Extract import statements for validation."""
        imports = set()
        for line in code.split('\n'):
            line = line.strip()
            if line.startswith('import '):
                imports.add(line.split()[1].split('.')[0])
            elif line.startswith('from '):
                imports.add(line.split()[1].split('.')[0])
        return imports

# Usage is flexible and adaptive
generator = LLMScriptGenerator(api_key="your-key")
script = generator.generate("""
Process all CSV files in a directory, calculate the sum of numeric columns,
and output results to a JSON file. Handle missing values gracefully.
""")

result = generator.execute_safely(
    script, 
    allowed_imports={'csv', 'json', 'pathlib', 'typing'}
)
```

### Key Insights

**Code as Structured Text:** LLMs don't "understand" code through formal grammars—they pattern-match on billions of examples. This means they excel at common patterns but can produce syntactically valid yet logically flawed code for edge cases. Always validate generated code through execution and testing, not just inspection.

**The Execution Gap:** Generation is only half the solution. The critical engineering challenge is safe, controlled execution. Production systems must treat generated code as untrusted input, requiring sandboxing, resource limits, and validation layers.

**Specification Quality Determines Output Quality:** Unlike traditional compilers with formal specifications, LLMs are sensitive to ambiguity. A vague request produces vague code. Engineers must develop skills in writing precise, executable specifications—effectively a new form of programming.

### Why This Matters Now

Traditional automation required either: (1) hand-coding every variation, or (2) building complex DSLs and compilers. LLM-based script generation collapses the development cycle for one-off tasks, data transformations, and glue code from hours to minutes. However, this power creates new risks: production systems can now generate and execute arbitrary code at runtime, requiring sophisticated guardrails that most engineering teams haven't built. The organizations that master safe, controlled script generation gain significant velocity advantages in data processing, automation, and integration tasks.

## Technical Components

### 1. Prompt Engineering for Code Generation

Code generation prompts require different structure than general text prompts. Successful prompts specify interfaces, constraints, and validation criteria explicitly.

**Technical Explanation:**

The prompt must establish the contract between generator and executor. This includes input/output specifications, error handling requirements, performance constraints, and execution environment assumptions. Unlike human developers who infer context, LLMs need explicit specification of what might otherwise be "obvious."

```python
from typing import Literal, Optional
from dataclasses import dataclass

@dataclass
class CodeGenerationPrompt:
    """Structured prompt for code generation."""
    task_description: str
    input_format: str
    output_format: str
    language: str = "python"
    style_requirements: Optional[str] = None
    constraints: Optional[list[str]] = None
    example_input: Optional[str] = None
    example_output: Optional[str] = None
    
    def build(self) -> str:
        """Construct optimized prompt for code generation."""
        sections = [
            f"Generate a {self.language} script that:\n{self.task_description}",
            f"\nInput format:\n{self.input_format}",
            f"\nOutput format:\n{self.output_format}"
        ]
        
        if self.constraints:
            constraints_text = "\n".join(f"- {c}" for c in self.constraints)
            sections.append(f"\nConstraints:\n{constraints_text}")
        
        if self.style_requirements:
            sections.append(f"\nCode style:\n{self.style_requirements}")
        
        if self.example_input and self.example_output:
            sections.append(
                f"\nExample:\nInput: {self.example_input}\n"
                f"Output: {self.example_output}"
            )
        
        sections.append(
            "\nRequirements:\n"
            "- Complete, executable code only\n"
            "- Include all imports\n"
            "- Add type hints\n"
            "- Handle errors gracefully\n"
            "- No explanatory text outside code"
        )
        
        return "\n".join(sections)

# Example usage
prompt = CodeGenerationPrompt(
    task_description="Parse log files and extract error messages with timestamps",
    input_format="Log file path as command-line argument",
    output_format="JSON array of {timestamp, level, message} objects to stdout",
    constraints=[
        "Support log files up to 1GB",
        "Memory usage must stay under 100MB",
        "Handle malformed lines gracefully"
    ],
    example_input="/var/log/app.log",
    example_output='[{"timestamp": "2024-01-15T10:30:00", "level": "ERROR", "message": "Connection failed"}]'
)

generated_prompt = prompt.build()
```

**Practical Implications:**

Well-structured prompts reduce generation failures by 60-80% compared to informal requests. The investment in prompt engineering pays off when generating multiple scripts or building reusable generation pipelines.

**Trade-offs:**

More detailed prompts increase token usage (3-5x longer) and generation time. For simple one-off scripts, informal prompts may suffice. For production pipelines or complex requirements, structured prompts are essential.

### 2. Code Extraction and Validation

LLMs often return code wrapped in markdown or mixed with explanatory text. Robust systems must extract clean code and validate it before execution.

```python
import re
import ast
from typing import Tuple, Optional

class CodeExtractor:
    """Extract and validate code from LLM responses."""
    
    @staticmethod
    def extract_code_blocks(response: str, language: str = "python") -> list[str]:
        """Extract code from markdown code blocks."""
        # Pattern for fenced code blocks with optional language
        pattern = rf'```(?:{language})?\n(.*?)```'
        blocks = re.findall(pattern, response, re.DOTALL)
        
        if not blocks:
            # Fallback: try to find code without fences
            # Look for common code patterns
            lines = response.split('\n')
            code_lines = []
            in_code = False
            
            for line in lines:
                # Heuristic: lines starting with import/def/class are likely code
                if any(line.strip().startswith(kw) for kw in ['import', 'from', 'def', 'class', '@']):
                    in_code = True
                
                if in_code:
                    code_lines.append(line)
                    # Stop at blank lines after code starts
                    if not line.strip() and code_lines:
                        break
            
            if code_lines:
                blocks = ['\n'.join(code_lines)]
        
        return blocks
    
    @staticmethod
    def validate_python_syntax(code: str) -> Tuple[bool, Optional[str]]:
        """Validate Python code syntax without executing."""
        try:
            ast.parse(code)
            return True, None
        except SyntaxError as e:
            return False, f"Syntax error at line {e.lineno}: {e.msg}"
    
    @staticmethod
    def extract_and_validate(response: str) -> Tuple[Optional[str], Optional[str]]:
        """Extract code and validate, return first valid block."""
        blocks = CodeExtractor.extract_code_blocks(response)
        
        if not blocks:
            return None, "No code blocks found in response"
        
        for block in blocks:
            is_valid, error = CodeExtractor.validate_python_syntax(block)
            if is_valid:
                return block, None
            
        return None, f"No valid code blocks found. Last error: {error}"

# Example usage
extractor = CodeExtractor()

# Typical LLM response with markdown
response = """
Here's a script that processes CSV files:

```python
import csv
import json

def process_csv(filepath: str) -> dict:
    with open(filepath, 'r') as f:
        reader = csv.DictReader(f)
        return list(reader)

if __name__ == '__main__':
    print(json.dumps(process_csv('data.csv')))
```

This script reads a CSV and outputs JSON.
"""

code, error = extractor.extract_and_validate(response)
if code:
    print("Extracted valid code:")
    print(code)
else:
    print(f"Extraction failed: {error}")
```

**Real Constraints:**

AST parsing catches syntax errors but not runtime issues like missing imports, type mismatches, or logical errors. Validation must be layered: syntax → static analysis → sandboxed execution → output verification.

### 3. Sandboxed Execution Environment

Production systems must execute generated code in isolated environments with resource limits and restricted capabilities.

```python
import docker
import json
from typing import Dict, Any, Optional
from pathlib import Path
import tempfile

class DockerSandbox:
    """Execute code in isolated Docker container."""
    
    def __init__(
        self,
        image: str = "python:3.11-slim",
        timeout: int = 30,
        memory_limit: str = "128m",
        cpu_period: int = 100000,
        cpu_quota: int = 50000  # 50% of one CPU
    ):
        self.client = docker.from_env()
        self.image = image
        self.timeout = timeout
        self.memory_limit = memory_limit
        self.cpu_period = cpu_period
        self.cpu_quota = cpu_quota
        
        # Ensure image is available
        try:
            self.client.images.get(self.image)
        except docker.errors.ImageNotFound:
            print(f"Pulling {self.image}...")
            self.client.images.pull(self.image)
    
    def execute(
        self,
        code: str,
        input_data: Optional[str] = None,
        allowed_packages: Optional[list[str]] = None
    ) -> Dict[str, Any]:
        """Execute code in sandboxed container."""
        
        # Create temporary directory for code and data
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            
            # Write code to file
            code_file = tmppath / "script.py"
            code_file.write_text(code)
            
            # Write input data if provided
            if input_data:
                input_file = tmppath / "input.txt"
                input_file.write_text(input_data)
            
            # Install allowed packages if specified
            setup_commands = []
            if allowed_packages:
                packages = " ".join(allowed_packages)
                setup_commands.append(f"pip install --quiet {packages} &&")
            
            command = " ".join(setup_commands) + " python /workspace/script.py"
            
            try:
                container = self.client.containers.run(
                    self.image,
                    command=f"sh -c '{command}'",
                    volumes={str(tmppath): {'bind': '/workspace', 'mode': 'rw'}},
                    working_dir='/workspace',
                    mem_limit=self.memory_limit,
                    cpu_period=self.cpu_period,
                    cpu_quota=self.cpu_quota,
                    network_mode='none',  # No network access
                    detach=True,
                    remove=False
                )
                
                # Wait for execution with timeout
                result = container.wait(timeout=self.timeout)
                stdout = container.logs(stdout=True, stderr=False).decode('utf-8')
                stderr = container.logs(stdout=False, stderr=True).decode('utf-8')
                
                container.remove()
                
                return {
                    "success": result['StatusCode'] == 0,
                    "stdout": stdout,
                    "stderr": stderr,
                    "exit_code": result['StatusCode']
                }
                
            except docker.errors.ContainerError as e:
                return {
                    "success": False,
                    "stdout": "",
                    "stderr": str(e),
                    "exit_code": e.exit_status
                }
            except Exception as e:
                return {
                