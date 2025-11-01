# Claude Code for Command-Line Agency

## Core Concepts

### Technical Definition

Command-line agency refers to giving LLM systems the ability to execute shell commands, manipulate files, and interact with the operating system directly through a conversation interface. Instead of being limited to text generation, the model becomes an autonomous agent that can read project structures, modify code files, run tests, install dependencies, and perform system operations—all through natural language instructions.

This represents a fundamental shift from LLMs as *assistants* (generating code you copy-paste) to *agents* (directly executing changes in your environment).

### Engineering Analogy: Traditional vs. Agentic Approach

**Traditional Workflow:**

```python
# You ask LLM: "Write a script to analyze log files"
# LLM responds with code
# You copy it manually
# You create the file
# You run it
# You find bugs
# You ask LLM to fix it
# You copy-paste the fix
# Repeat...

def analyze_logs(log_file: str) -> dict:
    """Manual copy-paste from LLM output"""
    # ... generated code here
    pass

# You manually: vim analyze.py
# You manually: python analyze.py logs/app.log
# You manually: pip install missing-package
# You manually: python analyze.py logs/app.log (again)
```

**Agentic Workflow:**

```python
# You say: "Analyze the log files in logs/ and create a summary report"
# Agent autonomously:
# 1. Lists files in logs/ directory
# 2. Reads sample entries to understand format
# 3. Writes analyze_logs.py
# 4. Detects missing dependencies
# 5. Installs them
# 6. Runs the script
# 7. Generates summary.md with findings
# 8. Reports completion

# All of this happens in one conversation turn
# No copy-paste, no context switching
```

The difference isn't just convenience—it's a **10-30x reduction in iteration cycles** because the agent maintains context, detects errors, and self-corrects without breaking your flow.

### Key Insights That Change Engineering Perspective

1. **Context Persistence**: The agent sees your entire project structure, recent changes, and command outputs. It's not starting from zero each time you ask a question.

2. **Error Recovery**: When a command fails, the agent sees the error message and can immediately try alternative approaches—no need to relay error messages back and forth.

3. **Multi-Step Automation**: What would require scripting or manual orchestration becomes a single natural language request. The agent plans and executes multi-step workflows autonomously.

4. **Cost-Benefit Inversion**: Tasks that were "too small to automate" (one-off refactors, project setup, exploratory analysis) now become worth delegating because the overhead dropped from hours to minutes.

### Why This Matters NOW

**Technical Reasons:**

1. **Token Context Windows** reached 200K+ tokens, enabling agents to see entire codebases
2. **Function Calling APIs** provide reliable tool execution rather than parsing freeform text
3. **Improved Reasoning** (models like Claude 3.5 Sonnet) enables multi-step planning without constant errors

**Practical Impact:**

- **Developer Velocity**: Teams report 2-5x faster completion of maintenance tasks (migrations, refactors, test writing)
- **Lower Expertise Barriers**: Junior engineers can safely perform complex operations with agent verification
- **Reduced Context Switching**: Stay in one interface for planning, coding, executing, and debugging

The constraint is no longer "can the model write good code?" but "can I safely let it execute in my environment?"

## Technical Components

### 1. Tool System Architecture

**Technical Explanation:**

Agentic systems use structured function calling where each tool is defined with:
- Name and description
- Typed parameters with validation
- Execution sandbox or permission model
- Return value schema

```python
from typing import Literal, Optional
import subprocess
import json

class ToolDefinition:
    """Schema for a command-line tool"""
    def __init__(
        self,
        name: str,
        description: str,
        parameters: dict,
        handler: callable
    ):
        self.name = name
        self.description = description
        self.parameters = parameters
        self.handler = handler

def execute_shell_tool(
    command: str,
    working_dir: Optional[str] = None,
    timeout: int = 30
) -> dict:
    """
    Execute shell command with safety constraints
    
    Returns:
        {
            "stdout": str,
            "stderr": str,
            "exit_code": int,
            "error": Optional[str]
        }
    """
    try:
        result = subprocess.run(
            command,
            shell=True,
            cwd=working_dir,
            timeout=timeout,
            capture_output=True,
            text=True
        )
        return {
            "stdout": result.stdout,
            "stderr": result.stderr,
            "exit_code": result.returncode,
            "error": None
        }
    except subprocess.TimeoutExpired:
        return {
            "stdout": "",
            "stderr": "",
            "exit_code": -1,
            "error": f"Command timeout after {timeout}s"
        }
    except Exception as e:
        return {
            "stdout": "",
            "stderr": "",
            "exit_code": -1,
            "error": str(e)
        }

# Tool registry
TOOLS = {
    "execute_shell": ToolDefinition(
        name="execute_shell",
        description="Execute a shell command and return output",
        parameters={
            "command": {"type": "string", "required": True},
            "working_dir": {"type": "string", "required": False},
            "timeout": {"type": "integer", "default": 30}
        },
        handler=execute_shell_tool
    )
}
```

**Practical Implications:**

- The LLM doesn't execute commands directly—it requests tool invocations through structured API calls
- Your code validates and executes the tool, returning results to the LLM
- This separation enables safety checks, logging, and sandboxing

**Real Constraints:**

- **Latency**: Each tool call is a full LLM round-trip (1-5 seconds)
- **Token Cost**: Tool definitions consume context window space
- **Error Handling**: LLMs may not gracefully handle unexpected tool failures

**Concrete Example:**

```python
# LLM generates this structured request:
tool_request = {
    "tool": "execute_shell",
    "parameters": {
        "command": "find . -name '*.py' | head -5",
        "working_dir": "/project"
    }
}

# Your system executes and returns:
tool_response = {
    "stdout": "./main.py\n./src/utils.py\n./src/api.py\n./tests/test_main.py\n./tests/test_utils.py\n",
    "stderr": "",
    "exit_code": 0
}

# LLM sees the result and can make informed next decision
```

### 2. File System Operations

**Technical Explanation:**

File operations require specialized handling because they're stateful and destructive:

```python
import os
from pathlib import Path
from typing import Optional

class FileOperations:
    """Safe file system operations for agents"""
    
    def __init__(self, base_path: str, max_file_size: int = 1_000_000):
        self.base_path = Path(base_path).resolve()
        self.max_file_size = max_file_size
    
    def _validate_path(self, file_path: str) -> Path:
        """Ensure path is within base directory"""
        full_path = (self.base_path / file_path).resolve()
        if not str(full_path).startswith(str(self.base_path)):
            raise ValueError(f"Path {file_path} escapes base directory")
        return full_path
    
    def read_file(self, file_path: str) -> dict:
        """Read file contents with size limits"""
        try:
            full_path = self._validate_path(file_path)
            
            if not full_path.exists():
                return {"error": f"File not found: {file_path}"}
            
            size = full_path.stat().st_size
            if size > self.max_file_size:
                return {
                    "error": f"File too large: {size} bytes (max {self.max_file_size})"
                }
            
            content = full_path.read_text(encoding='utf-8')
            return {
                "content": content,
                "size": size,
                "error": None
            }
        except UnicodeDecodeError:
            return {"error": "File is not valid UTF-8 text"}
        except Exception as e:
            return {"error": str(e)}
    
    def write_file(
        self,
        file_path: str,
        content: str,
        create_dirs: bool = True
    ) -> dict:
        """Write file with safety checks"""
        try:
            full_path = self._validate_path(file_path)
            
            if create_dirs:
                full_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Create backup if file exists
            backup_path = None
            if full_path.exists():
                backup_path = full_path.with_suffix(full_path.suffix + '.backup')
                full_path.rename(backup_path)
            
            full_path.write_text(content, encoding='utf-8')
            
            # Remove backup on success
            if backup_path and backup_path.exists():
                backup_path.unlink()
            
            return {
                "success": True,
                "bytes_written": len(content.encode('utf-8')),
                "error": None
            }
        except Exception as e:
            # Restore backup on failure
            if backup_path and backup_path.exists():
                backup_path.rename(full_path)
            return {"success": False, "error": str(e)}
    
    def list_directory(
        self,
        dir_path: str = ".",
        pattern: str = "*"
    ) -> dict:
        """List directory contents with filtering"""
        try:
            full_path = self._validate_path(dir_path)
            
            if not full_path.is_dir():
                return {"error": f"Not a directory: {dir_path}"}
            
            entries = []
            for item in full_path.glob(pattern):
                relative = item.relative_to(self.base_path)
                entries.append({
                    "path": str(relative),
                    "type": "dir" if item.is_dir() else "file",
                    "size": item.stat().st_size if item.is_file() else 0
                })
            
            return {"entries": entries, "error": None}
        except Exception as e:
            return {"error": str(e)}
```

**Practical Implications:**

- Always sandbox to a specific directory
- Implement automatic backups for destructive operations
- Limit file sizes to prevent context window exhaustion
- Use relative paths to prevent path traversal attacks

**Real Constraints:**

- **Binary Files**: Can't read non-text files (images, compiled binaries)
- **Large Files**: Reading 10MB+ files consumes excessive tokens
- **Concurrency**: Multiple agents writing same files causes conflicts

### 3. Context Management & Memory

**Technical Explanation:**

Agents need to maintain awareness across multi-step operations:

```python
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Any

@dataclass
class ConversationContext:
    """Track agent's working memory"""
    project_root: str
    current_directory: str
    command_history: List[Dict[str, Any]] = field(default_factory=list)
    file_cache: Dict[str, str] = field(default_factory=dict)
    goals: List[str] = field(default_factory=list)
    max_history: int = 20
    
    def add_command(
        self,
        command: str,
        output: str,
        exit_code: int
    ) -> None:
        """Record command execution"""
        self.command_history.append({
            "timestamp": datetime.now().isoformat(),
            "command": command,
            "output": output[:1000],  # Truncate long outputs
            "exit_code": exit_code
        })
        
        # Keep only recent history
        if len(self.command_history) > self.max_history:
            self.command_history = self.command_history[-self.max_history:]
    
    def cache_file(self, path: str, content: str) -> None:
        """Cache frequently accessed files"""
        # Simple LRU: keep only 10 most recent files
        if len(self.file_cache) >= 10:
            first_key = next(iter(self.file_cache))
            del self.file_cache[first_key]
        self.file_cache[path] = content
    
    def get_context_summary(self) -> str:
        """Generate context summary for LLM"""
        summary = [
            f"Project Root: {self.project_root}",
            f"Current Directory: {self.current_directory}",
            f"\nRecent Commands ({len(self.command_history)}):"
        ]
        
        for cmd in self.command_history[-5:]:
            summary.append(
                f"  $ {cmd['command']} (exit: {cmd['exit_code']})"
            )
        
        if self.goals:
            summary.append("\nActive Goals:")
            for goal in self.goals:
                summary.append(f"  - {goal}")
        
        return "\n".join(summary)
```

**Practical Implications:**

- Include context summary in each LLM prompt to maintain continuity
- Cache small frequently-accessed files to reduce redundant reads
- Track command history to enable "what did we just try?" reasoning

**Real Constraints:**

- Context summaries consume tokens (typically 500-2000 tokens)
- Stale cache can mislead agent if files change externally
- Long conversations (50+ turns) degrade coherence even with context

### 4. Safety & Permissions Model

**Technical Explanation:**

Production agents require explicit permission controls:

```python
from enum import Enum
from typing import Set, Callable
import re

class PermissionLevel(Enum):
    READ_ONLY = "read_only"
    SAFE_WRITE = "safe_write"  # Create/modify files, safe commands
    FULL_ACCESS = "full_access"  # System commands, deletions

class SafetyGuard:
    """Enforce permission boundaries for agent actions"""
    
    DANGEROUS_COMMANDS = {
        r'\brm\b.*-rf',  # rm -rf
        r'\bsudo\b',
        r'\bchmod\b.*777',
        r'>\s*/dev/',
        r'\bcurl\b.*\|\s*bash',  # Pipe to bash
        r'\bwget\b.*\|\s*sh',
    }
    
    SAFE_WRITE_PATTERNS = {
        r'\.py$', r'\.js$', r'\.md$', r'\.txt$', r'\.json$',
        r'\.yaml$', r'\.toml$', r'\.csv$'
    }
    
    def __init__(self, permission_level: PermissionLevel):
        self.permission_level = permission_level
    
    def check_command(self, command: str) -> tuple[bool, str]:
        """Validate command is allowed"""
        if self.permission_level == PermissionLevel.FULL_ACCESS:
            return True, "Allowed"
        
        # Check for dangerous patterns
        for pattern in self.DANGEROUS_COMMANDS:
            if re.search(pattern, command, re.IGNORECASE):
                return False, f"Dangerous command blocked: {pattern}"
        
        # Safe writes can run read operations and safe utilities
        if self.permission_level == PermissionLevel.SAFE_WRITE:
            safe_commands = ['ls', 'cat', 'grep', 'find', 'echo', 
                           'mkdir', 'touch', 'python', 'node', 'git']
            first_word = command.split()[0] if command.split() else ""
            if any(first_word.startswith(cmd) for cmd in safe_commands):
                return True, "Safe command"
            return False, "Command not