# Design Thinking: General-Purpose Lesson Generator Agent

## Overview

A unified, configurable agent that generates lessons for any curriculum folder using batch APIs (OpenAI or Claude), eliminating the need for domain-specific scripts.

## Problem Statement

**Current State:**
- Multiple domain-specific scripts (`generate_datagov_lessons_batch.py`, `generate_mdm_lessons_batch_claude.py`, `generate_rpa_lessons_batch_claude.py`, etc.)
- Duplicated code with only minor variations
- Hardcoded paths and domain-specific terminology replacements
- Manual configuration changes required for each new curriculum

**Target State:**
- Single agent script: `generate_lessons_batch.py`
- Takes folder path as argument: `python generate_lessons_batch.py <folder_path>`
- Auto-detects content structure file and output directories
- Configurable via optional config file or CLI arguments
- Supports both OpenAI and Claude batch APIs

## Architecture Design

### Core Components

```
generate_lessons_batch.py (Main Entry Point)
├── CurriculumLoader (Abstract Base)
│   ├── parse_content_structure()
│   └── extract_lessons()
├── BatchAPIProvider (Strategy Pattern)
│   ├── OpenAIBatchProvider
│   └── ClaudeBatchProvider
├── PromptEngine
│   ├── load_template()
│   ├── detect_domain()
│   └── apply_replacements()
└── ConfigManager
    ├── load_config()
    ├── merge_defaults()
    └── validate()
```

### 1. Input Detection & Discovery

**Auto-Discovery Logic:**
```
Input: folder_path (e.g., "src/data/rpa")

1. Content Structure File Detection:
   - Pattern: content_structure_*.md
   - Fallback: Content Structure.md
   - Error if not found

2. Lessons Output Directory:
   - Default: {folder_path}/lessons/
   - Create if missing

3. Config File (Optional):
   - Look for: {folder_path}/.lesson-gen-config.json
   - Or: {folder_path}/lesson_gen_config.json
   - Fallback: Default config

4. Prompt Template:
   - Default: LESSON_GENERATION_PROMPT_GENERIC.md (project root)
   - Override: {folder_path}/prompt_template.md (optional)
```

### 2. Configuration Structure

**Default Config (Built-in):**
```json
{
  "api_provider": "claude",  // "openai" | "claude"
  "model": "claude-haiku-4-5-20251001",  // Auto-set based on provider
  "max_tokens": 16384,
  "start_from_lesson": null,
  "test_mode": false,
  "test_lesson_codes": [],
  "metadata": {
    "audience": "Technical professionals and developers",
    "firm_type": "Organizations implementing solutions",
    "industry": "Technology and digital transformation"
  },
  "prompt_replacements": {},
  "api_keys": {
    "openai": "openaiapikey.txt",
    "claude": "claudeAPIkey.txt"
  }
}
```

**Domain-Specific Config Example** (`src/data/rpa/.lesson-gen-config.json`):
```json
{
  "api_provider": "claude",
  "model": "claude-haiku-4-5-20251001",
  "metadata": {
    "audience": "RPA developers, process automation engineers, and business analysts",
    "firm_type": "Organizations implementing Robotic Process Automation",
    "industry": "Robotic Process Automation, business process automation, digital transformation"
  },
  "prompt_replacements": {
    "an AI-Native SaaS Curriculum": "a Robotic Process Automation (RPA) Curriculum",
    "content_structure_ai-native-saas-curriculum-lesson-maps.md": "content_structure_rpa-curriculum-lesson-maps.md",
    "realistic SaaS architecture patterns": "realistic RPA automation patterns",
    "SaaS-specific terminology: multi-tenancy, tenant isolation, subscription models, usage-based pricing, CI/CD, and observability": "RPA-specific terminology: attended bots, unattended bots, orchestrators, process mining, exception handling, and bot governance",
    "common SaaS stacks — Next.js, Postgres, Vercel, AWS, and Terraform": "common RPA platforms — UiPath, Automation Anywhere, Power Automate, Blue Prism, and open-source alternatives",
    "Languages accepted: TypeScript/Node.js, SQL, Terraform, or YAML": "Languages accepted: RPA platform-specific languages (UiPath XAML, Automation Anywhere AAL), Python, VB.NET, C#, or YAML (for configuration)",
    "tenancy isolation": "bot isolation"
  }
}
```

### 3. API Provider Strategy Pattern

**Abstract Interface:**
```python
class BatchAPIProvider(ABC):
    @abstractmethod
    def create_batch_requests(self, lessons, prompt_template) -> dict
    
    @abstractmethod
    def submit_batch(self, batch_data) -> tuple[str, dict]  # (batch_id, response)
    
    @abstractmethod
    def poll_batch_status(self, batch_id) -> dict
    
    @abstractmethod
    def download_results(self, batch_status, output_file) -> bool
    
    @abstractmethod
    def process_results(self, results_file, lessons_map) -> dict
```

**OpenAI Implementation:**
- Uses OpenAI SDK
- JSONL upload format
- Different result structure

**Claude Implementation:**
- Uses REST API (requests)
- JSON batch format
- Different polling mechanism

### 4. Prompt Engine

**Domain Detection:**
```python
def detect_domain(folder_path: str, content_structure: str) -> str:
    """
    Detect domain from folder name or content structure.
    Returns: 'rpa', 'mdm', 'snowflake_tune', 'data_gov', 'saas', etc.
    """
    # Extract from path: src/data/rpa -> 'rpa'
    folder_name = Path(folder_path).name
    
    # Or infer from content structure metadata
    # Or use config file domain field
```

**Replacement Strategy:**
```python
def apply_replacements(template: str, replacements: dict) -> str:
    """
    Apply string replacements to prompt template.
    Order matters: longest matches first to avoid partial replacements.
    """
    sorted_replacements = sorted(
        replacements.items(),
        key=lambda x: len(x[0]),
        reverse=True
    )
    
    result = template
    for old, new in sorted_replacements:
        result = result.replace(old, new)
    
    return result
```

### 5. Workflow Orchestration

**Main Flow:**
```
1. Parse Arguments
   ├── folder_path (required)
   ├── --api-provider (optional, auto-detect from config)
   ├── --model (optional)
   ├── --start-from (optional)
   └── --test-mode (optional)

2. Discover Resources
   ├── Content structure file
   ├── Config file (if exists)
   ├── Prompt template
   └── Output directory

3. Load Configuration
   ├── Load defaults
   ├── Merge folder config
   ├── Apply CLI overrides
   └── Validate

4. Initialize Provider
   ├── Detect API provider (from config or model name)
   ├── Load API key
   └── Create provider instance

5. Parse Curriculum
   ├── Extract lessons
   ├── Filter by start_from/test_mode
   └── Map lesson metadata

6. Generate Batch
   ├── Prepare prompts for each lesson
   ├── Create batch requests
   └── Validate size constraints

7. Execute Batch
   ├── Submit batch
   ├── Poll for completion
   ├── Download results
   └── Process and save lessons

8. Report Results
   ├── Success/failure counts
   ├── Files created
   └── Batch ID for reference
```

## Design Patterns

### 1. Strategy Pattern
- `BatchAPIProvider` interface with `OpenAIBatchProvider` and `ClaudeBatchProvider` implementations
- Allows easy addition of new providers (e.g., `GeminiBatchProvider`)

### 2. Factory Pattern
- `ProviderFactory.create(provider_name, config)` returns appropriate provider instance

### 3. Template Method Pattern
- Common workflow in `LessonGeneratorAgent.run()` with provider-specific steps delegated

### 4. Configuration Chain
- Default → Folder Config → CLI Args (priority: CLI > Folder > Default)

## Error Handling

**Validation Points:**
1. Folder path exists and is accessible
2. Content structure file found
3. Config file valid JSON (if provided)
4. API key available for selected provider
5. Prompt template exists
6. Batch size within limits
7. At least one lesson to process

**Recovery Strategies:**
- Graceful degradation (use defaults if config missing)
- Partial batch recovery (save completed lessons even if batch fails)
- Resume capability (skip already-generated lessons, configurable)

## CLI Interface

**Usage:**
```bash
# Basic usage (auto-detect everything)
python scripts/generate_lessons_batch.py src/data/rpa

# With API provider override
python scripts/generate_lessons_batch.py src/data/rpa --api-provider openai

# Test mode (2 lessons only)
python scripts/generate_lessons_batch.py src/data/rpa --test-mode --test-lessons M0-L001 M0-L002

# Start from specific lesson
python scripts/generate_lessons_batch.py src/data/rpa --start-from M0-L010

# Override model
python scripts/generate_lessons_batch.py src/data/rpa --model gpt-5-mini-2025-08-07
```

**Arguments:**
```
Required:
  folder_path           Path to curriculum folder (e.g., src/data/rpa)

Optional:
  --api-provider        API provider: 'openai' or 'claude' (default: auto-detect)
  --model               Model name (default: from config)
  --start-from          Start processing from this lesson code (inclusive)
  --test-mode           Enable test mode (process limited lessons)
  --test-lessons        Lesson codes to process in test mode (space-separated)
  --overwrite           Overwrite existing lessons (default: skip)
  --config              Path to config file (default: auto-detect)
  --verbose             Enable verbose logging
```

## File Structure

```
scripts/
├── generate_lessons_batch.py          # Main agent script
├── lesson_generator/
│   ├── __init__.py
│   ├── curriculum_loader.py           # Content structure parsing
│   ├── batch_providers.py             # OpenAI/Claude providers
│   ├── prompt_engine.py               # Template processing
│   ├── config_manager.py              # Configuration handling
│   └── models.py                  # Data models/classes
└── batch_api/                         # Batch API artifacts
    └── {folder_name}_batch_*.json     # Generated files
```

## Benefits

1. **DRY Principle**: Single codebase for all curricula
2. **Maintainability**: Bug fixes and improvements apply to all domains
3. **Extensibility**: Easy to add new providers or domains
4. **Consistency**: Uniform behavior across all lesson generation
5. **Flexibility**: Config-driven customization per domain
6. **Developer Experience**: Simple CLI, clear errors, helpful defaults

## Migration Strategy

1. Create new general-purpose agent
2. Test with one domain (RPA) as reference
3. Migrate existing scripts to use agent (backward compatible wrappers)
4. Deprecate domain-specific scripts
5. Update documentation

## Testing Strategy

**Unit Tests:**
- Config loading and merging
- Content structure parsing
- Prompt replacements
- Provider factory

**Integration Tests:**
- End-to-end with test mode (2-3 lessons)
- Both API providers
- Error handling scenarios

**Validation Tests:**
- Generated lesson structure
- Content quality checks (optional, future)

## Future Enhancements

1. **Multi-provider Support**: Generate with multiple providers and compare
2. **Resume Capability**: Automatic resume of interrupted batches
3. **Progress Tracking**: Real-time progress dashboard
4. **Quality Validation**: Automatic content quality checks
5. **Dry Run Mode**: Validate configuration without submitting batch
6. **Batch Splitting**: Automatically split large batches
7. **Cost Estimation**: Estimate costs before submission
8. **Template Variants**: Support multiple prompt templates per domain

