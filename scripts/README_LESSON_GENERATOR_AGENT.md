# General-Purpose Lesson Generator Agent

A unified, configurable agent that generates lessons for any curriculum folder using batch APIs (OpenAI or Claude).

## Features

- ✅ **Universal**: Works with any curriculum folder
- ✅ **Auto-Detection**: Automatically finds content structure files and configs
- ✅ **Multi-Provider**: Supports both OpenAI and Claude batch APIs
- ✅ **Config-Driven**: Domain-specific customization via JSON config files
- ✅ **Test Mode**: Validate with limited lessons before full runs
- ✅ **Resume Capability**: Skips already-generated lessons by default

## Quick Start

```bash
# Basic usage (auto-detects everything)
python scripts/generate_lessons_batch.py src/data/rpa

# Test mode (2 lessons only)
python scripts/generate_lessons_batch.py src/data/rpa --test-mode

# Test specific lessons
python scripts/generate_lessons_batch.py src/data/rpa --test-mode --test-lessons M0-L001 M0-L002

# Override API provider
python scripts/generate_lessons_batch.py src/data/rpa --api-provider openai

# Start from specific lesson
python scripts/generate_lessons_batch.py src/data/rpa --start-from M0-L010

# Overwrite existing lessons
python scripts/generate_lessons_batch.py src/data/rpa --overwrite
```

## Configuration

### Default Config

Located at `lesson_generator_config.default.json` - contains shared defaults:

```json
{
  "api_provider": "claude",
  "model": "claude-haiku-4-5-20251001",
  "max_tokens": 16384,
  "metadata": {
    "audience": "Technical professionals and developers",
    "firm_type": "Organizations implementing solutions",
    "industry": "Technology and digital transformation"
  }
}
```

### Domain-Specific Config

Place `.lesson-gen-config.json` in your curriculum folder (e.g., `src/data/rpa/.lesson-gen-config.json`):

```json
{
  "api_provider": "claude",
  "metadata": {
    "audience": "RPA developers, process automation engineers",
    "firm_type": "Organizations implementing RPA",
    "industry": "Robotic Process Automation"
  },
  "prompt_replacements": {
    "an AI-Native SaaS Curriculum": "a Robotic Process Automation (RPA) Curriculum",
    "SaaS-specific terminology": "RPA-specific terminology: attended bots, unattended bots..."
  }
}
```

**Config Priority**: CLI Arguments > Folder Config > Default Config

## Command-Line Options

```
Required:
  folder_path           Path to curriculum folder (e.g., src/data/rpa)

Optional:
  --api-provider        API provider: 'openai' or 'claude
  --model               Model name (overrides config)
  --start-from        Start processing from this lesson code
  --test-mode         Enable test mode (process limited lessons)
  --test-lessons      Lesson codes to process in test mode
  --overwrite         Overwrite existing lessons (default: skip)
  --config            Path to custom config file
  --verbose           Enable verbose logging
```

## File Structure

The agent expects the following structure:

```
src/data/
└── {curriculum_name}/
    ├── content_structure_*.md    # Auto-detected
    ├── .lesson-gen-config.json    # Optional
    └── lessons/                   # Created automatically
        └── M0-L001.md            # Generated lessons
```

## Auto-Detection

The agent automatically detects:

1. **Content Structure File**: 
   - Looks for `content_structure_*.md`
   - Falls back to `Content Structure.md`

2. **Config File**:
   - Looks for `.lesson-gen-config.json` in folder
   - Falls back to `lesson_gen_config.json`
   - Merges with default config

3. **API Provider**:
   - From config file `api_provider` field
   - Auto-detected from model name if not specified
   - Can be overridden with `--api-provider`

## Examples

### RPA Curriculum

```bash
# Generate all remaining lessons
python scripts/generate_lessons_batch.py src/data/rpa

# Test first 2 lessons
python scripts/generate_lessons_batch.py src/data/rpa --test-mode
```

### MDM Curriculum

```bash
# Generate with OpenAI
python scripts/generate_lessons_batch.py src/data/mdm --api-provider openai

# Start from lesson M0-L050
python scripts/generate_lessons_batch.py src/data/mdm --start-from M0-L050
```

### New Curriculum

1. Create folder: `src/data/my_curriculum/`
2. Add content structure: `content_structure_my-curriculum.md`
3. (Optional) Add config: `.lesson-gen-config.json`
4. Run: `python scripts/generate_lessons_batch.py src/data/my_curriculum`

## Architecture

The agent uses several design patterns:

- **Strategy Pattern**: `BatchAPIProvider` interface with `OpenAIBatchProvider` and `ClaudeBatchProvider` implementations
- **Factory Pattern**: Automatic provider selection based on config
- **Configuration Chain**: Default → Folder → CLI (priority order)

## Migration from Domain-Specific Scripts

Old scripts like `generate_rpa_lessons_batch_claude.py` are now replaced by this agent. 

To migrate:
1. Use the new agent instead
2. Domain-specific configs (`.lesson-gen-config.json`) provide the same customization
3. CLI arguments provide runtime flexibility

## Troubleshooting

**"Content structure file not found"**
- Ensure your folder contains `content_structure_*.md` or `Content Structure.md`

**"API key not found"**
- Check `claudeAPIkey.txt` or `openaiapikey.txt` in project root
- Or set `ANTHROPIC_API_KEY` or `OPENAI_API_KEY` environment variable

**"No lessons to process"**
- All lessons already generated (use `--overwrite` to regenerate)
- Test mode selected lessons don't exist
- Start-from lesson not found

**"Batch size exceeds limit"**
- Split into multiple batches using `--start-from` to process in chunks

## Requirements

- Python 3.11+
- `openai` library (for OpenAI provider)
- `requests` library (for Claude provider)

Install dependencies:
```bash
pip install openai requests
```

