#!/usr/bin/env python3
"""
General-Purpose Lesson Generator Agent

Generates lessons for any curriculum folder using batch APIs (OpenAI or Claude).

Usage:
    python scripts/generate_lessons_batch.py <folder_path> [options]

Examples:
    # Basic usage (auto-detects everything)
    python scripts/generate_lessons_batch.py src/data/rpa

    # With API provider override
    python scripts/generate_lessons_batch.py src/data/rpa --api-provider openai

    # Test mode (first 2 lessons)
    python scripts/generate_lessons_batch.py src/data/rpa --test-mode

    # Test specific lessons
    python scripts/generate_lessons_batch.py src/data/rpa --test-mode --test-lessons M0-L001 M0-L002

    # Start from specific lesson
    python scripts/generate_lessons_batch.py src/data/rpa --start-from M0-L010
"""
import os
import sys
import re
import json
import time
import argparse
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Fix Windows console encoding for Unicode
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

# Try importing optional dependencies
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False


# ============================================================================
# Configuration Management
# ============================================================================

class ConfigManager:
    """Manages configuration with priority: CLI > Folder Config > Defaults."""
    
    DEFAULT_CONFIG_PATH = "lesson_generator_config.default.json"
    
    def __init__(self, folder_path: str, cli_args: dict):
        self.folder_path = Path(folder_path)
        self.cli_args = cli_args
        self.config = self._load_config()
    
    def _load_config(self) -> dict:
        """Load and merge configurations."""
        # Load defaults
        default_config = self._load_json(self.DEFAULT_CONFIG_PATH, {})
        
        # Load folder config if exists
        folder_config_paths = [
            self.folder_path / ".lesson-gen-config.json",
            self.folder_path / "lesson_gen_config.json"
        ]
        folder_config = {}
        for path in folder_config_paths:
            if path.exists():
                folder_config = self._load_json(str(path), {})
                break
        
        # Merge: defaults -> folder -> CLI
        merged = {**default_config}
        merged.update(folder_config)
        merged.update(self.cli_args)
        
        # Validate
        self._validate_config(merged)
        
        return merged
    
    def _load_json(self, path: str, default: dict) -> dict:
        """Load JSON file safely."""
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            return default
        except json.JSONDecodeError as e:
            print(f"⚠ Warning: Invalid JSON in {path}: {e}")
            return default
    
    def _validate_config(self, config: dict):
        """Validate configuration."""
        if config.get('api_provider') not in ['openai', 'claude']:
            raise ValueError(f"Invalid api_provider: {config.get('api_provider')}")
        
        # Auto-detect provider from model name if not set
        if not config.get('api_provider'):
            model = config.get('model', '').lower()
            if 'claude' in model or 'haiku' in model:
                config['api_provider'] = 'claude'
            elif 'gpt' in model:
                config['api_provider'] = 'openai'
            else:
                raise ValueError("Cannot auto-detect API provider. Please specify --api-provider")
    
    def get(self, key: str, default=None):
        """Get config value."""
        return self.config.get(key, default)
    
    def __getitem__(self, key: str):
        """Get config value with [] syntax."""
        return self.config[key]


# ============================================================================
# Curriculum Loading
# ============================================================================

class CurriculumLoader:
    """Parses content structure files and extracts lesson metadata."""
    
    def __init__(self, folder_path: Path):
        self.folder_path = folder_path
        self.content_structure_path = self._find_content_structure()
    
    def _find_content_structure(self) -> Path:
        """Auto-detect content structure file."""
        # Try multiple patterns
        patterns = [
            "content_structure_*.md",
            "Content Structure.md",
            "content_structure*.md"
        ]
        
        for pattern in patterns:
            matches = list(self.folder_path.glob(pattern))
            if matches:
                return matches[0]
        
        raise FileNotFoundError(
            f"Content structure file not found in {self.folder_path}. "
            f"Expected: content_structure_*.md or Content Structure.md"
        )
    
    def parse(self) -> Tuple[List[dict], Dict[str, dict]]:
        """Parse content structure and return lessons and modules."""
        content = self.content_structure_path.read_text(encoding='utf-8')
        
        # Find all module headers: # M0:, # M1:, etc.
        module_pattern = r'^# (M\d+):\s+(.+?)\s+—\s+Lesson Map'
        
        # Find all lesson headers: ## L001: Title [F], etc.
        lesson_pattern = r'^## (L\d{3}):\s+(.+?)\s+\[([FIEA])\]'
        
        modules = {}
        current_module = None
        
        for line in content.split('\n'):
            # Check for module header
            module_match = re.match(module_pattern, line)
            if module_match:
                module_code = module_match.group(1)
                module_title = module_match.group(2)
                current_module = module_code
                modules[current_module] = {
                    'code': module_code,
                    'title': module_title,
                    'lessons': []
                }
                continue
            
            # Check for lesson header (only if we have a current module)
            if current_module:
                lesson_match = re.match(lesson_pattern, line)
                if lesson_match:
                    lesson_code = lesson_match.group(1)
                    lesson_title = lesson_match.group(2).strip()
                    complexity = lesson_match.group(3)
                    
                    full_lesson_code = f"{current_module}-{lesson_code}"
                    
                    lesson_data = {
                        'LESSON_CODE': full_lesson_code,
                        'MODULE_CODE': current_module,
                        'LESSON_TITLE': lesson_title,
                        'COMPLEXITY': complexity,
                        'MODULE_NAME': modules[current_module]['title']
                    }
                    modules[current_module]['lessons'].append(lesson_data)
        
        # Flatten to list of all lessons
        all_lessons = []
        for module_code in sorted(modules.keys(), key=lambda x: int(x[1:]) if x[1:].isdigit() else 999):
            all_lessons.extend(modules[module_code]['lessons'])
        
        return all_lessons, modules


# ============================================================================
# Prompt Engine
# ============================================================================

class PromptEngine:
    """Handles prompt template loading, formatting, and domain-specific replacements."""
    
    def __init__(self, config: ConfigManager, content_structure_path: Path):
        self.config = config
        self.content_structure_path = content_structure_path
        self.template_path = self._find_template()
    
    def _find_template(self) -> Path:
        """Find prompt template file."""
        # Try folder-specific first
        folder_template = self.config.folder_path / "prompt_template.md"
        if folder_template.exists():
            return folder_template
        
        # Try default location
        default_template = Path(self.config.get('paths', {}).get('prompt_template', 'LESSON_GENERATION_PROMPT_GENERIC.md'))
        if default_template.exists():
            return default_template
        
        raise FileNotFoundError(f"Prompt template not found. Expected: {default_template}")
    
    def prepare_prompt(self, lesson_details: dict) -> str:
        """Prepare the complete prompt for a lesson."""
        # Load template
        template_content = self.template_path.read_text(encoding='utf-8').strip()
        
        # Extract f-string content from template
        if template_content.startswith('prompt = f"""'):
            template_content = template_content[13:]
        if template_content.endswith('"""'):
            template_content = template_content[:-3]
        template_content = template_content.strip()
        
        # Get metadata from config
        metadata = self.config.get('metadata', {})
        DEFAULT_AUDIENCE = metadata.get('audience', 'Technical professionals and developers')
        DEFAULT_FIRM_TYPE = metadata.get('firm_type', 'Organizations implementing solutions')
        DEFAULT_INDUSTRY = metadata.get('industry', 'Technology and digital transformation')
        
        # Prepare variables
        complexity_label = {'F': 'Foundation', 'I': 'Intermediate', 'A': 'Advanced', 'E': 'Expert'}.get(
            lesson_details['COMPLEXITY'], 'Foundation'
        )
        module_code = lesson_details.get('MODULE_CODE', 'M0')
        module_name = lesson_details.get('MODULE_NAME', 'Unknown Module')
        
        # Estimate duration based on complexity
        duration_map = {'F': '45', 'I': '60', 'A': '75', 'E': '90'}
        duration = duration_map.get(lesson_details['COMPLEXITY'], '60')
        
        # Replace template variables
        replacements = {
            "{lesson_details['LESSON_CODE']}": lesson_details['LESSON_CODE'],
            "{lesson_details['LESSON_TITLE']}": lesson_details['LESSON_TITLE'],
            "{lesson_details.get('MODULE_CODE', 'M0')}": module_code,
            "{lesson_details.get('MODULE_NAME', MODULE_METADATA.get(lesson_details.get('MODULE_CODE', 'M0'), {}).get('MODULE_NAME', 'Unknown Module'))}": module_name,
            "{lesson_details.get('SPECIFIC_FOCUS', 'General')}": lesson_details.get('SPECIFIC_FOCUS', 'General'),
            "{'Foundation' if lesson_details['COMPLEXITY'] == 'F' else 'Intermediate' if lesson_details['COMPLEXITY'] == 'I' else 'Advanced' if lesson_details['COMPLEXITY'] == 'A' else 'Expert'}": complexity_label,
            "{lesson_details['COMPLEXITY']}": lesson_details['COMPLEXITY'],
            "{lesson_details['TIME']}": duration,
            "{MODULE_METADATA.get(lesson_details.get('MODULE_CODE', 'M0'), {}).get('AUDIENCE_DESCRIPTION', DEFAULT_AUDIENCE)}": DEFAULT_AUDIENCE,
            "{DEFAULT_FIRM_TYPE}": DEFAULT_FIRM_TYPE,
            "{lesson_details['LIST_PREREQUISITES']}": lesson_details.get('LIST_PREREQUISITES', 'Basic understanding of the domain'),
            "{lesson_details['RELATED_LESSON_CODES']}": lesson_details.get('RELATED_LESSON_CODES', 'Related lessons in the same module'),
            "{MODULE_METADATA.get(lesson_details.get('MODULE_CODE', 'M0'), {}).get('INDUSTRY_DOMAIN', DEFAULT_INDUSTRY)}": DEFAULT_INDUSTRY,
        }
        
        formatted_prompt = template_content
        for old, new in replacements.items():
            formatted_prompt = formatted_prompt.replace(old, new)
        
        # Apply domain-specific replacements
        prompt_replacements = self.config.get('prompt_replacements', {})
        # Sort by length (longest first) to avoid partial replacements
        sorted_replacements = sorted(prompt_replacements.items(), key=lambda x: len(x[0]), reverse=True)
        for old, new in sorted_replacements:
            formatted_prompt = formatted_prompt.replace(old, new)
        
        # Read content structure preview
        content_structure_text = self.content_structure_path.read_text(encoding='utf-8')
        content_structure_preview = content_structure_text[:8000] if len(content_structure_text) > 8000 else content_structure_text
        truncation_note = '\n\n[Content structure truncated for token limits. Focus on generating the lesson based on the template structure.]' if len(content_structure_text) > 8000 else ''
        
        # Construct final prompt
        content_structure_filename = self.content_structure_path.name
        curriculum_type = self._detect_curriculum_type()
        
        prompt = f"""You are generating a polished, publication-ready lesson for {curriculum_type}.
Read and internalize these two inputs before writing anything:

1. LESSON_GENERATION_PROMPT_GENERIC.md — this defines the canonical 10-section structure.
2. {content_structure_filename} — this defines where the lesson fits in the broader curriculum.

### CONTENT STRUCTURE REFERENCE (Curriculum Map - first 8000 chars for context):
{content_structure_preview}{truncation_note}

---

{formatted_prompt}"""
        
        return prompt
    
    def _detect_curriculum_type(self) -> str:
        """Detect curriculum type from folder name or config."""
        folder_name = self.config.folder_path.name.lower()
        domain_map = {
            'rpa': 'a Robotic Process Automation (RPA) Curriculum',
            'mdm': 'a Master Data Management (MDM) Curriculum',
            'snowflake_tune': 'a Snowflake Best Practices and Tuning Curriculum',
            'data_gov': 'a Data Governance Curriculum',
            'saas': 'an AI-Native SaaS Curriculum'
        }
        return domain_map.get(folder_name, 'a Technical Curriculum')


# ============================================================================
# Batch API Providers (Strategy Pattern)
# ============================================================================

class BatchAPIProvider(ABC):
    """Abstract base class for batch API providers."""
    
    def __init__(self, config: ConfigManager):
        self.config = config
        self.api_key = self._load_api_key()
    
    @abstractmethod
    def _load_api_key(self) -> str:
        """Load API key for the provider."""
        pass
    
    @abstractmethod
    def create_batch_requests(self, lessons: List[dict], prompt_engine: PromptEngine) -> dict:
        """Create batch requests for submission."""
        pass
    
    @abstractmethod
    def submit_batch(self, batch_data: dict) -> Tuple[str, dict]:
        """Submit batch and return (batch_id, response)."""
        pass
    
    @abstractmethod
    def poll_batch_status(self, batch_id: str) -> dict:
        """Poll batch status until completion."""
        pass
    
    @abstractmethod
    def download_results(self, batch_status: dict, output_file: Path) -> bool:
        """Download batch results."""
        pass
    
    @abstractmethod
    def process_results(self, results_file: Path, lessons_map: dict, output_dir: Path) -> dict:
        """Process results and return success/failure counts."""
        pass


class OpenAIBatchProvider(BatchAPIProvider):
    """OpenAI Batch API provider."""
    
    def __init__(self, config: ConfigManager):
        if not OPENAI_AVAILABLE:
            raise ImportError("OpenAI library not installed. Run: pip install openai")
        super().__init__(config)
        self.client = OpenAI(api_key=self.api_key)
        self.model = config.get('model', 'gpt-5-mini-2025-08-07')
        self.max_tokens = config.get('max_tokens', 16384)
    
    def _load_api_key(self) -> str:
        """Load OpenAI API key."""
        api_key_file = self.config.get('api_keys', {}).get('openai', 'openaiapikey.txt')
        try:
            with open(api_key_file, 'r', encoding='utf-8') as f:
                return f.read().strip()
        except FileNotFoundError:
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError(f"OpenAI API key not found in {api_key_file} or OPENAI_API_KEY env var")
            return api_key
    
    def create_batch_requests(self, lessons: List[dict], prompt_engine: PromptEngine) -> dict:
        """Create JSONL file for OpenAI Batch API."""
        batch_output_dir = Path(self.config.get('paths', {}).get('batch_output_dir', 'scripts/batch_api'))
        batch_output_dir.mkdir(parents=True, exist_ok=True)
        
        folder_name = self.config.folder_path.name
        jsonl_file = batch_output_dir / f"{folder_name}_batch_input.jsonl"
        
        print(f"\n{'='*80}")
        print(f"Creating OpenAI Batch JSONL file")
        print(f"Total lessons: {len(lessons)}")
        print(f"{'='*80}\n")
        
        with open(jsonl_file, 'w', encoding='utf-8') as f:
            for i, lesson in enumerate(lessons, 1):
                prompt = prompt_engine.prepare_prompt(lesson)
                
                request_data = {
                    "custom_id": f"lesson_{lesson['LESSON_CODE']}",
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": {
                        "model": self.model,
                        "messages": [{"role": "user", "content": prompt}],
                        "max_completion_tokens": self.max_tokens
                    }
                }
                
                f.write(json.dumps(request_data, ensure_ascii=False) + '\n')
                
                if i % 10 == 0:
                    print(f"  Prepared {i}/{len(lessons)} requests...")
        
        print(f"\n✓ Created JSONL file: {jsonl_file}")
        # Return data structure expected by submit_batch
        return {'jsonl_file': jsonl_file}
    
    def submit_batch(self, batch_data: dict) -> Tuple[str, dict]:
        """Upload JSONL and create batch job."""
        print(f"\n{'='*80}")
        print("Uploading JSONL file to OpenAI...")
        print(f"{'='*80}\n")
        
        jsonl_file = batch_data['jsonl_file']
        
        # Upload file
        with open(jsonl_file, 'rb') as f:
            file_obj = self.client.files.create(file=f, purpose='batch')
        print(f"✓ Uploaded file: {file_obj.id}")
        
        # Create batch
        print("\nCreating batch job...")
        batch = self.client.batches.create(
            input_file_id=file_obj.id,
            endpoint="/v1/chat/completions",
            completion_window="24h"
        )
        
        print(f"✓ Batch created: {batch.id}")
        print(f"  Status: {batch.status}")
        
        return batch.id, {'id': batch.id, 'status': batch.status, 'file_id': file_obj.id}
    
    def poll_batch_status(self, batch_id: str) -> dict:
        """Poll batch status."""
        print(f"\n{'='*80}")
        print("Polling batch status...")
        print(f"{'='*80}\n")
        
        while True:
            batch = self.client.batches.retrieve(batch_id)
            status = batch.status
            
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Status: {status}")
            if hasattr(batch, 'request_counts') and batch.request_counts:
                total = getattr(batch.request_counts, 'total', None)
                completed = getattr(batch.request_counts, 'completed', None)
                failed = getattr(batch.request_counts, 'failed', None)
                print(f"  Total: {total}, Completed: {completed}, Failed: {failed}")
            
            if status == "completed":
                print(f"\n✓ Batch completed!")
                break
            elif status in ["failed", "cancelled", "cancelling"]:
                print(f"\n✗ Batch {status}!")
                break
            
            time.sleep(60)  # Poll every minute
        
        return {'status': status, 'batch_id': batch_id}
    
    def download_results(self, batch_status: dict, output_file: Path) -> bool:
        """Download batch results."""
        batch_id = batch_status['batch_id']
        batch = self.client.batches.retrieve(batch_id)
        
        if not batch.output_file_id:
            print("✗ No output file available")
            return False
        
        print(f"\n{'='*80}")
        print("Downloading batch results...")
        print(f"{'='*80}\n")
        
        result_content = self.client.files.content(batch.output_file_id)
        with open(output_file, 'wb') as f:
            f.write(result_content.read())
        
        print(f"✓ Downloaded results to: {output_file}")
        return True
    
    def process_results(self, results_file: Path, lessons_map: dict, output_dir: Path) -> dict:
        """Process OpenAI batch results."""
        print(f"\n{'='*80}")
        print("Processing batch results...")
        print(f"{'='*80}\n")
        
        lessons_map_by_id = {f"lesson_{lesson['LESSON_CODE']}": lesson for lesson in lessons_map}
        success_count = 0
        failed_count = 0
        
        with open(results_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    result = json.loads(line.strip())
                    custom_id = result.get('custom_id', '')
                    response_obj = result.get('response', {})
                    status_code = response_obj.get('status_code')
                    
                    if status_code == 200:
                        lesson_data = lessons_map_by_id.get(custom_id)
                        if not lesson_data:
                            continue
                        
                        response_body = response_obj.get('body', {})
                        choices = response_body.get('choices', [])
                        if choices:
                            lesson_content = choices[0].get('message', {}).get('content', '')
                            
                            if lesson_content:
                                self._save_lesson(lesson_data, lesson_content, output_dir)
                                success_count += 1
                                if success_count % 10 == 0:
                                    print(f"  Saved {success_count} lessons...")
                            else:
                                failed_count += 1
                    else:
                        error_body = response_obj.get('body', {})
                        error_msg = error_body.get('error', {}).get('message', f'HTTP {status_code}')
                        print(f"✗ Failed: {custom_id} - {error_msg}")
                        failed_count += 1
                except Exception as e:
                    print(f"✗ Error processing line {line_num}: {e}")
                    failed_count += 1
        
        return {'success': success_count, 'failed': failed_count}
    
    def _save_lesson(self, lesson_data: dict, content: str, output_dir: Path):
        """Save a single lesson with content cleaning."""
        filename = f"{lesson_data['LESSON_CODE']}.md"
        output_path = output_dir / filename
        
        # Clean content
        cleaned = content
        cleaned = re.sub(r'^# LLM Prompt:.*?\n', '', cleaned, flags=re.MULTILINE)
        cleaned = re.sub(r'^## \*\*Context & Parameters\*\*.*?\n(?=##|$)', '', cleaned, flags=re.MULTILINE | re.DOTALL)
        cleaned = re.sub(r'^## \*\*Content Output Schema.*?\n(?=##|#)', '', cleaned, flags=re.MULTILINE | re.DOTALL)
        cleaned = re.sub(r'\n---\s*\nQuality and governance notes.*$', '', cleaned, flags=re.DOTALL)
        
        # Ensure content starts properly
        if not re.match(r'^(# Lesson |# Section 1:|## Section 1:)', cleaned):
            match = re.search(r'(^# Lesson .*|^# Section 1:.*|^## Section 1:)', cleaned, re.MULTILINE)
            if match:
                cleaned = cleaned[match.start():]
        
        output_path.write_text(cleaned, encoding='utf-8')


class ClaudeBatchProvider(BatchAPIProvider):
    """Claude Message Batches API provider."""
    
    API_BASE_URL = "https://api.anthropic.com/v1"
    ANTHROPIC_VERSION = "2023-06-01"
    
    def __init__(self, config: ConfigManager):
        if not REQUESTS_AVAILABLE:
            raise ImportError("requests library not installed. Run: pip install requests")
        super().__init__(config)
        self.model = config.get('model', 'claude-haiku-4-5-20251001')
        self.max_tokens = config.get('max_tokens', 16384)
    
    def _load_api_key(self) -> str:
        """Load Anthropic API key."""
        api_key_file = self.config.get('api_keys', {}).get('claude', 'claudeAPIkey.txt')
        try:
            with open(api_key_file, 'r', encoding='utf-8') as f:
                return f.read().strip()
        except FileNotFoundError:
            api_key = os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                raise ValueError(f"Anthropic API key not found in {api_key_file} or ANTHROPIC_API_KEY env var")
            return api_key
    
    def create_batch_requests(self, lessons: List[dict], prompt_engine: PromptEngine) -> dict:
        """Create batch requests JSON for Claude."""
        batch_output_dir = Path(self.config.get('paths', {}).get('batch_output_dir', 'scripts/batch_api'))
        batch_output_dir.mkdir(parents=True, exist_ok=True)
        
        folder_name = self.config.folder_path.name
        batch_file = batch_output_dir / f"{folder_name}_batch_requests.json"
        
        print(f"\n{'='*80}")
        print(f"Creating Claude Batch Requests JSON")
        print(f"Total lessons: {len(lessons)}")
        print(f"{'='*80}\n")
        
        requests_list = []
        
        for i, lesson in enumerate(lessons, 1):
            prompt = prompt_engine.prepare_prompt(lesson)
            
            request_data = {
                "custom_id": f"lesson_{lesson['LESSON_CODE']}",
                "params": {
                    "model": self.model,
                    "max_tokens": self.max_tokens,
                    "messages": [{"role": "user", "content": prompt}]
                }
            }
            
            requests_list.append(request_data)
            
            if i % 10 == 0:
                print(f"  Prepared {i}/{len(lessons)} requests...")
        
        batch_data = {"requests": requests_list}
        
        # Validate size
        batch_json = json.dumps(batch_data, ensure_ascii=False)
        size_mb = len(batch_json.encode('utf-8')) / (1024 * 1024)
        
        print(f"\n✓ Created batch requests file")
        print(f"  File: {batch_file}")
        print(f"  Requests: {len(requests_list)}")
        print(f"  Size: {size_mb:.2f} MB")
        
        if size_mb > 256:
            raise ValueError("Batch size exceeds 256 MB limit!")
        
        with open(batch_file, 'w', encoding='utf-8') as f:
            json.dump(batch_data, f, ensure_ascii=False, indent=2)
        
        # Return batch_data dict expected by submit_batch
        return batch_data
    
    def submit_batch(self, batch_data: dict) -> Tuple[str, dict]:
        """Submit batch to Claude API."""
        print(f"\n{'='*80}")
        print("Submitting batch to Anthropic API...")
        print(f"{'='*80}\n")
        
        url = f"{self.API_BASE_URL}/messages/batches"
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": self.ANTHROPIC_VERSION,
            "content-type": "application/json"
        }
        
        response = requests.post(url, headers=headers, json=batch_data)
        response.raise_for_status()
        
        batch_response = response.json()
        batch_id = batch_response.get('id')
        
        print(f"✓ Batch submitted successfully")
        print(f"  Batch ID: {batch_id}")
        print(f"  Status: {batch_response.get('processing_status', 'unknown')}")
        
        return batch_id, batch_response
    
    def poll_batch_status(self, batch_id: str) -> dict:
        """Poll batch status."""
        print(f"\n{'='*80}")
        print("Polling batch status...")
        print(f"{'='*80}\n")
        
        url = f"{self.API_BASE_URL}/messages/batches/{batch_id}"
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": self.ANTHROPIC_VERSION
        }
        
        max_attempts = 1440  # 24 hours
        attempt = 0
        
        while attempt < max_attempts:
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            batch_status = response.json()
            
            processing_status = batch_status.get('processing_status')
            request_counts = batch_status.get('request_counts', {})
            
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Status: {processing_status}")
            if request_counts:
                print(f"  Processing: {request_counts.get('processing', 0)}, "
                      f"Succeeded: {request_counts.get('succeeded', 0)}, "
                      f"Errored: {request_counts.get('errored', 0)}")
            
            if processing_status == "ended":
                print(f"\n✓ Batch processing completed!")
                break
            elif processing_status in ["canceled", "expired"]:
                print(f"\n✗ Batch {processing_status}!")
                break
            
            attempt += 1
            time.sleep(60)
        
        return batch_status
    
    def download_results(self, batch_status: dict, output_file: Path) -> bool:
        """Download batch results."""
        results_url = batch_status.get('results_url')
        
        if not results_url:
            print("✗ No results URL available")
            return False
        
        print(f"\n{'='*80}")
        print("Downloading batch results...")
        print(f"{'='*80}\n")
        
        headers = {
            "anthropic-version": self.ANTHROPIC_VERSION,
            "x-api-key": self.api_key
        }
        
        response = requests.get(results_url, headers=headers, stream=True)
        response.raise_for_status()
        
        with open(output_file, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        print(f"✓ Downloaded results to: {output_file}")
        return True
    
    def process_results(self, results_file: Path, lessons_map: dict, output_dir: Path) -> dict:
        """Process Claude batch results."""
        print(f"\n{'='*80}")
        print("Processing batch results...")
        print(f"{'='*80}\n")
        
        lessons_map_by_id = {f"lesson_{lesson['LESSON_CODE']}": lesson for lesson in lessons_map}
        success_count = 0
        failed_count = 0
        
        with open(results_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    result = json.loads(line.strip())
                    custom_id = result.get('custom_id', '')
                    result_obj = result.get('result', {})
                    result_type = result_obj.get('type', '')
                    
                    if result_type == 'succeeded':
                        lesson_data = lessons_map_by_id.get(custom_id)
                        if not lesson_data:
                            continue
                        
                        message = result_obj.get('message', {})
                        content = message.get('content', [])
                        
                        if content and len(content) > 0:
                            lesson_content = content[0].get('text', '')
                            
                            if lesson_content:
                                self._save_lesson(lesson_data, lesson_content, output_dir)
                                success_count += 1
                                if success_count % 10 == 0:
                                    print(f"  Saved {success_count} lessons...")
                            else:
                                failed_count += 1
                    elif result_type == 'errored':
                        error = result_obj.get('error', {})
                        error_msg = error.get('message', 'Unknown error')
                        print(f"✗ Failed: {custom_id} - {error_msg}")
                        failed_count += 1
                except Exception as e:
                    print(f"✗ Error processing line {line_num}: {e}")
                    failed_count += 1
        
        return {'success': success_count, 'failed': failed_count}
    
    def _save_lesson(self, lesson_data: dict, content: str, output_dir: Path):
        """Save a single lesson with content cleaning."""
        filename = f"{lesson_data['LESSON_CODE']}.md"
        output_path = output_dir / filename
        
        # Clean content (same as OpenAI)
        cleaned = content
        cleaned = re.sub(r'^# LLM Prompt:.*?\n', '', cleaned, flags=re.MULTILINE)
        cleaned = re.sub(r'^## \*\*Context & Parameters\*\*.*?\n(?=##|$)', '', cleaned, flags=re.MULTILINE | re.DOTALL)
        cleaned = re.sub(r'^## \*\*Content Output Schema.*?\n(?=##|#)', '', cleaned, flags=re.MULTILINE | re.DOTALL)
        cleaned = re.sub(r'\n---\s*\nQuality and governance notes.*$', '', cleaned, flags=re.DOTALL)
        
        # Ensure content starts properly
        if not re.match(r'^(# Lesson |# Section 1:|## Section 1:)', cleaned):
            match = re.search(r'(^# Lesson .*|^# Section 1:.*|^## Section 1:)', cleaned, re.MULTILINE)
            if match:
                cleaned = cleaned[match.start():]
        
        output_path.write_text(cleaned, encoding='utf-8')


# ============================================================================
# Main Agent
# ============================================================================

class LessonGeneratorAgent:
    """Main agent orchestrating lesson generation."""
    
    def __init__(self, folder_path: str, config: ConfigManager):
        self.folder_path = Path(folder_path)
        self.config = config
        self.output_dir = self.folder_path / "lessons"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.loader = CurriculumLoader(self.folder_path)
        self.prompt_engine = PromptEngine(self.config, self.loader.content_structure_path)
        
        # Initialize provider
        provider_name = self.config.get('api_provider')
        if provider_name == 'openai':
            self.provider = OpenAIBatchProvider(self.config)
        elif provider_name == 'claude':
            self.provider = ClaudeBatchProvider(self.config)
        else:
            raise ValueError(f"Unknown provider: {provider_name}")
    
    def _filter_lessons(self, lessons: List[dict]) -> List[dict]:
        """Filter lessons based on config."""
        # Check existing lessons
        if not self.config.get('overwrite_existing', False):
            generated = set()
            for f in self.output_dir.glob("*.md"):
                match = re.match(r'(M\d+-L\d{3})', f.stem)
                if match:
                    generated.add(match.group(1))
            
            lessons = [l for l in lessons if l['LESSON_CODE'] not in generated]
            if generated:
                print(f"ℹ Skipping {len(generated)} already-generated lessons")
        
        # Test mode
        if self.config.get('test_mode', False):
            test_lessons = self.config.get('test_lesson_codes', [])
            if test_lessons:
                lessons = [l for l in lessons if l['LESSON_CODE'] in test_lessons]
            else:
                # Smart default: first lesson of M0, first lesson of M1
                lessons_by_module = {}
                for lesson in lessons:
                    module = lesson['MODULE_CODE']
                    if module not in lessons_by_module:
                        lessons_by_module[module] = []
                    lessons_by_module[module].append(lesson)
                
                smart_lessons = []
                for module in sorted(lessons_by_module.keys())[:2]:
                    if lessons_by_module[module]:
                        smart_lessons.append(lessons_by_module[module][0])
                
                lessons = smart_lessons
                print(f"ℹ Test mode: Auto-selected {len(lessons)} lessons")
        
        # Start from lesson
        start_from = self.config.get('start_from_lesson')
        if start_from:
            filtered = []
            start_found = False
            for lesson in lessons:
                if lesson['LESSON_CODE'] == start_from:
                    start_found = True
                if start_found or lesson['LESSON_CODE'] >= start_from:
                    filtered.append(lesson)
            lessons = filtered
            if not start_found:
                print(f"⚠ Note: {start_from} not found, starting from first lesson")
        
        return lessons
    
    def _get_smart_test_lessons(self, lessons: List[dict]) -> List[dict]:
        """Get smart default test lessons (first of M0, first of M1)."""
        lessons_by_module = {}
        for lesson in lessons:
            module = lesson['MODULE_CODE']
            if module not in lessons_by_module:
                lessons_by_module[module] = []
            lessons_by_module[module].append(lesson)
        
        smart_lessons = []
        for module in sorted(lessons_by_module.keys())[:2]:
            if lessons_by_module[module]:
                smart_lessons.append(lessons_by_module[module][0])
        
        return smart_lessons
    
    def run(self) -> dict:
        """Execute the full lesson generation workflow."""
        print(f"\n{'='*80}")
        print("General-Purpose Lesson Generator Agent")
        print(f"Folder: {self.folder_path}")
        print(f"Provider: {self.config.get('api_provider')}")
        print(f"{'='*80}\n")
        
        # Load curriculum
        print("Parsing content structure...")
        lessons, modules = self.loader.parse()
        print(f"✓ Found {len(lessons)} lessons across {len(modules)} modules\n")
        
        # Filter lessons
        lessons = self._filter_lessons(lessons)
        
        if not lessons:
            print("No lessons to process!")
            return {'success': 0, 'failed': 0, 'total': 0}
        
        print(f"Lessons to process: {len(lessons)}\n")
        
        # Create batch requests
        batch_request_data = self.provider.create_batch_requests(lessons, self.prompt_engine)
        
        # Submit batch (provider may return different data structures)
        batch_id, batch_response = self.provider.submit_batch(batch_request_data)
        print(f"\n{'='*80}")
        print(f"Batch ID: {batch_id}")
        print(f"{'='*80}\n")
        
        # Poll for completion
        final_status = self.provider.poll_batch_status(batch_id)
        
        # Download and process results
        if final_status.get('processing_status') == 'ended' or final_status.get('status') == 'completed':
            # Determine results file path
            batch_output_dir = Path(self.config.get('paths', {}).get('batch_output_dir', 'scripts/batch_api'))
            folder_name = self.folder_path.name
            
            if isinstance(self.provider, OpenAIBatchProvider):
                results_file = batch_output_dir / f"{folder_name}_batch_results.jsonl"
            else:
                results_file = batch_output_dir / f"{folder_name}_batch_results.jsonl"
            
            if self.provider.download_results(final_status, results_file):
                results = self.provider.process_results(results_file, lessons, self.output_dir)
                
                print(f"\n{'='*80}")
                print("Results Summary:")
                print(f"  ✓ Success: {results['success']}")
                print(f"  ✗ Failed: {results['failed']}")
                print(f"  Total: {results['success'] + results['failed']}")
                print(f"{'='*80}\n")
                
                return results
        
        return {'success': 0, 'failed': 0, 'total': 0}


# ============================================================================
# CLI Interface
# ============================================================================

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate lessons for any curriculum folder using batch APIs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python scripts/generate_lessons_batch.py src/data/rpa

  # With options
  python scripts/generate_lessons_batch.py src/data/rpa --api-provider claude --test-mode
        """
    )
    
    parser.add_argument(
        'folder_path',
        help='Path to curriculum folder (e.g., src/data/rpa)'
    )
    
    parser.add_argument(
        '--api-provider',
        choices=['openai', 'claude'],
        help='API provider to use (default: auto-detect from config/model)'
    )
    
    parser.add_argument(
        '--model',
        help='Model name (overrides config)'
    )
    
    parser.add_argument(
        '--start-from',
        help='Start processing from this lesson code (inclusive)'
    )
    
    parser.add_argument(
        '--test-mode',
        action='store_true',
        help='Enable test mode (process limited lessons)'
    )
    
    parser.add_argument(
        '--test-lessons',
        nargs='+',
        help='Lesson codes to process in test mode (e.g., M0-L001 M0-L002)'
    )
    
    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='Overwrite existing lessons (default: skip)'
    )
    
    parser.add_argument(
        '--config',
        help='Path to config file (default: auto-detect)'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    
    # Build CLI config overrides
    cli_config = {}
    if args.api_provider:
        cli_config['api_provider'] = args.api_provider
    if args.model:
        cli_config['model'] = args.model
    if args.start_from:
        cli_config['start_from_lesson'] = args.start_from
    if args.test_mode:
        cli_config['test_mode'] = True
    if args.test_lessons:
        cli_config['test_lesson_codes'] = args.test_lessons
    if args.overwrite:
        cli_config['overwrite_existing'] = True
    
    # Initialize config
    try:
        config = ConfigManager(args.folder_path, cli_config)
    except Exception as e:
        print(f"✗ Configuration error: {e}")
        sys.exit(1)
    
    # Create and run agent
    try:
        agent = LessonGeneratorAgent(args.folder_path, config)
        results = agent.run()
        
        if results['success'] > 0:
            print("✓ Lesson generation completed successfully!")
            sys.exit(0)
        else:
            print("⚠ No lessons were generated")
            sys.exit(1)
    except KeyboardInterrupt:
        print("\n⚠ Interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"✗ Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()

