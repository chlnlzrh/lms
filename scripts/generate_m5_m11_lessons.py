"""
Generate M5-M11 lessons in parallel batches (3 at a time) with 10-second pauses between batches using OpenAI
This script is based on generate_missing_lessons.py but specifically for modules M5-M11
"""
import os
import sys
import time
import asyncio
from datetime import datetime
from openai import AsyncOpenAI

# Fix Windows console encoding for Unicode
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

# Read OpenAI API key
try:
    with open("openaiapikey.txt", "r", encoding="utf-8") as f:
        api_key = f.read().strip()
except:
    api_key = None

if not api_key:
    api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    print("="*80)
    print("ERROR: OpenAI API key not found!")
    print("="*80)
    raise ValueError("OpenAI API key required in openaiapikey.txt")

client = AsyncOpenAI(api_key=api_key)
MODEL_NAME = "gpt-5-mini"  # Primary model
FALLBACK_MODEL = "gpt-5-nano"  # Fallback if gpt-5-mini not available

# Note: Prompt template is reloaded fresh in generate_lesson() to ensure we always use the latest version

# Load M5-M11 lessons from extracted file
import re
import json

# Read the extracted lessons file
with open("scripts/m5_m11_lessons.txt", "r", encoding="utf-8") as f:
    m5_m11_content = f.read()

# Extract the missing_lessons list using regex
lessons_match = re.search(r'missing_lessons = \[(.*?)\]', m5_m11_content, re.DOTALL)
if lessons_match:
    # Extract just the list content, not the variable assignment
    lessons_str = "[" + lessons_match.group(1) + "]"
    # Use eval safely here since we control the input
    m5_m11_lessons = eval(lessons_str)
else:
    m5_m11_lessons = []

# Filter out already generated lessons
lesson_dir = "src/data/saas/lessons"
generated = set()
if os.path.exists(lesson_dir):
    files = [f for f in os.listdir(lesson_dir) if f.endswith('.md')]
    for f in files:
        match = re.match(r'(M[5-9]-L\d{3}|M1[01]-L\d{3})', f)
        if match:
            generated.add(match.group(1))

print(f"Already generated: {len(generated)} M5-M11 lessons")
remaining_lessons = [l for l in m5_m11_lessons if l['LESSON_CODE'] not in generated]
print(f"Remaining to generate: {len(remaining_lessons)} lessons")

if remaining_lessons:
    counts = {}
    for lesson in remaining_lessons:
        mod = lesson['MODULE_CODE']
        counts[mod] = counts.get(mod, 0) + 1
    print("Breakdown by module:")
    for mod in ['M5', 'M6', 'M7', 'M8', 'M9', 'M10', 'M11']:
        print(f"  {mod}: {counts.get(mod, 0)} lessons")

# Use remaining lessons
missing_lessons = remaining_lessons

# Default module metadata
DEFAULT_AUDIENCE = "Software engineers building SaaS platforms"
DEFAULT_FIRM_TYPE = "Technology companies and engineering teams"
DEFAULT_INDUSTRY = "SaaS platform development"

# Module-specific metadata
MODULE_METADATA = {
    "M5": {
        "MODULE_NAME": "Identity, AuthN/Z & Privacy",
        "AUDIENCE_DESCRIPTION": "Security engineers and developers building authentication and authorization systems",
        "INDUSTRY_DOMAIN": "Identity and access management, security, privacy compliance"
    },
    "M6": {
        "MODULE_NAME": "Payments, Billing & Monetization",
        "AUDIENCE_DESCRIPTION": "Payment engineers and developers building billing and subscription systems",
        "INDUSTRY_DOMAIN": "Payments, billing, subscriptions, revenue operations"
    },
    "M7": {
        "MODULE_NAME": "Files, Media & CDN",
        "AUDIENCE_DESCRIPTION": "Media engineers and developers building file storage and CDN systems",
        "INDUSTRY_DOMAIN": "File storage, media processing, CDN, object storage"
    },
    "M8": {
        "MODULE_NAME": "Jobs, Schedulers & Integrations",
        "AUDIENCE_DESCRIPTION": "Backend engineers building job processing and integration systems",
        "INDUSTRY_DOMAIN": "Job queues, schedulers, integrations, workflow automation"
    },
    "M9": {
        "MODULE_NAME": "Testing & Quality Engineering",
        "AUDIENCE_DESCRIPTION": "QA engineers and developers building testing infrastructure",
        "INDUSTRY_DOMAIN": "Software testing, quality assurance, test automation"
    },
    "M10": {
        "MODULE_NAME": "CI/CD, Release & Environments",
        "AUDIENCE_DESCRIPTION": "DevOps engineers and developers building CI/CD pipelines",
        "INDUSTRY_DOMAIN": "CI/CD, DevOps, deployment, infrastructure automation"
    },
    "M11": {
        "MODULE_NAME": "Observability, SRE & Operations",
        "AUDIENCE_DESCRIPTION": "SREs and operations engineers building observability systems",
        "INDUSTRY_DOMAIN": "Observability, monitoring, SRE, operations"
    }
}

async def generate_lesson(lesson_details, lesson_num, total_in_batch):
    # Reload prompt template fresh for each lesson to ensure we always use the latest version
    with open("LESSON_GENERATION_PROMPT_GENERIC.md", "r", encoding="utf-8") as f:
        current_prompt_template = f.read()
    
    # Prepare variables for the prompt template (LESSON_GENERATION_PROMPT_GENERIC.md is a Python f-string)
    complexity_label = {'F': 'Foundation', 'I': 'Intermediate', 'A': 'Advanced', 'E': 'Expert'}.get(lesson_details['COMPLEXITY'], 'Foundation')
    module_code = lesson_details.get('MODULE_CODE', 'M5')
    module_name = lesson_details.get('MODULE_NAME', MODULE_METADATA.get(module_code, {}).get('MODULE_NAME', 'Unknown Module'))
    audience_desc = MODULE_METADATA.get(module_code, {}).get('AUDIENCE_DESCRIPTION', DEFAULT_AUDIENCE)
    industry_domain = MODULE_METADATA.get(module_code, {}).get('INDUSTRY_DOMAIN', DEFAULT_INDUSTRY)
    
    # Evaluate the prompt template as an f-string
    # The template file contains: prompt = f"""..."""
    # We need to extract the f-string content and evaluate it
    template_content = current_prompt_template.strip()
    if template_content.startswith('prompt = f"""'):
        template_content = template_content[13:]  # Remove 'prompt = f"""'
    if template_content.endswith('"""'):
        template_content = template_content[:-3]  # Remove trailing '"""'
    template_content = template_content.strip()
    
    # Now evaluate it as an f-string by replacing all the f-string expressions manually for safety
    formatted_prompt = template_content
    formatted_prompt = formatted_prompt.replace("{lesson_details['LESSON_CODE']}", lesson_details['LESSON_CODE'])
    formatted_prompt = formatted_prompt.replace("{lesson_details['LESSON_TITLE']}", lesson_details['LESSON_TITLE'])
    formatted_prompt = formatted_prompt.replace("{lesson_details.get('MODULE_CODE', 'M0')}", module_code)
    formatted_prompt = formatted_prompt.replace("{lesson_details.get('MODULE_NAME', MODULE_METADATA.get(lesson_details.get('MODULE_CODE', 'M0'), {}).get('MODULE_NAME', 'Unknown Module'))}", module_name)
    formatted_prompt = formatted_prompt.replace("{lesson_details.get('SPECIFIC_FOCUS', 'General')}", lesson_details.get('SPECIFIC_FOCUS', 'General'))
    formatted_prompt = formatted_prompt.replace("{'Foundation' if lesson_details['COMPLEXITY'] == 'F' else 'Intermediate' if lesson_details['COMPLEXITY'] == 'I' else 'Advanced' if lesson_details['COMPLEXITY'] == 'A' else 'Expert'}", complexity_label)
    formatted_prompt = formatted_prompt.replace("{lesson_details['COMPLEXITY']}", lesson_details['COMPLEXITY'])
    formatted_prompt = formatted_prompt.replace("{lesson_details['TIME']}", str(lesson_details['TIME']))
    formatted_prompt = formatted_prompt.replace("{MODULE_METADATA.get(lesson_details.get('MODULE_CODE', 'M0'), {}).get('AUDIENCE_DESCRIPTION', DEFAULT_AUDIENCE)}", audience_desc)
    formatted_prompt = formatted_prompt.replace("{DEFAULT_FIRM_TYPE}", DEFAULT_FIRM_TYPE)
    formatted_prompt = formatted_prompt.replace("{lesson_details['LIST_PREREQUISITES']}", lesson_details['LIST_PREREQUISITES'])
    formatted_prompt = formatted_prompt.replace("{lesson_details['RELATED_LESSON_CODES']}", lesson_details['RELATED_LESSON_CODES'])
    formatted_prompt = formatted_prompt.replace("{MODULE_METADATA.get(lesson_details.get('MODULE_CODE', 'M0'), {}).get('INDUSTRY_DOMAIN', DEFAULT_INDUSTRY)}", industry_domain)
    
    # Reload content structure fresh for each lesson (though this changes less frequently)
    with open("src/data/saas/content_structure_ai-native-saas-curriculum-lesson-maps.md", "r", encoding="utf-8") as f:
        current_content_structure = f.read()
    
    # Add the content structure reference (limited to avoid token limits)
    content_structure_preview = current_content_structure[:8000] if len(current_content_structure) > 8000 else current_content_structure
    truncation_note = '\n\n[Content structure truncated for token limits. Focus on generating the lesson based on the template structure.]' if len(current_content_structure) > 8000 else ''
    
    # Construct the final prompt
    prompt = f"""You are generating a polished, publication-ready lesson for an AI-Native SaaS Curriculum.
Read and internalize these two inputs before writing anything:

1. LESSON_GENERATION_PROMPT_GENERIC.md — this defines the canonical 10-section structure.
2. content_structure_ai-native-saas-curriculum-lesson-maps.md — this defines where the lesson fits in the broader curriculum.

### CONTENT STRUCTURE REFERENCE (Curriculum Map - first 8000 chars for context):
{content_structure_preview}{truncation_note}

---

{formatted_prompt}"""

    print(f"\n[Batch {lesson_num}/{total_in_batch}] Starting: {lesson_details['LESSON_CODE']} - {lesson_details['LESSON_TITLE']}")
    print(f"  Complexity: {lesson_details['COMPLEXITY']} | Duration: {lesson_details['TIME']} minutes")
    
    # Use gpt-5-mini as primary (it uses max_completion_tokens, not max_tokens)
    # gpt-5-mini doesn't support temperature (only default value of 1)
    model_to_use = MODEL_NAME
    print(f"  Using model: {model_to_use}")
    
    # gpt-5-mini uses max_completion_tokens instead of max_tokens
    api_params = {
        "model": model_to_use,
        "messages": [{"role": "user", "content": prompt}],
        "stream": True
    }
    
    # gpt-5-mini requires max_completion_tokens (not max_tokens) and doesn't support temperature
    if model_to_use == "gpt-5-mini":
        api_params["max_completion_tokens"] = 16384
        # gpt-5-mini only supports default temperature (1), don't set it
    # gpt-5-nano: no max_tokens or max_completion_tokens, temperature defaults to 1
    
    try:
        # OpenAI async streaming
        stream = await client.chat.completions.create(**api_params)
        
        lesson_content_parts = []
        async for chunk in stream:
            if chunk.choices[0].delta.content:
                content = chunk.choices[0].delta.content
                lesson_content_parts.append(content)
                if len(lesson_content_parts) % 100 == 0:
                    print(".", end="", flush=True)
        lesson_content = "".join(lesson_content_parts)
        print()  # New line after dots
        
    except Exception as e:
        print(f"  [WARNING] Streaming failed for {lesson_details['LESSON_CODE']}, trying non-streaming: {e}")
        # Non-streaming attempt
        non_stream_params = {
            "model": model_to_use,
            "messages": [{"role": "user", "content": prompt}]
        }
        if model_to_use == "gpt-5-mini":
            non_stream_params["max_completion_tokens"] = 16384  # gpt-5-mini uses max_completion_tokens, not max_tokens
            # gpt-5-mini only supports default temperature (1), don't set it
        
        response = await client.chat.completions.create(**non_stream_params)
        lesson_content = response.choices[0].message.content

    # Post-process to remove any prompt artifacts
    # Remove common LLM prompt sections that might appear
    import re as re_module
    patterns_to_remove = [
        r'^(?:You are generating|CRITICAL|IMPORTANT|Note:|## LLM Prompt|## Prompt Template).*?\n\n',
        r'^---\s*\n*(?:LLM|Prompt|Template).*?\n\n',
    ]
    for pattern in patterns_to_remove:
        lesson_content = re_module.sub(pattern, '', lesson_content, flags=re_module.MULTILINE | re_module.IGNORECASE)

    date_str = datetime.now().strftime("%Y-%m-%d")
    # Clean filename: remove/replace all special characters
    filename_slug = lesson_details['LESSON_TITLE'].lower()
    # Replace Unicode arrows and special chars
    filename_slug = filename_slug.replace('→', 'to').replace('→', 'to')
    filename_slug = filename_slug.replace('(', '').replace(')', '').replace('/', '-')
    filename_slug = filename_slug.replace(':', '').replace(',', '').replace('&', 'and')
    filename_slug = re_module.sub(r'[^\w\s-]', '', filename_slug)
    filename_slug = re_module.sub(r'[-\s]+', '-', filename_slug)
    filename_slug = filename_slug.strip('-')
    
    filename = f"{lesson_details['LESSON_CODE']}-{filename_slug}--{date_str}.md"
    output_path = os.path.join(lesson_dir, filename)
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(lesson_content)
    
    lines = len(lesson_content.split('\n'))
    chars = len(lesson_content)
    
    print(f"  [SUCCESS] {lesson_details['LESSON_CODE']}: {lines} lines, {chars:,} chars -> {filename}")
    
    return output_path

async def main():
    print("="*80)
    print("GENERATING M5-M11 LESSONS IN PARALLEL BATCHES")
    print(f"Total lessons to generate: {len(missing_lessons)}")
    print("Batch size: 3 lessons in parallel")
    print("Delay between batches: 10 seconds")
    print("="*80)
    
    generated_files = []
    batch_size = 3
    
    # Process lessons in batches of 3
    for batch_start in range(0, len(missing_lessons), batch_size):
        batch_num = (batch_start // batch_size) + 1
        total_batches = (len(missing_lessons) + batch_size - 1) // batch_size
        batch_lessons = missing_lessons[batch_start:batch_start + batch_size]
        
        print(f"\n{'='*80}")
        print(f"BATCH {batch_num}/{total_batches}: Generating {len(batch_lessons)} lessons in parallel")
        print("="*80)
        
        # Create tasks for parallel execution
        tasks = []
        for idx, lesson in enumerate(batch_lessons, 1):
            task = generate_lesson(lesson, idx, len(batch_lessons))
            tasks.append(task)
        
        # Execute all tasks in parallel
        try:
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results
            for i, result in enumerate(batch_results):
                if isinstance(result, Exception):
                    lesson_code = batch_lessons[i]['LESSON_CODE']
                    print(f"  [ERROR] Failed to generate {lesson_code}: {result}")
                else:
                    generated_files.append(result)
            
        except Exception as e:
            print(f"[ERROR] Batch {batch_num} failed: {e}")
        
        # Wait 10 seconds before next batch (except after the last batch)
        if batch_start + batch_size < len(missing_lessons):
            print(f"\nWaiting 10 seconds before next batch...")
            await asyncio.sleep(10)
    
    print("\n" + "="*80)
    print("GENERATION COMPLETE")
    print("="*80)
    print(f"Successfully generated {len(generated_files)} lessons:")
    for f in generated_files:
        print(f"  - {f}")

if __name__ == "__main__":
    asyncio.run(main())

