"""
Generate Snowflake Tune lessons using Anthropic Claude Message Batches API.

This script follows the 10-step process for calling claude-haiku-4-5-20251001 in Batch API mode:
1. Prepare API Key
2. Structure Batch Requests
3. Validate Batch Size
4. Submit Batch Request
5. Store Batch ID
6. Poll for Batch Status
7. Wait for Processing
8. Retrieve Results URL
9. Download Results
10. Parse and Process Results

Based on generate_mdm_lessons_batch.py adapted for Anthropic Message Batches API.
"""
import os
import sys
import re
import json
import time
import requests
from datetime import datetime
from pathlib import Path

# Fix Windows console encoding for Unicode
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

# Step 1: Prepare API Key
api_key_file = "claudeAPIkey.txt"
try:
    with open(api_key_file, "r", encoding="utf-8") as f:
        api_key = f.read().strip()
except:
    api_key = None

if not api_key:
    api_key = os.getenv("ANTHROPIC_API_KEY")

if not api_key:
    print("="*80)
    print("ERROR: Anthropic API key not found!")
    print("="*80)
    print(f"Please add your Anthropic API key to {api_key_file}")
    print("Or set ANTHROPIC_API_KEY environment variable")
    print("="*80)
    raise ValueError("Anthropic API key required")

# API Configuration
API_BASE_URL = "https://api.anthropic.com/v1"
ANTHROPIC_VERSION = "2023-06-01"
MODEL_NAME = "claude-haiku-4-5-20251001"  # Claude Haiku model as specified

# Paths
CONTENT_STRUCTURE_PATH = "src/data/snowflake_tune/content_structure_snowflake-best-practices-and-tuning—lesson-maps.md"
PROMPT_TEMPLATE_PATH = "LESSON_GENERATION_PROMPT_GENERIC.md"
LESSON_OUTPUT_DIR = "src/data/snowflake_tune/lessons"
BATCH_INPUT_DIR = "scripts/batch_api"
BATCH_REQUESTS_FILE = f"{BATCH_INPUT_DIR}/snowflake_tune_batch_requests.json"
BATCH_RESULTS_FILE = f"{BATCH_INPUT_DIR}/snowflake_tune_batch_results.jsonl"

# TEST MODE: Set to False to process all remaining lessons
TEST_MODE = False
START_FROM_LESSON = "M0-L005"  # Start processing from this lesson onwards (inclusive)

# Ensure directories exist
os.makedirs(LESSON_OUTPUT_DIR, exist_ok=True)
os.makedirs(BATCH_INPUT_DIR, exist_ok=True)


def parse_content_structure(file_path):
    """Parse Content Structure.md and extract all lessons with their metadata."""
    content = Path(file_path).read_text(encoding='utf-8')
    
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


def prepare_prompt(lesson_details, prompt_template_path, content_structure_path):
    """Prepare the prompt for a lesson."""
    # Read prompt template
    with open(prompt_template_path, "r", encoding="utf-8") as f:
        current_prompt_template = f.read()
    
    # Default metadata for Snowflake Tune curriculum
    DEFAULT_AUDIENCE = "Snowflake engineers, data platform administrators, and performance optimization specialists"
    DEFAULT_FIRM_TYPE = "Organizations optimizing Snowflake performance and costs"
    DEFAULT_INDUSTRY = "Snowflake optimization, data warehousing, performance tuning, cost optimization"
    
    # Prepare variables for the prompt template
    complexity_label = {'F': 'Foundation', 'I': 'Intermediate', 'A': 'Advanced', 'E': 'Expert'}.get(
        lesson_details['COMPLEXITY'], 'Foundation'
    )
    module_code = lesson_details.get('MODULE_CODE', 'M0')
    module_name = lesson_details.get('MODULE_NAME', 'Unknown Module')
    
    # Extract f-string content from template
    template_content = current_prompt_template.strip()
    if template_content.startswith('prompt = f"""'):
        template_content = template_content[13:]
    if template_content.endswith('"""'):
        template_content = template_content[:-3]
    template_content = template_content.strip()
    
    # Replace f-string expressions manually
    formatted_prompt = template_content
    formatted_prompt = formatted_prompt.replace("{lesson_details['LESSON_CODE']}", lesson_details['LESSON_CODE'])
    formatted_prompt = formatted_prompt.replace("{lesson_details['LESSON_TITLE']}", lesson_details['LESSON_TITLE'])
    formatted_prompt = formatted_prompt.replace("{lesson_details.get('MODULE_CODE', 'M0')}", module_code)
    formatted_prompt = formatted_prompt.replace("{lesson_details.get('MODULE_NAME', MODULE_METADATA.get(lesson_details.get('MODULE_CODE', 'M0'), {}).get('MODULE_NAME', 'Unknown Module'))}", module_name)
    formatted_prompt = formatted_prompt.replace("{lesson_details.get('SPECIFIC_FOCUS', 'General')}", lesson_details.get('SPECIFIC_FOCUS', 'General'))
    formatted_prompt = formatted_prompt.replace("{'Foundation' if lesson_details['COMPLEXITY'] == 'F' else 'Intermediate' if lesson_details['COMPLEXITY'] == 'I' else 'Advanced' if lesson_details['COMPLEXITY'] == 'A' else 'Expert'}", complexity_label)
    formatted_prompt = formatted_prompt.replace("{lesson_details['COMPLEXITY']}", lesson_details['COMPLEXITY'])
    
    # Estimate duration based on complexity
    duration_map = {'F': '45', 'I': '60', 'A': '75', 'E': '90'}
    duration = duration_map.get(lesson_details['COMPLEXITY'], '60')
    formatted_prompt = formatted_prompt.replace("{lesson_details['TIME']}", duration)
    
    formatted_prompt = formatted_prompt.replace("{MODULE_METADATA.get(lesson_details.get('MODULE_CODE', 'M0'), {}).get('AUDIENCE_DESCRIPTION', DEFAULT_AUDIENCE)}", DEFAULT_AUDIENCE)
    formatted_prompt = formatted_prompt.replace("{DEFAULT_FIRM_TYPE}", DEFAULT_FIRM_TYPE)
    
    # Handle prerequisites and related lessons
    prerequisites = lesson_details.get('LIST_PREREQUISITES', 'Basic understanding of Snowflake and data warehousing concepts')
    related_lessons = lesson_details.get('RELATED_LESSON_CODES', 'Related lessons in the same module')
    
    formatted_prompt = formatted_prompt.replace("{lesson_details['LIST_PREREQUISITES']}", prerequisites)
    formatted_prompt = formatted_prompt.replace("{lesson_details['RELATED_LESSON_CODES']}", related_lessons)
    formatted_prompt = formatted_prompt.replace("{MODULE_METADATA.get(lesson_details.get('MODULE_CODE', 'M0'), {}).get('INDUSTRY_DOMAIN', DEFAULT_INDUSTRY)}", DEFAULT_INDUSTRY)
    
    # Read content structure (limited to avoid token limits)
    with open(content_structure_path, "r", encoding="utf-8") as f:
        current_content_structure = f.read()
    
    content_structure_preview = current_content_structure[:8000] if len(current_content_structure) > 8000 else current_content_structure
    truncation_note = '\n\n[Content structure truncated for token limits. Focus on generating the lesson based on the template structure.]' if len(current_content_structure) > 8000 else ''
    
    # Construct the final prompt
    # Replace SaaS references in the template with Snowflake-specific content
    formatted_prompt_snowflake = formatted_prompt.replace(
        "an AI-Native SaaS Curriculum",
        "a Snowflake Best Practices and Tuning Curriculum"
    )
    formatted_prompt_snowflake = formatted_prompt_snowflake.replace(
        "content_structure_ai-native-saas-curriculum-lesson-maps.md",
        "content_structure_snowflake-best-practices-and-tuning—lesson-maps.md"
    )
    formatted_prompt_snowflake = formatted_prompt_snowflake.replace(
        "realistic SaaS architecture patterns",
        "realistic Snowflake optimization patterns"
    )
    formatted_prompt_snowflake = formatted_prompt_snowflake.replace(
        "SaaS-specific terminology: multi-tenancy, tenant isolation, subscription models, usage-based pricing, CI/CD, and observability",
        "Snowflake-specific terminology: micro-partitions, clustering keys, warehouse sizing, credit optimization, query performance, and cost management"
    )
    formatted_prompt_snowflake = formatted_prompt_snowflake.replace(
        "common SaaS stacks — Next.js, Postgres, Vercel, AWS, and Terraform",
        "common Snowflake patterns — SQL optimization, warehouse configuration, data modeling, storage optimization, and performance tuning"
    )
    formatted_prompt_snowflake = formatted_prompt_snowflake.replace(
        "Languages accepted: TypeScript/Node.js, SQL, Terraform, or YAML",
        "Languages accepted: SQL (primary), Python (for Snowpark/Stored Procedures), or YAML (for configuration)"
    )
    formatted_prompt_snowflake = formatted_prompt_snowflake.replace(
        "tenancy isolation",
        "warehouse isolation"
    )
    
    prompt = f"""You are generating a polished, publication-ready lesson for a Snowflake Best Practices and Tuning Curriculum.
Read and internalize these two inputs before writing anything:

1. LESSON_GENERATION_PROMPT_GENERIC.md — this defines the canonical 10-section structure.
2. content_structure_snowflake-best-practices-and-tuning—lesson-maps.md — this defines where the lesson fits in the broader curriculum.

### CONTENT STRUCTURE REFERENCE (Curriculum Map - first 8000 chars for context):
{content_structure_preview}{truncation_note}

---

{formatted_prompt_snowflake}"""
    
    return prompt


def create_batch_requests(lessons, output_file):
    """Step 2: Structure Batch Requests - Create JSON with requests array."""
    print(f"\n{'='*80}")
    print(f"Step 2: Creating Batch Requests JSON")
    print(f"Total lessons to process: {len(lessons)}")
    print(f"{'='*80}\n")
    
    requests_list = []
    
    for i, lesson in enumerate(lessons, 1):
        # Prepare prompt for this lesson
        prompt = prepare_prompt(
            lesson,
            PROMPT_TEMPLATE_PATH,
            CONTENT_STRUCTURE_PATH
        )
        
        # Create request according to Anthropic Batch API format
        request_data = {
            "custom_id": f"lesson_{lesson['LESSON_CODE']}",
            "params": {
                "model": MODEL_NAME,
                "max_tokens": 16384,
                "messages": [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            }
        }
        
        requests_list.append(request_data)
        
        if i % 10 == 0:
            print(f"  Prepared {i}/{len(lessons)} requests...")
    
    # Step 3: Validate Batch Size Constraints
    batch_data = {
        "requests": requests_list
    }
    
    # Check size (approximately - actual validation done by API)
    batch_json = json.dumps(batch_data, ensure_ascii=False)
    size_mb = len(batch_json.encode('utf-8')) / (1024 * 1024)
    
    print(f"\n✓ Created batch requests file")
    print(f"  File: {output_file}")
    print(f"  Requests: {len(requests_list)}")
    print(f"  Size: {size_mb:.2f} MB")
    
    if size_mb > 256:
        print(f"  ⚠ WARNING: Batch size exceeds 256 MB limit!")
        return None
    
    # Save batch requests file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(batch_data, f, ensure_ascii=False, indent=2)
    
    return batch_data


def submit_batch(batch_data):
    """Step 4: Submit Batch Request."""
    print(f"\n{'='*80}")
    print("Step 4: Submitting batch to Anthropic API...")
    print(f"{'='*80}\n")
    
    url = f"{API_BASE_URL}/messages/batches"
    headers = {
        "x-api-key": api_key,
        "anthropic-version": ANTHROPIC_VERSION,
        "content-type": "application/json"
    }
    
    response = requests.post(url, headers=headers, json=batch_data)
    response.raise_for_status()
    
    batch_response = response.json()
    batch_id = batch_response.get('id')
    
    print(f"✓ Batch submitted successfully")
    print(f"  Batch ID: {batch_id}")
    print(f"  Status: {batch_response.get('processing_status', 'unknown')}")
    print(f"  Request counts: {batch_response.get('request_counts', {})}")
    
    return batch_id, batch_response


def poll_batch_status(batch_id):
    """Step 6 & 7: Poll for Batch Status and Wait for Completion."""
    print(f"\n{'='*80}")
    print("Step 6 & 7: Polling batch status...")
    print(f"{'='*80}\n")
    
    url = f"{API_BASE_URL}/messages/batches/{batch_id}"
    headers = {
        "x-api-key": api_key,
        "anthropic-version": ANTHROPIC_VERSION
    }
    
    max_attempts = 1440  # 24 hours with 1-minute intervals
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
            return batch_status
        elif processing_status in ["canceled", "expired"]:
            print(f"\n✗ Batch {processing_status}!")
            return batch_status
        
        attempt += 1
        time.sleep(60)  # Wait 1 minute between polls
    
    print(f"\n⚠ Maximum polling attempts reached")
    return batch_status


def download_results(batch_status, output_file):
    """Step 8 & 9: Retrieve Results URL and Download Results."""
    print(f"\n{'='*80}")
    print("Step 8 & 9: Downloading batch results...")
    print(f"{'='*80}\n")
    
    results_url = batch_status.get('results_url')
    
    if not results_url:
        print(f"✗ No results URL available")
        return False
    
    print(f"Results URL: {results_url}")
    print(f"Downloading...")
    
    headers = {
        "anthropic-version": ANTHROPIC_VERSION,
        "x-api-key": api_key
    }
    
    response = requests.get(results_url, headers=headers, stream=True)
    response.raise_for_status()
    
    with open(output_file, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    
    print(f"✓ Downloaded results to: {output_file}")
    return True


def process_results(results_file, lessons_map):
    """Step 10: Parse and Process Results."""
    print(f"\n{'='*80}")
    print("Step 10: Processing batch results and saving lessons...")
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
                        print(f"⚠ Warning: Unknown lesson ID: {custom_id}")
                        continue
                    
                    # Extract message content
                    message = result_obj.get('message', {})
                    content = message.get('content', [])
                    
                    if content and len(content) > 0:
                        lesson_content = content[0].get('text', '')
                        
                        if lesson_content:
                            # Save lesson to file
                            filename = f"{lesson_data['LESSON_CODE']}.md"
                            output_path = os.path.join(LESSON_OUTPUT_DIR, filename)
                            
                            # Clean content
                            cleaned_content = lesson_content
                            cleaned_content = re.sub(r'^# LLM Prompt:.*?\n', '', cleaned_content, flags=re.MULTILINE)
                            cleaned_content = re.sub(r'^## \*\*Context & Parameters\*\*.*?\n(?=##|$)', '', cleaned_content, flags=re.MULTILINE | re.DOTALL)
                            cleaned_content = re.sub(r'^## \*\*Content Output Schema.*?\n(?=##|#)', '', cleaned_content, flags=re.MULTILINE | re.DOTALL)
                            cleaned_content = re.sub(r'\n---\s*\nQuality and governance notes.*$', '', cleaned_content, flags=re.DOTALL)
                            
                            # Ensure content starts with lesson title
                            if not re.match(r'^(# Lesson |# Section 1:|## Section 1:)', cleaned_content):
                                match = re.search(r'(^# Lesson .*|^# Section 1:.*|^## Section 1:)', cleaned_content, re.MULTILINE)
                                if match:
                                    cleaned_content = cleaned_content[match.start():]
                            
                            with open(output_path, 'w', encoding='utf-8') as out_f:
                                out_f.write(cleaned_content)
                            
                            success_count += 1
                            if success_count % 10 == 0:
                                print(f"  Saved {success_count} lessons...")
                        else:
                            print(f"⚠ Warning: Empty content for {custom_id}")
                            failed_count += 1
                    else:
                        print(f"⚠ Warning: No content in message for {custom_id}")
                        failed_count += 1
                elif result_type == 'errored':
                    error = result_obj.get('error', {})
                    error_msg = error.get('message', 'Unknown error')
                    print(f"✗ Failed request: {custom_id} - {error_msg}")
                    failed_count += 1
                else:
                    print(f"⚠ Unknown result type for {custom_id}: {result_type}")
                    failed_count += 1
                    
            except json.JSONDecodeError as e:
                print(f"✗ Error parsing line {line_num}: {e}")
                failed_count += 1
            except Exception as e:
                print(f"✗ Error processing line {line_num}: {e}")
                failed_count += 1
    
    print(f"\n{'='*80}")
    print(f"Results Summary:")
    print(f"  ✓ Success: {success_count}")
    print(f"  ✗ Failed: {failed_count}")
    print(f"  Total: {success_count + failed_count}")
    print(f"{'='*80}\n")


def main():
    """Main execution flow - Following the 10-step process."""
    print(f"\n{'='*80}")
    print("Snowflake Tune Lessons - Claude Message Batches API")
    if START_FROM_LESSON:
        print(f"Processing all lessons from {START_FROM_LESSON} onwards")
    print(f"{'='*80}\n")
    
    # Step 1: Prepare API Key (already done above)
    print("Step 1: API Key prepared ✓\n")
    
    # Parse content structure
    print("Parsing Content Structure...")
    lessons, modules = parse_content_structure(CONTENT_STRUCTURE_PATH)
    print(f"✓ Found {len(lessons)} lessons across {len(modules)} modules\n")
    
    # Filter out already generated lessons
    generated = set()
    if os.path.exists(LESSON_OUTPUT_DIR):
        for f in os.listdir(LESSON_OUTPUT_DIR):
            if f.endswith('.md'):
                match = re.match(r'(M\d+-L\d{3})', f)
                if match:
                    generated.add(match.group(1))
    
    remaining_lessons = [l for l in lessons if l['LESSON_CODE'] not in generated]
    print(f"Already generated: {len(generated)}")
    print(f"Remaining to generate: {len(remaining_lessons)}")
    
    # Filter to start from specified lesson onwards
    if START_FROM_LESSON:
        filtered_lessons = []
        start_found = False
        for lesson in remaining_lessons:
            if lesson['LESSON_CODE'] == START_FROM_LESSON:
                start_found = True
            if start_found or lesson['LESSON_CODE'] >= START_FROM_LESSON:
                filtered_lessons.append(lesson)
        remaining_lessons = filtered_lessons
        if not start_found:
            print(f"⚠ Note: {START_FROM_LESSON} not found, starting from first remaining lesson")
        else:
            print(f"Starting from: {START_FROM_LESSON} onwards")
    
    print(f"Lessons to process in this batch: {len(remaining_lessons)}\n")
    
    if not remaining_lessons:
        print("All lessons already generated!")
        return
    
    # Step 2 & 3: Create and validate batch requests
    batch_data = create_batch_requests(remaining_lessons, BATCH_REQUESTS_FILE)
    if not batch_data:
        print("✗ Failed to create batch requests")
        return
    
    # Step 4: Submit batch
    batch_id, batch_response = submit_batch(batch_data)
    
    # Step 5: Store Batch ID
    print(f"\n{'='*80}")
    print("Step 5: Batch ID stored")
    print(f"  Batch ID: {batch_id}")
    print(f"\nYou can check status with:")
    print(f"  curl https://api.anthropic.com/v1/messages/batches/{batch_id} \\")
    print(f"    --header \"x-api-key: <your-key>\" \\")
    print(f"    --header \"anthropic-version: {ANTHROPIC_VERSION}\"")
    print(f"{'='*80}\n")
    
    # Step 6 & 7: Poll for completion
    final_batch_status = poll_batch_status(batch_id)
    
    # Step 8 & 9: Download results
    if final_batch_status.get('processing_status') == 'ended':
        if download_results(final_batch_status, BATCH_RESULTS_FILE):
            # Step 10: Process results
            process_results(BATCH_RESULTS_FILE, remaining_lessons)
    
    print("\n✓ Done!")


if __name__ == '__main__':
    main()
