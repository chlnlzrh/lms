"""
Generate Data Governance lessons using OpenAI Batch API.

This script:
1. Parses Content Structure.md to extract all lessons (M0-M14)
2. Creates a JSONL file with one request per lesson
3. Uploads JSONL to OpenAI Batch API
4. Creates a batch job
5. Polls for completion
6. Downloads results and saves each lesson

Based on generate_missing_lessons.py but adapted for Batch API workflow.
"""
import os
import sys
import re
import json
import time
from datetime import datetime
from pathlib import Path
from openai import OpenAI

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
    print("Please add your OpenAI API key to openaiapikey.txt")
    print("="*80)
    raise ValueError("OpenAI API key required in openaiapikey.txt")

client = OpenAI(api_key=api_key)

# Model to use
MODEL_NAME = "gpt-5-mini-2025-08-07"  # Batch API compatible model
# Note: Using gpt-5-mini-2025-08-07 as specified

# Paths
CONTENT_STRUCTURE_PATH = "src/data/data_gov/Content Structure.md"
PROMPT_TEMPLATE_PATH = "LESSON_GENERATION_PROMPT_GENERIC.md"
LESSON_OUTPUT_DIR = "src/data/data_gov/lessons"
BATCH_INPUT_DIR = "scripts/batch_api"
BATCH_INPUT_FILE = f"{BATCH_INPUT_DIR}/datagov_lessons_input.jsonl"
BATCH_RESULTS_FILE = f"{BATCH_INPUT_DIR}/datagov_lessons_results.jsonl"

# Ensure directories exist
os.makedirs(LESSON_OUTPUT_DIR, exist_ok=True)
os.makedirs(BATCH_INPUT_DIR, exist_ok=True)


def parse_content_structure(file_path):
    """Parse Content Structure.md and extract all lessons with their metadata."""
    content = Path(file_path).read_text(encoding='utf-8')
    
    # Find all module headers: # M0:, # M1:, etc.
    module_pattern = r'^# (M\d+):\s+(.+?)\s+—\s+Lesson Map'
    
    # Find all lesson headers: ## L001: Title [F], etc.
    # Pattern matches: ## L001: Title [F] or ## L001: Title [I]
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
    """Prepare the prompt for a lesson, similar to generate_lesson() in existing scripts."""
    # Read prompt template
    with open(prompt_template_path, "r", encoding="utf-8") as f:
        current_prompt_template = f.read()
    
    # Default metadata for Data Governance curriculum
    DEFAULT_AUDIENCE = "Data governance professionals, data stewards, and compliance officers"
    DEFAULT_FIRM_TYPE = "Organizations implementing data governance programs"
    DEFAULT_INDUSTRY = "Data governance, data management, compliance"
    
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
    
    # Replace f-string expressions manually (simplified approach)
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
    
    # Handle prerequisites and related lessons (extract from context or use defaults)
    prerequisites = lesson_details.get('LIST_PREREQUISITES', 'Basic understanding of data governance concepts')
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
    prompt = f"""You are generating a polished, publication-ready lesson for a Data Governance Curriculum.
Read and internalize these two inputs before writing anything:

1. LESSON_GENERATION_PROMPT_GENERIC.md — this defines the canonical 10-section structure.
2. Content Structure.md — this defines where the lesson fits in the broader curriculum.

### CONTENT STRUCTURE REFERENCE (Curriculum Map - first 8000 chars for context):
{content_structure_preview}{truncation_note}

---

{formatted_prompt}"""
    
    return prompt


def create_jsonl_input(lessons, output_file):
    """Create JSONL file for OpenAI Batch API with one request per lesson."""
    print(f"\n{'='*80}")
    print(f"Creating JSONL input file: {output_file}")
    print(f"Total lessons to process: {len(lessons)}")
    print(f"{'='*80}\n")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for i, lesson in enumerate(lessons, 1):
            # Prepare prompt for this lesson
            prompt = prepare_prompt(
                lesson,
                PROMPT_TEMPLATE_PATH,
                CONTENT_STRUCTURE_PATH
            )
            
            # Create request JSON (Batch API format)
            request_data = {
                "custom_id": f"lesson_{lesson['LESSON_CODE']}",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": MODEL_NAME,
                    "messages": [
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    "max_completion_tokens": 16384
                }
            }
            
            # Write as JSON line
            f.write(json.dumps(request_data, ensure_ascii=False) + '\n')
            
            if i % 10 == 0:
                print(f"  Prepared {i}/{len(lessons)} lessons...")
    
    print(f"\n✓ Created JSONL file with {len(lessons)} requests")
    print(f"  File: {output_file}")
    print(f"  Size: {Path(output_file).stat().st_size / 1024:.1f} KB\n")


def upload_file(file_path):
    """Upload JSONL file to OpenAI for Batch API."""
    print(f"Uploading file: {file_path}")
    with open(file_path, 'rb') as f:
        file_obj = client.files.create(
            file=f,
            purpose='batch'
        )
    print(f"✓ Uploaded: {file_obj.id}")
    return file_obj.id


def create_batch(file_id):
    """Create a batch job from uploaded file."""
    print(f"\nCreating batch job from file: {file_id}")
    batch = client.batches.create(
        input_file_id=file_id,
        endpoint="/v1/chat/completions",
        completion_window="24h"
    )
    print(f"✓ Batch created: {batch.id}")
    print(f"  Status: {batch.status}")
    return batch


def poll_batch_status(batch_id):
    """Poll batch job until completion."""
    print(f"\n{'='*80}")
    print("Polling batch status (this may take hours for large batches)...")
    print(f"{'='*80}\n")
    
    while True:
        batch = client.batches.retrieve(batch_id)
        status = batch.status
        
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Status: {status}")
        
        if status == "completed":
            print(f"\n✓ Batch completed!")
            print(f"  Request counts:")
            if hasattr(batch, 'request_counts') and batch.request_counts:
                # Handle Pydantic model object
                try:
                    total = getattr(batch.request_counts, 'total', None)
                    completed = getattr(batch.request_counts, 'completed', None)
                    failed = getattr(batch.request_counts, 'failed', None)
                    print(f"    Total: {total if total is not None else 'N/A'}")
                    print(f"    Completed: {completed if completed is not None else 'N/A'}")
                    print(f"    Failed: {failed if failed is not None else 'N/A'}")
                except AttributeError:
                    # Fallback if structure is different
                    print(f"    Counts: {batch.request_counts}")
            break
        elif status == "failed":
            print(f"\n✗ Batch failed!")
            if hasattr(batch, 'errors'):
                print(f"  Errors: {batch.errors}")
            break
        elif status in ["cancelling", "cancelled"]:
            print(f"\n⚠ Batch {status}")
            break
        
        # Wait before next poll (adjust interval as needed)
        time.sleep(60)  # Poll every minute
    
    return batch


def download_batch_results(batch_id, output_file):
    """Download batch results."""
    print(f"\nDownloading batch results...")
    batch = client.batches.retrieve(batch_id)
    
    if batch.output_file_id:
        # Download the results file
        result_content = client.files.content(batch.output_file_id)
        
        with open(output_file, 'wb') as f:
            f.write(result_content.read())
        
        print(f"✓ Downloaded results to: {output_file}")
        return True
    else:
        print(f"✗ No output file available")
        return False


def process_results(results_file, lessons_map):
    """Process batch results and save each lesson to file."""
    print(f"\n{'='*80}")
    print("Processing batch results and saving lessons...")
    print(f"{'='*80}\n")
    
    lessons_map_by_id = {f"lesson_{lesson['LESSON_CODE']}": lesson for lesson in lessons_map}
    
    success_count = 0
    failed_count = 0
    
    with open(results_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                result = json.loads(line.strip())
                custom_id = result.get('custom_id', '')
                
                # Batch API results structure: response.status_code indicates success
                response_obj = result.get('response', {})
                status_code = response_obj.get('status_code')
                
                if status_code == 200:
                    lesson_data = lessons_map_by_id.get(custom_id)
                    if not lesson_data:
                        print(f"⚠ Warning: Unknown lesson ID: {custom_id}")
                        continue
                    
                    # Extract response content
                    response_body = result.get('response', {}).get('body', {})
                    choices = response_body.get('choices', [])
                    if choices:
                        lesson_content = choices[0].get('message', {}).get('content', '')
                        
                        if lesson_content:
                            # Save lesson to file
                            filename = f"{lesson_data['LESSON_CODE']}.md"
                            output_path = os.path.join(LESSON_OUTPUT_DIR, filename)
                            
                            # Clean content (similar to existing scripts)
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
                        print(f"⚠ Warning: No choices in response for {custom_id}")
                        failed_count += 1
                elif status_code and status_code != 200:
                    # Request failed
                    error_body = response_obj.get('body', {})
                    error_msg = error_body.get('error', {}).get('message', f'HTTP {status_code}')
                    print(f"✗ Failed request: {custom_id} - {error_msg}")
                    failed_count += 1
                elif 'error' in result:
                    # Error at batch level
                    error_msg = result.get('error', {}).get('message', 'Unknown error')
                    print(f"✗ Failed request: {custom_id} - {error_msg}")
                    failed_count += 1
                else:
                    print(f"⚠ Unknown status for {custom_id}: status_code={status_code}")
                    
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
    """Main execution flow."""
    print(f"\n{'='*80}")
    print("Data Governance Lessons - Batch API Generator")
    print(f"{'='*80}\n")
    
    # Step 1: Parse content structure
    print("Step 1: Parsing Content Structure.md...")
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
    print(f"Remaining to generate: {len(remaining_lessons)}\n")
    
    if not remaining_lessons:
        print("All lessons already generated!")
        return
    
    # Step 2: Create JSONL input file
    print("Step 2: Creating JSONL input file...")
    create_jsonl_input(remaining_lessons, BATCH_INPUT_FILE)
    
    # Step 3: Upload file
    print("Step 3: Uploading JSONL file to OpenAI...")
    file_id = upload_file(BATCH_INPUT_FILE)
    
    # Step 4: Create batch job
    print("\nStep 4: Creating batch job...")
    batch = create_batch(file_id)
    batch_id = batch.id
    
    print(f"\n{'='*80}")
    print(f"Batch Job Information:")
    print(f"  Batch ID: {batch_id}")
    print(f"  Status: {batch.status}")
    print(f"  Input file: {file_id}")
    print(f"\nYou can check status with:")
    print(f"  batch = client.batches.retrieve('{batch_id}')")
    print(f"{'='*80}\n")
    
    # Step 5: Poll for completion
    print("Step 5: Waiting for batch completion...")
    print("(You can stop this script and resume later by polling the batch ID)\n")
    
    try:
        final_batch = poll_batch_status(batch_id)
        
        # Step 6: Download results
        if final_batch.status == "completed":
            print("\nStep 6: Downloading batch results...")
            if download_batch_results(batch_id, BATCH_RESULTS_FILE):
                # Step 7: Process and save lessons
                print("\nStep 7: Processing results and saving lessons...")
                process_results(BATCH_RESULTS_FILE, remaining_lessons)
        
    except KeyboardInterrupt:
        print(f"\n\n⚠ Script interrupted. Batch ID saved: {batch_id}")
        print(f"You can resume by running:")
        print(f"  python scripts/resume_batch.py {batch_id}")
    
    print("\n✓ Done!")


if __name__ == '__main__':
    main()

