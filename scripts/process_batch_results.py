"""
Process downloaded OpenAI Batch API results and save lessons.

Usage:
  python scripts/process_batch_results.py <results.jsonl> <lessons_map.json>

Or run without args to use default paths from generate_datagov_lessons_batch.py
"""
import os
import sys
import re
import json
from pathlib import Path

# Fix Windows console encoding
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

LESSON_OUTPUT_DIR = "src/data/data_gov/lessons"


def load_lessons_map(map_file=None):
    """Load lessons map from JSON file or parse from Content Structure."""
    if map_file and os.path.exists(map_file):
        with open(map_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    # If no map file, try to parse from Content Structure
    # (This is a simplified approach - full script would regenerate the map)
    print("⚠ No lessons map file provided. Using custom_id from results.")
    return {}


def process_results(results_file, lessons_map=None, output_dir=None):
    """Process batch results and save each lesson to file."""
    if output_dir is None:
        output_dir = LESSON_OUTPUT_DIR
    
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n{'='*80}")
    print("Processing batch results and saving lessons...")
    print(f"Results file: {results_file}")
    print(f"Output directory: {output_dir}")
    print(f"{'='*80}\n")
    
    success_count = 0
    failed_count = 0
    
    if not os.path.exists(results_file):
        print(f"✗ Results file not found: {results_file}")
        return
    
    with open(results_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                result = json.loads(line.strip())
                custom_id = result.get('custom_id', '')
                
                # Extract lesson code from custom_id (format: "lesson_M0-L001")
                lesson_code_match = re.search(r'lesson_(M\d+-L\d{3})', custom_id)
                lesson_code = lesson_code_match.group(1) if lesson_code_match else custom_id.replace('lesson_', '')
                
                # Batch API results structure: response.status_code indicates success
                response_obj = result.get('response', {})
                status_code = response_obj.get('status_code')
                
                if status_code == 200:
                    # Extract response content
                    response_body = response_obj.get('body', {})
                    choices = response_body.get('choices', [])
                    if choices:
                        lesson_content = choices[0].get('message', {}).get('content', '')
                        
                        if lesson_content:
                            # Save lesson to file
                            filename = f"{lesson_code}.md"
                            output_path = os.path.join(output_dir, filename)
                            
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
    """Main execution."""
    results_file = sys.argv[1] if len(sys.argv) > 1 else "scripts/batch_api/datagov_lessons_results.jsonl"
    lessons_map_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    if not os.path.exists(results_file):
        print(f"Error: Results file not found: {results_file}")
        print("\nUsage:")
        print("  python scripts/process_batch_results.py <results.jsonl> [lessons_map.json]")
        sys.exit(1)
    
    lessons_map = load_lessons_map(lessons_map_file)
    process_results(results_file, lessons_map)


if __name__ == '__main__':
    main()

