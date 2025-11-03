"""
Remove LLM prompt metadata from generated lesson files, keeping only the actual lesson content
"""
import os
import re
import glob

lesson_dir = "src/data/saas/lessons"

def clean_lesson_file(filepath):
    """Remove LLM prompt section, keep only lesson content"""
    with open(filepath, "r", encoding="utf-8") as f:
        lines = f.readlines()
    
    content = ''.join(lines)
    
    # Check if file already starts with lesson content (no prompt)
    # Files starting with "Section 1:" (no hash) or "# Lesson" are already clean
    first_lines = ''.join(lines[:5]).lower()
    if (content.startswith('# Lesson ') or 
        content.startswith('Section 1:') or
        content.startswith('# Section 1:') or
        'section 1: header' in first_lines and '# llm prompt' not in first_lines):
        # Already clean, no need to process
        return True
    
    # Find where actual lesson content starts
    # Patterns that indicate start of actual lesson:
    # - "# Lesson" or "## Lesson"
    # - "# Section 1:" or "## Section 1:" or "## **Section 1:"
    # - Remove everything from "# LLM Prompt" until we hit actual lesson
    
    start_line = 0
    in_prompt = False
    
    for i, line in enumerate(lines):
        # Detect prompt start
        if re.match(r'^# LLM Prompt:', line) or re.match(r'^# LLM Prompt ', line):
            in_prompt = True
            continue
        
        # If we're in prompt section, look for lesson start markers
        if in_prompt:
            # Look for actual lesson content start
            if (re.match(r'^# Lesson ', line) or 
                re.match(r'^## Lesson ', line) or
                re.match(r'^# Section 1:', line) or
                re.match(r'^## \*\*Section 1:', line) or
                re.match(r'^### \*\*Section 1:', line) or
                re.match(r'^## Section 1:', line) or
                re.match(r'^Section 1:', line)):
                start_line = i
                break
            
            # Also check for "Quality and governance notes" which is at the end of prompt
            if 'Quality and governance notes' in line:
                # Skip forward to find actual lesson start
                # Look for next heading or section
                for j in range(i+1, len(lines)):
                    if (re.match(r'^# Lesson ', lines[j]) or 
                        re.match(r'^## Lesson ', lines[j]) or
                        re.match(r'^# Section 1:', lines[j]) or
                        re.match(r'^## \*\*Section 1:', lines[j]) or
                        re.match(r'^### \*\*Section 1:', lines[j]) or
                        re.match(r'^## Section 1:', lines[j]) or
                        re.match(r'^Section 1:', lines[j])):
                        start_line = j
                        break
                break
            
            # Also check for "### **Section 1:" which is prompt template, skip to actual content
            if re.match(r'^### \*\*Section 1:.*\*\*$', line):
                # This is still prompt template, continue
                continue
    
    # If no prompt found, check if file starts with lesson directly
    if not in_prompt:
        for i, line in enumerate(lines):
            if (re.match(r'^# Lesson ', line) or 
                re.match(r'^## Lesson ', line) or
                re.match(r'^# Section 1:', line) or
                re.match(r'^## \*\*Section 1:', line) or
                re.match(r'^### \*\*Section 1:', line) or
                re.match(r'^## Section 1:', line)):
                start_line = i
                break
    
    # Extract lesson content
    if start_line > 0:
        cleaned_lines = lines[start_line:]
        cleaned = ''.join(cleaned_lines)
    else:
        # Fallback: try to remove everything before first "## Section 1" or "# Lesson"
        content = ''.join(lines)
        match = re.search(r'(^# Lesson .*|^## \*\*Section 1:.*|^### \*\*Section 1:.*|^## Section 1:.*)', content, re.MULTILINE)
        if match:
            cleaned = content[match.start():]
        else:
            print(f"  [WARNING] {os.path.basename(filepath)} - couldn't find lesson start, keeping original")
            return False
    
    # Clean up any double newlines at start
    cleaned = cleaned.lstrip()
    
    # If the cleaned content is very short, keep original
    if len(cleaned) < 500:
        print(f"  [WARNING] {os.path.basename(filepath)} - cleaned content too short ({len(cleaned)} chars), keeping original")
        return False
    
    # Write cleaned content
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(cleaned)
    
    return True

# Process all lesson files
files = glob.glob(os.path.join(lesson_dir, "M[0-4]-L*.md"))
print(f"Cleaning {len(files)} lesson files...")

cleaned_count = 0
for filepath in files:
    try:
        if clean_lesson_file(filepath):
            cleaned_count += 1
            if cleaned_count % 20 == 0:
                print(f"  Processed {cleaned_count} files...")
    except Exception as e:
        print(f"  [ERROR] {os.path.basename(filepath)}: {e}")

print(f"\nCleaned {cleaned_count} lesson files")
print(f"Skipped {len(files) - cleaned_count} files")
