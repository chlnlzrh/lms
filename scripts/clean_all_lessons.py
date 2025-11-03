"""
Clean ALL lesson files - remove LLM prompts and metadata sections
"""
import os
import re
import glob

lesson_dir = "src/data/saas/lessons"

def clean_lesson_file(filepath):
    """Remove LLM prompt section, keep only lesson content"""
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()
    
    original_len = len(content)
    
    # Pattern 1: Remove everything from "# LLM Prompt" until actual lesson starts
    # Find "# LLM Prompt" and remove everything until we hit lesson content
    llm_prompt_match = re.search(r'^# LLM Prompt:.*?\n', content, re.MULTILINE)
    if llm_prompt_match:
        # Find where actual lesson content starts after the prompt
        after_prompt = content[llm_prompt_match.end():]
        
        # Look for lesson start patterns
        lesson_start = None
        for pattern in [
            r'^# Lesson ',
            r'^## Lesson ',
            r'^# Section 1:',
            r'^## Section 1:',
            r'^### Section 1:',
            r'^Section 1:',
            r'^## \*\*Section 1:',
            r'^### \*\*Section 1:',
        ]:
            match = re.search(pattern, after_prompt, re.MULTILINE)
            if match:
                lesson_start = llm_prompt_match.end() + match.start()
                break
        
        if lesson_start:
            content = content[lesson_start:]
    
    # Pattern 2: Remove "## **Context & Parameters**" sections
    content = re.sub(r'^## \*\*Context & Parameters\*\*.*?\n(?=##|# Lesson|# Section)', '', content, flags=re.MULTILINE | re.DOTALL)
    
    # Pattern 3: Remove "## **Content Output Schema**" sections
    content = re.sub(r'^## \*\*Content Output Schema.*?\n(?=##|# Lesson|# Section)', '', content, flags=re.MULTILINE | re.DOTALL)
    
    # Pattern 4: Remove "Quality and governance notes" at the end
    content = re.sub(r'\n---\s*\nQuality and governance notes.*$', '', content, flags=re.DOTALL)
    content = re.sub(r'\nQuality and governance notes.*$', '', content, flags=re.DOTALL)
    
    # Pattern 5: Remove standalone prompt template sections (sections that are just templates, not actual content)
    content = re.sub(r'^### \*\*Section \d+.*?\*\*.*?\n(?=##|#)', '', content, flags=re.MULTILINE | re.DOTALL)
    
    # Pattern 6: Ensure content starts with lesson content, not prompt
    # If it starts with "### **Section" that's still prompt template, find actual content
    if re.match(r'^### \*\*Section \d+:.*?\*\*\s*\n', content):
        # Still prompt template, find actual lesson
        match = re.search(r'(^# Lesson .*|^# Section 1:.*|^## Section 1:.*|^Section 1:.*)', content, re.MULTILINE)
        if match:
            content = content[match.start():]
    
    # Pattern 7: Remove any remaining "LLM Prompt:" references
    content = re.sub(r'^# LLM Prompt:.*?\n', '', content, flags=re.MULTILINE)
    
    # Clean up leading whitespace
    content = content.lstrip()
    
    # If cleaned content is too short, keep original
    if len(content) < 500:
        return False
    
    # Write cleaned content
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)
    
    return True

# Process all lesson files
files = glob.glob(os.path.join(lesson_dir, "M*.md"))
print(f"Cleaning {len(files)} lesson files...")

cleaned_count = 0
skipped_count = 0
for filepath in sorted(files):
    try:
        if clean_lesson_file(filepath):
            cleaned_count += 1
            if cleaned_count % 20 == 0:
                print(f"  Processed {cleaned_count} files...")
        else:
            skipped_count += 1
            print(f"  [SKIPPED] {os.path.basename(filepath)} - content too short after cleaning")
    except Exception as e:
        print(f"  [ERROR] {os.path.basename(filepath)}: {e}")
        skipped_count += 1

print(f"\nCleaned {cleaned_count} lesson files")
print(f"Skipped {skipped_count} files")

