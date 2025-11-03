"""Check which M1-M4 lessons have already been generated"""
import os
import re

# Get all generated lesson files
lesson_dir = "src/data/saas/lessons"
if not os.path.exists(lesson_dir):
    generated = set()
else:
    files = [f for f in os.listdir(lesson_dir) if f.endswith('.md')]
    generated = set()
    for f in files:
        # Extract lesson code (e.g., M1-L001 from filename)
        match = re.match(r'(M[1-4]-L\d{3})', f)
        if match:
            generated.add(match.group(1))

print(f"Already generated: {len(generated)} lessons")
if generated:
    print("Generated lessons:")
    for code in sorted(generated):
        print(f"  {code}")

# Load all M1-M4 lessons
with open("scripts/m1_m2_m3_m4_lessons.txt", "r", encoding="utf-8") as f:
    content = f.read()

match = re.search(r'missing_lessons = \[(.*?)\]', content, re.DOTALL)
if match:
    lessons_str = "[" + match.group(1) + "]"
    all_lessons = eval(lessons_str)
    
    # Filter out already generated
    remaining = [l for l in all_lessons if l['LESSON_CODE'] not in generated]
    
    print(f"\nRemaining to generate: {len(remaining)} lessons")
    print(f"Breakdown by module:")
    counts = {}
    for lesson in remaining:
        mod = lesson['MODULE_CODE']
        counts[mod] = counts.get(mod, 0) + 1
    for mod in ['M1', 'M2', 'M3', 'M4']:
        print(f"  {mod}: {counts.get(mod, 0)} lessons")
    
    # Save remaining lessons
    import json
    print(f"\nFirst remaining: {remaining[0]['LESSON_CODE'] if remaining else 'None'}")
    print(f"Last remaining: {remaining[-1]['LESSON_CODE'] if remaining else 'None'}")

