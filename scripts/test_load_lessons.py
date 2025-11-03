"""Test loading M1-M4 lessons"""
import re

with open("scripts/m1_m2_m3_m4_lessons.txt", "r", encoding="utf-8") as f:
    content = f.read()

match = re.search(r'missing_lessons = \[(.*?)\]', content, re.DOTALL)
if match:
    lessons_str = "[" + match.group(1) + "]"
    lessons = eval(lessons_str)
    print(f"[SUCCESS] Loaded {len(lessons)} lessons")
    print(f"  First: {lessons[0]['LESSON_CODE']}")
    print(f"  Last: {lessons[-1]['LESSON_CODE']}")
    
    # Count by module
    counts = {}
    for lesson in lessons:
        mod = lesson.get('MODULE_CODE', 'Unknown')
        counts[mod] = counts.get(mod, 0) + 1
    print(f"\nBreakdown by module:")
    for mod, count in sorted(counts.items()):
        print(f"  {mod}: {count} lessons")
else:
    print("[ERROR] Failed to extract lessons")

