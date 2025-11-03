"""
Count lessons in the curriculum structure
"""
import re

with open("src/data/saas/content_structure_ai-native-saas-curriculum-lesson-maps.md", "r", encoding="utf-8") as f:
    content = f.read()

# Find all lesson markers
lessons = re.findall(r'^## L\d{3}:', content, re.MULTILINE)

# Find all module headers
modules = re.findall(r'^# M\d+:', content, re.MULTILINE)

# Count by module
lines = content.split('\n')
module_lessons = {}
current_module = None

for line in lines:
    # Check for module header
    m_match = re.match(r'^# (M\d+):', line)
    if m_match:
        current_module = m_match.group(1)
        if current_module not in module_lessons:
            module_lessons[current_module] = []
    
    # Check for lesson header
    l_match = re.match(r'^## (L\d{3}): (.+)', line)
    if l_match and current_module:
        lesson_code = l_match.group(1)
        lesson_title = l_match.group(2)
        module_lessons[current_module].append((lesson_code, lesson_title))

print("="*80)
print("LESSON COUNT BY MODULE")
print("="*80)
print()

total = 0
for mod in sorted(module_lessons.keys()):
    count = len(module_lessons[mod])
    total += count
    print(f"{mod}: {count} lessons")

print()
print("="*80)
print(f"TOTAL LESSONS: {total}")
print(f"TOTAL MODULES: {len(modules)}")
print("="*80)

