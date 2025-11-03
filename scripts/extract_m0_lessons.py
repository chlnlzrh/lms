"""
Extract all M0 lessons from L017 onwards
"""
import re

with open("src/data/saas/content_structure_ai-native-saas-curriculum-lesson-maps.md", "r", encoding="utf-8") as f:
    content = f.read()

# Extract M0 section
m0_match = re.search(r'# M0:.*?(?=# M1:|$)', content, re.DOTALL)
if not m0_match:
    print("M0 section not found")
    exit(1)

m0_section = m0_match.group(0)
lines = m0_section.split('\n')

lessons = []
current_focus = "Core Architectural Patterns"  # Default

for line in lines:
    # Check for section headers (these define the focus area)
    section_match = re.match(r'^## ([^L].+)$', line)
    if section_match and not section_match.group(1).startswith('L'):
        current_focus = section_match.group(1)
        continue
    
    # Check for lesson headers
    lesson_match = re.match(r'^## L(\d+): (.+?) \[([FIAE])\]', line)
    if lesson_match:
        num = int(lesson_match.group(1))
        title = lesson_match.group(2)
        complexity = lesson_match.group(3)
        lessons.append((num, title, complexity, current_focus))

# Filter lessons >= L017
remaining = [l for l in lessons if l[0] >= 17]

print(f"Found {len(remaining)} lessons from L017 onwards:")
for num, title, comp, focus in remaining:
    print(f"  L{num:03d}: {title} [{comp}] - {focus}")

