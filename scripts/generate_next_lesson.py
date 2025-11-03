"""
Generate the next single missing lesson
"""
import os
import sys
import re
import asyncio
from openai import AsyncOpenAI

# Fix Windows console encoding
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
    raise ValueError("OpenAI API key required")

client = AsyncOpenAI(api_key=api_key)
MODEL_NAME = "gpt-5-mini"

# Load lessons list
with open("scripts/m1_m2_m3_m4_lessons.txt", "r", encoding="utf-8") as f:
    m1_m4_content = f.read()

lessons_match = re.search(r'missing_lessons = \[(.*?)\]', m1_m4_content, re.DOTALL)
if lessons_match:
    lessons_str = "[" + lessons_match.group(1) + "]"
    m1_m4_lessons = eval(lessons_str)
else:
    m1_m4_lessons = []

# Check what's already generated
lesson_dir = "src/data/saas/lessons"
generated = set()
if os.path.exists(lesson_dir):
    files = [f for f in os.listdir(lesson_dir) if f.endswith('.md')]
    for f in files:
        match = re.match(r'(M[1-4]-L\d{3})', f)
        if match:
            generated.add(match.group(1))

remaining_lessons = [l for l in m1_m4_lessons if l['LESSON_CODE'] not in generated]

if not remaining_lessons:
    print("No remaining lessons to generate!")
    sys.exit(0)

next_lesson = remaining_lessons[0]
print(f"Next lesson to generate: {next_lesson['LESSON_CODE']} - {next_lesson['LESSON_TITLE']}")

# Import the generation function from the main script
sys.path.insert(0, 'scripts')
from generate_missing_lessons import generate_lesson

# Generate it
asyncio.run(generate_lesson(next_lesson, 1, 1))

