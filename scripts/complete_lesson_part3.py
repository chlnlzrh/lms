"""
Complete lesson part 3 - finish remaining sections
"""
import os
from anthropic import Anthropic

api_key = os.getenv("ANTHROPIC_API_KEY")
if not api_key:
    raise ValueError("ANTHROPIC_API_KEY environment variable is required")

client = Anthropic(api_key=api_key)

lesson_path = "src/data/saas/lessons/M00-L001-the-saas-landscape-models-terms-and-trade-offs--2025-11-01.md"
with open(lesson_path, "r", encoding="utf-8") as f:
    content = f.read()

# Get last context
last_context = "\n".join(content.split("\n")[-10:])

continuation_prompt = f"""Continue completing a lesson about "The SaaS Landscape: Models, Terms, and Trade-offs". 

The lesson currently ends mid-Section 6 (Real-World Consulting Scenarios) with an incomplete SQL code block:

{last_context}

Complete the remaining sections:

**Complete Section 6: Real-World Consulting Scenarios**
- Finish Scenario 2 (B2B SaaS Scheduling Tool) - complete the SQL schema and implementation
- Add Scenario 3: another realistic consulting scenario
- Add Scenario 4: one more scenario

**Section 7: Key Takeaways** (5-7 strategic takeaways)
Format: ✅ [Takeaway]: [explanation]

**Section 8: Discussion & Application Questions** (exactly 3 questions)
- Question 1 (Recognition)
- Question 2 (Application) 
- Question 3 (Reflection)

**Section 9: Recommended Next Steps & Progression**
- Next Lessons: 2-3 related lessons
- Deeper Dive: resources
- Hands-On Practice: exercise
- Advanced Topics: concepts

**Section 10: Metadata & References**
- Version: 1.0
- Last Updated: 2025-11-01
- Audience: Software engineers and architects
- Domain: SaaS platform development
- Complexity: Foundation [F]
- References: Official docs, papers, tools

Continue exactly where it left off, maintain the same style, include complete code examples, and finish all remaining sections."""

print("Generating final completion (Part 3)...")

with client.messages.stream(
    model="claude-haiku-4-5-20251001",
    max_tokens=4096,
    messages=[{"role": "user", "content": continuation_prompt}]
) as stream:
    parts = []
    for text in stream.text_stream:
        parts.append(text)
        print(".", end="", flush=True)
    print()

continuation = "".join(parts)

with open(lesson_path, "a", encoding="utf-8") as f:
    f.write(continuation)

with open(lesson_path, "r", encoding="utf-8") as f:
    total_lines = len(f.readlines())

print(f"\nCompletion added!")
print(f"Total lesson lines: {total_lines}")

