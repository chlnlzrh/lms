"""
Finalize the lesson - complete remaining sections with full 64K token limit
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

# Remove the incomplete ending
if "CREATE TABLE tenants_brokerage" in content:
    # Find where it starts cutting off
    lines = content.split("\n")
    for i, line in enumerate(lines):
        if "CREATE TABLE tenants_brokerage" in line and i > 1700:
            # Keep everything up to but not including this incomplete section
            content = "\n".join(lines[:i])
            break

# Get last context
last_100_lines = "\n".join(content.split("\n")[-100:])

completion_prompt = f"""Complete a lesson about "The SaaS Landscape: Models, Terms, and Trade-offs" (M00-L001, Foundation level).

The lesson currently ends around Scenario 3 or 4 in Section 6. Here's the ending context:

{last_100_lines}

Complete ALL remaining sections with full detail:

**Complete Section 6: Real-World Consulting Scenarios**
- Finish any incomplete scenario
- Ensure you have 3-4 complete scenarios total
- Each scenario must include: Context & Constraints, Recommended Approach, Implementation Notes (with code examples), Pitfalls, Success Metrics, Trade-offs

**Section 7: Key Takeaways** (5-7 strategic takeaways)
Format each as: ✅ [Memorable takeaway]: [1-2 sentence explanation]
- Ground in strategic business/architectural principles
- Include one that challenges a common misconception
- Final takeaway should guide when to revisit this decision

**Section 8: Discussion & Application Questions** (exactly 3 questions)
- **Question 1 (Recognition):** Ask learner to identify the concept in a real-world product/system they know, probe for evidence-based reasoning
- **Question 2 (Application):** Present realistic scenario requiring application of decision framework with trade-off analysis
- **Question 3 (Reflection):** Ask learner to evaluate risks, mitigations, or edge cases beyond textbook answers

**Section 9: Recommended Next Steps & Progression**
- **Next Lessons:** Link to 2-3 related lessons (M00-L002: Tenancy Isolation 101, M00-L003: Choosing a Tenancy Model) with brief context
- **Deeper Dive:** Specific resources (official docs, research papers, tools) with 1-sentence description of why each matters
- **Hands-On Practice:** Suggest a mini-project or exercise (e.g., "Build a multi-tenant prototype with RLS")
- **Advanced Topics:** Introduce concepts for after mastery (hybrid architectures, migration strategies, compliance frameworks)

**Section 10: Metadata & References**
**Lesson Metadata:**
- Version: 1.0
- Last Updated: 2025-11-01
- Audience: Software engineers and architects building or evaluating SaaS platforms
- Domain: SaaS platform development, multi-tenant systems
- Complexity: Foundation [F]

**References & Further Reading:**
- Official documentation links (specific pages, not homepages) with context
- Research papers or case studies with 1-sentence context
- Community resources or best practices guides
- Tools or platforms referenced (PostgreSQL RLS docs, Terraform modules, etc.)

Continue exactly where the lesson left off. Maintain professional prose, include code examples where relevant, use concrete metrics, and ensure all 10 sections are complete. Target Foundation level length: 350-400 lines total (we're completing the final sections)."""

print("Completing remaining sections with 64K token limit...")

try:
    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=64000,
        messages=[{"role": "user", "content": completion_prompt}]
    )
    completion = response.content[0].text
    print("Generated via non-streaming")
except:
    print("Using streaming...")
    with client.messages.stream(
        model="claude-haiku-4-5-20251001",
        max_tokens=64000,
        messages=[{"role": "user", "content": completion_prompt}]
    ) as stream:
        parts = []
        for text in stream.text_stream:
            parts.append(text)
            print(".", end="", flush=True)
        print()
    completion = "".join(parts)

# Append completion
with open(lesson_path, "a", encoding="utf-8") as f:
    f.write("\n\n")
    f.write(completion)

with open(lesson_path, "r", encoding="utf-8") as f:
    total_lines = len(f.readlines())

print(f"\nLesson completion added!")
print(f"Total lesson lines: {total_lines}")
print(f"File: {lesson_path}")

