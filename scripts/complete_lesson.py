"""
Complete a lesson using split generation - generates remaining sections
"""
import os
from anthropic import Anthropic

# Initialize client
api_key = os.getenv("ANTHROPIC_API_KEY")
if not api_key:
    raise ValueError("ANTHROPIC_API_KEY environment variable is required")

client = Anthropic(api_key=api_key)

# Read the incomplete lesson
lesson_path = "src/data/saas/lessons/M00-L001-the-saas-landscape-models-terms-and-trade-offs--2025-11-01.md"
with open(lesson_path, "r", encoding="utf-8") as f:
    existing_content = f.read()

# Extract the last few lines to provide context
last_section_context = existing_content.split("\n")[-20:]
context_preview = "\n".join(last_section_context)

# Continuation prompt
continuation_prompt = f"""You are completing a lesson that was partially generated. The lesson is about "The SaaS Landscape: Models, Terms, and Trade-offs" for Module M0 (SaaS Architecture & System Design), Foundation level.

The lesson so far has covered:
- Section 1: Header & Learning Objectives (complete)
- Section 2: Core Definition & Strategic Importance (complete)
- Section 3: Primary Concepts (Multi-Tenant, Single-Tenant, Hybrid) (complete)
- Section 4: Comparative Trade-Off Framework (complete)
- Section 5: Systematic Decision Framework (STARTED but incomplete)

The lesson currently ends at:
{context_preview}

Continue from where it left off and complete ALL remaining sections:

**Section 5: Systematic Decision Framework** (complete the remaining steps)
- Continue from Step 1: Identify Addressable Market
- Include Steps 2-6 with full branching logic and decision criteria

**Section 6: Real-World Consulting Scenarios** (3-4 scenarios)
For each scenario, include:
- Context & Constraints (business requirements, constraints, SLAs)
- Recommended Approach (which concept applies and why, alternatives rejected)
- Implementation Notes (concrete steps, timeline, pitfalls, success metrics)
- Example Trade-offs (what choice buys/costs)

**Section 7: Key Takeaways** (5-7 strategic takeaways)
Format: ✅ [Takeaway]: [1-2 sentence explanation]
- Ground in strategic principles
- Include one that challenges a common misconception
- Final takeaway should guide when to revisit this decision

**Section 8: Discussion & Application Questions** (exactly 3 questions)
- Question 1 (Recognition): Identify concept in real-world product/system
- Question 2 (Application): Apply decision framework to realistic scenario
- Question 3 (Reflection): Evaluate risks, mitigations, edge cases

**Section 9: Recommended Next Steps & Progression**
- Next Lessons: 2-3 related lessons with context
- Deeper Dive: Specific resources (docs, papers, tools) with descriptions
- Hands-On Practice: Mini-project or exercise suggestion
- Advanced Topics: Concepts for after mastery

**Section 10: Metadata & References**
- Lesson Metadata: Version, Last Updated, Audience, Domain, Complexity
- References & Further Reading: Official docs, research papers, community resources, tools

Important:
- Continue in the same style and tone as the existing content
- Maintain the same level of detail and concrete examples
- Include code examples where relevant (TypeScript, SQL, Terraform)
- Use concrete metrics and trade-offs
- Target total lesson length: 350-400 lines for Foundation level
- Write in professional prose, no placeholders

Generate the complete remaining content now, starting exactly where the lesson left off."""

print("Generating remaining sections (Part 2)...")
print("This may take a moment...")

# Make API call with streaming
with client.messages.stream(
    model="claude-haiku-4-5-20251001",
    max_tokens=4096,
    messages=[{
        "role": "user",
        "content": continuation_prompt
    }]
) as stream:
    continuation_parts = []
    for text in stream.text_stream:
        continuation_parts.append(text)
        print(".", end="", flush=True)
    print()

continuation_content = "".join(continuation_parts)

# Append to existing file
with open(lesson_path, "a", encoding="utf-8") as f:
    f.write("\n\n")
    f.write(continuation_content)

# Count total lines
with open(lesson_path, "r", encoding="utf-8") as f:
    total_lines = len(f.readlines())

print(f"Lesson completion generated!")
print(f"Continuation length: ~{len(continuation_content.split(chr(10)))} lines")
print(f"Total lesson lines: {total_lines}")
print(f"Appended to: {lesson_path}")

