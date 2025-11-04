# File Manipulation & Batch Processing with LLMs

## Core Concepts

File manipulation and batch processing represent a fundamental shift in how we handle document transformation tasks. Traditional approaches require explicit parsing logic, format-specific libraries, and intricate transformation rules. LLMs treat files as semantic content rather than structured data, enabling transformations based on meaning rather than syntax.

### Traditional vs. LLM-Based Approach

```python
# Traditional approach: Extract key points from markdown files
import re
from pathlib import Path

def extract_key_points_traditional(markdown_file: Path) -> list[str]:
    """Extract bullet points under '## Key Points' heading."""
    with open(markdown_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Find the Key Points section
    pattern = r'## Key Points\n(.*?)(?=\n##|\Z)'
    match = re.search(pattern, content, re.DOTALL)
    
    if not match:
        return []
    
    # Extract bullet points
    section = match.group(1)
    points = re.findall(r'^\s*[-*]\s+(.+)$', section, re.MULTILINE)
    return points

# LLM approach: Extract semantic insights regardless of format
import anthropic
from typing import List

def extract_key_points_llm(markdown_file: Path, client: anthropic.Anthropic) -> List[str]:
    """Extract key insights using semantic understanding."""
    with open(markdown_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    message = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=1024,
        messages=[{
            "role": "user",
            "content": f"Extract the 3-5 most important insights from this document as a JSON array:\n\n{content}"
        }]
    )
    
    import json
    return json.loads(message.content[0].text)
```

The traditional approach breaks when:
- The heading format changes ("Key Takeaways" instead of "Key Points")
- Bullet points use different markers (numbers, letters)
- Important content exists outside the designated section
- The file format changes (PDF, DOCX instead of Markdown)

The LLM approach handles all these variations because it understands **content semantics** rather than **structural patterns**.

### Key Engineering Insights

**1. Content transformation becomes format-agnostic.** You write transformation logic once and it works across markdown, plain text, code files, even PDFs (when converted to text). The LLM handles format variations internally.

**2. Batch processing scales differently.** Traditional file processing scales linearly with CPU cores. LLM batch processing scales with API rate limits and token throughput. A 100-file job might complete in 30 seconds with proper concurrency vs. 2+ minutes sequential.

**3. Error handling shifts from syntax to semantics.** Traditional parsing fails on malformed input. LLMs degrade gracefully—they'll extract what they can and indicate uncertainty. Your error handling focuses on content quality rather than parse failures.

### Why This Matters Now

Organizations have massive document repositories that need transformation: documentation that needs summarization, code that needs explanation, reports that need standardization. Traditional approaches require custom parsers for each format and transformation. LLMs enable **universal content pipelines** where one codebase handles diverse inputs and outputs.

The economic shift is significant: a developer spending two weeks building format-specific parsers can now build one flexible pipeline in two days. But the engineering challenge shifts from "parse correctly" to "process efficiently at scale."

## Technical Components

### 1. Streaming File Content with Context Management

LLM APIs have token limits (typically 200k-1M input tokens). Large files require chunking strategies that preserve semantic context.

```python
from pathlib import Path
from typing import Iterator, TypedDict
import anthropic

class FileChunk(TypedDict):
    content: str
    file_path: Path
    chunk_index: int
    total_chunks: int

def chunk_file_with_overlap(
    file_path: Path,
    chunk_size: int = 100000,  # characters, ~25k tokens
    overlap: int = 2000  # overlap between chunks
) -> Iterator[FileChunk]:
    """Split large files into overlapping chunks to preserve context."""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    if len(content) <= chunk_size:
        yield FileChunk(
            content=content,
            file_path=file_path,
            chunk_index=0,
            total_chunks=1
        )
        return
    
    chunks = []
    start = 0
    chunk_index = 0
    
    while start < len(content):
        end = start + chunk_size
        chunk_content = content[start:end]
        
        chunks.append(FileChunk(
            content=chunk_content,
            file_path=file_path,
            chunk_index=chunk_index,
            total_chunks=0  # Will update after counting
        ))
        
        start = end - overlap  # Overlap for context
        chunk_index += 1
    
    # Update total_chunks count
    total = len(chunks)
    for chunk in chunks:
        chunk['total_chunks'] = total
        yield chunk

# Usage with progress tracking
def process_large_file(file_path: Path, client: anthropic.Anthropic) -> list[str]:
    """Process large file in chunks, maintaining context."""
    results = []
    previous_summary = ""
    
    for chunk in chunk_file_with_overlap(file_path):
        context = f"Previous context: {previous_summary}\n\n" if previous_summary else ""
        
        message = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=500,
            messages=[{
                "role": "user",
                "content": (
                    f"{context}"
                    f"Summarize key points from chunk {chunk['chunk_index'] + 1}/{chunk['total_chunks']}:\n\n"
                    f"{chunk['content']}"
                )
            }]
        )
        
        summary = message.content[0].text
        results.append(summary)
        previous_summary = summary  # Carry context forward
    
    return results
```

**Practical implications:** Overlap size trades off between context preservation and redundant processing costs. 2000 characters (≈500 tokens) typically captures enough context for continuity without significant waste.

**Real constraints:** Each chunk incurs API latency (100-500ms minimum). A 10-chunk file takes 1-5 seconds sequential, vs. 300ms-1s with concurrent processing. Choose chunking strategy based on latency requirements.

### 2. Concurrent Batch Processing with Rate Limiting

Processing multiple files sequentially wastes time waiting for API responses. Proper concurrency can provide 5-10x throughput improvement.

```python
import asyncio
from pathlib import Path
from typing import List, Callable, Any
import anthropic
from dataclasses import dataclass
import time

@dataclass
class ProcessingResult:
    file_path: Path
    result: Any
    duration: float
    error: Exception | None = None

class BatchProcessor:
    """Process multiple files concurrently with rate limiting."""
    
    def __init__(
        self,
        client: anthropic.Anthropic,
        max_concurrent: int = 5,
        requests_per_minute: int = 50
    ):
        self.client = client
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.rate_limiter = asyncio.Semaphore(requests_per_minute)
        self.rpm = requests_per_minute
        
    async def process_file(
        self,
        file_path: Path,
        transform_fn: Callable[[str, anthropic.Anthropic], Any]
    ) -> ProcessingResult:
        """Process single file with rate limiting."""
        start_time = time.time()
        
        async with self.semaphore:  # Limit concurrent requests
            try:
                # Read file
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Rate limiting
                async with self.rate_limiter:
                    result = transform_fn(content, self.client)
                    
                    # Refill rate limiter token after 60s
                    asyncio.create_task(self._refill_rate_limit())
                
                duration = time.time() - start_time
                return ProcessingResult(
                    file_path=file_path,
                    result=result,
                    duration=duration
                )
                
            except Exception as e:
                duration = time.time() - start_time
                return ProcessingResult(
                    file_path=file_path,
                    result=None,
                    duration=duration,
                    error=e
                )
    
    async def _refill_rate_limit(self):
        """Release rate limiter token after delay."""
        await asyncio.sleep(60.0 / self.rpm)
        self.rate_limiter.release()
    
    async def process_batch(
        self,
        file_paths: List[Path],
        transform_fn: Callable[[str, anthropic.Anthropic], Any]
    ) -> List[ProcessingResult]:
        """Process multiple files concurrently."""
        tasks = [
            self.process_file(path, transform_fn)
            for path in file_paths
        ]
        return await asyncio.gather(*tasks)

# Example transformation function
def summarize_content(content: str, client: anthropic.Anthropic) -> str:
    """Summarize file content."""
    message = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=300,
        messages=[{
            "role": "user",
            "content": f"Summarize this in 2-3 sentences:\n\n{content[:50000]}"
        }]
    )
    return message.content[0].text

# Usage
async def main():
    client = anthropic.Anthropic()
    processor = BatchProcessor(
        client=client,
        max_concurrent=5,
        requests_per_minute=50
    )
    
    files = list(Path("./documents").glob("*.md"))
    results = await processor.process_batch(files, summarize_content)
    
    # Report results
    successful = [r for r in results if r.error is None]
    failed = [r for r in results if r.error is not None]
    
    print(f"Processed {len(successful)}/{len(results)} files successfully")
    print(f"Total time: {sum(r.duration for r in results):.2f}s")
    print(f"Avg time per file: {sum(r.duration for r in results)/len(results):.2f}s")

# Run with: asyncio.run(main())
```

**Practical implications:** `max_concurrent=5` typically provides optimal throughput for most API rate limits without triggering 429 errors. Higher values don't improve throughput once you hit rate limits.

**Trade-offs:** Aggressive concurrency reduces total processing time but increases peak memory usage (more files loaded simultaneously). For 1000 large files, process in batches of 50-100 to avoid memory issues.

### 3. Result Aggregation and Quality Validation

Individual file processing is only half the challenge. Aggregating results consistently and validating output quality prevents downstream failures.

```python
from typing import TypedDict, List, Literal
from pathlib import Path
import json
from dataclasses import dataclass
import anthropic

class ValidationResult(TypedDict):
    is_valid: bool
    confidence: float
    issues: List[str]

@dataclass
class AggregatedResult:
    """Container for validated batch results."""
    total_files: int
    successful: int
    failed: int
    results: List[dict]
    validation_report: dict

class ResultAggregator:
    """Aggregate and validate batch processing results."""
    
    def __init__(self, client: anthropic.Anthropic):
        self.client = client
    
    def validate_output(
        self,
        content: str,
        expected_format: Literal["json", "markdown", "text"]
    ) -> ValidationResult:
        """Validate LLM output meets quality criteria."""
        issues = []
        
        if expected_format == "json":
            try:
                parsed = json.loads(content)
                is_valid = isinstance(parsed, (dict, list))
                if not is_valid:
                    issues.append("JSON is not object or array")
            except json.JSONDecodeError as e:
                is_valid = False
                issues.append(f"Invalid JSON: {e}")
        
        elif expected_format == "markdown":
            # Check for markdown structure
            is_valid = any([
                content.startswith("#"),
                "##" in content,
                "* " in content or "- " in content
            ])
            if not is_valid:
                issues.append("No markdown formatting detected")
        
        else:  # text
            is_valid = len(content.strip()) > 0
            if not is_valid:
                issues.append("Empty output")
        
        # Check for common LLM failures
        if "I cannot" in content or "I'm unable" in content:
            issues.append("LLM refused task")
            is_valid = False
        
        if len(content) < 10:
            issues.append("Output too short")
            is_valid = False
        
        confidence = 1.0 if is_valid and not issues else 0.5
        
        return ValidationResult(
            is_valid=is_valid,
            confidence=confidence,
            issues=issues
        )
    
    def aggregate_results(
        self,
        results: List[ProcessingResult],
        expected_format: Literal["json", "markdown", "text"] = "text"
    ) -> AggregatedResult:
        """Aggregate results with validation."""
        validated_results = []
        validation_summary = {
            "total_validated": 0,
            "passed": 0,
            "failed": 0,
            "common_issues": {}
        }
        
        for result in results:
            if result.error is None:
                validation = self.validate_output(result.result, expected_format)
                validated_results.append({
                    "file": str(result.file_path),
                    "content": result.result,
                    "validation": validation,
                    "duration": result.duration
                })
                
                validation_summary["total_validated"] += 1
                if validation["is_valid"]:
                    validation_summary["passed"] += 1
                else:
                    validation_summary["failed"] += 1
                    
                # Track common issues
                for issue in validation["issues"]:
                    validation_summary["common_issues"][issue] = \
                        validation_summary["common_issues"].get(issue, 0) + 1
        
        successful = sum(1 for r in results if r.error is None)
        failed = sum(1 for r in results if r.error is not None)
        
        return AggregatedResult(
            total_files=len(results),
            successful=successful,
            failed=failed,
            results=validated_results,
            validation_report=validation_summary
        )
    
    def save_results(self, aggregated: AggregatedResult, output_path: Path):
        """Save aggregated results with validation report."""
        output_data = {
            "summary": {
                "total_files": aggregated.total_files,
                "successful": aggregated.successful,
                "failed": aggregated.failed,
                "validation": aggregated.validation_report
            },
            "results": aggregated.results
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"\nResults saved to {output_path}")
        print(f"Validation: {aggregated.validation_report['passed']}/{aggregated.validation_report['total_validated']} passed")
```

**Practical implications:** Validation catches common LLM failure modes (refusals, empty outputs, format violations) before they cause downstream pipeline failures. Add validation rules specific to your use case.

**Real constraints:** Validation adds 1