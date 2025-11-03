# Design Thinking: OpenAI Batch API for Lesson Generation

## Problem Statement

Generate lessons for Data Governance curriculum (`Content Structure.md`) using OpenAI Batch API instead of sequential/parallel API calls.

## Requirements

1. **Don't change existing Python programs** (can reference/take initial copy)
2. Use **OpenAI Batch API** (same model as existing scripts)
3. Generate lessons for Data Governance curriculum
4. Handle all lessons (M0-M14, L001-L999)

## Design Approach

### Architecture Decision: Batch API vs Async API

**Why Batch API?**

1. **Cost Efficiency**: 50% discount on OpenAI API calls
   - Standard: ~$0.15 per lesson
   - Batch: ~$0.075 per lesson
   - For 336 lessons: ~$25 vs ~$50

2. **Scalability**: Handle hundreds of lessons without rate limits
   - No need for complex rate limiting logic
   - No need for retry mechanisms
   - Built-in reliability

3. **Asynchronous Processing**: Submit batch and process results when ready
   - No need to wait for completion
   - Can check status and resume later
   - Better for long-running operations

4. **Built-in Retries**: Failed requests automatically retried
   - No custom error handling needed
   - Higher success rate

### Workflow Design

```
┌─────────────────────────────────────┐
│  1. Parse Content Structure.md      │
│     Extract all lessons (M0-M14)    │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│  2. Create JSONL Input File          │
│     One request per lesson           │
│     Format: OpenAI Batch API spec   │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│  3. Upload JSONL to OpenAI           │
│     Purpose: 'batch'                │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│  4. Create Batch Job                │
│     Completion window: 24h          │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│  5. Poll for Completion             │
│     Check status periodically       │
│     (can be interrupted/resumed)    │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│  6. Download Results                │
│     Get output file from batch      │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│  7. Process & Save Lessons          │
│     Extract content from results    │
│     Clean and save to files         │
└─────────────────────────────────────┘
```

### Implementation Strategy

#### 1. Parsing Content Structure

- **Pattern Matching**: Regex to extract module headers (`# M0: Title — Lesson Map`)
- **Lesson Extraction**: Regex to extract lesson headers (`## L001: Title [F]`)
- **Metadata Extraction**: Module code, title, lesson code, title, complexity

**Key Design Decision**: Use regex patterns similar to existing `count_datagov_lessons.py`

#### 2. Prompt Generation

- **Reuse Existing Template**: Adapt `LESSON_GENERATION_PROMPT_GENERIC.md`
- **Adapt for Data Governance**: Change context from "SaaS Curriculum" to "Data Governance Curriculum"
- **Default Values**: Use reasonable defaults for prerequisites/related lessons (can be enhanced later)

**Key Design Decision**: Reference existing `generate_lesson()` function logic from `generate_missing_lessons.py`

#### 3. JSONL Format

- **OpenAI Batch API Spec**: Each line is a JSON object with:
  - `custom_id`: Unique identifier for tracking
  - `method`: "POST"
  - `url`: "/v1/chat/completions"
  - `body`: Chat completion request (model, messages, max_completion_tokens)

**Key Design Decision**: Follow OpenAI Batch API specification exactly

#### 4. Batch Job Management

- **Upload**: Use `client.files.create()` with purpose='batch'
- **Create Batch**: Use `client.batches.create()` with completion_window="24h"
- **Poll Status**: Check `batch.status` periodically
- **Download**: Use `client.files.content()` to download results

**Key Design Decision**: Use synchronous polling (can be enhanced with async/resume capability)

#### 5. Results Processing

- **Parse JSONL**: Each line is a result object
- **Extract Content**: Get `response.body.choices[0].message.content`
- **Clean Content**: Remove prompt artifacts (similar to existing scripts)
- **Save Files**: Write to `src/data/data_gov/lessons/M{module}-L{lesson}.md`

**Key Design Decision**: Reuse content cleaning logic from existing scripts

### File Structure

```
scripts/
├── generate_datagov_lessons_batch.py    # Main script
├── check_batch_status.py                # Helper to check status
├── process_batch_results.py             # Process downloaded results
└── batch_api/                           # Batch API working directory
    ├── datagov_lessons_input.jsonl      # Input file (created)
    └── datagov_lessons_results.jsonl    # Results file (downloaded)

src/data/data_gov/
├── Content Structure.md                 # Source file
└── lessons/                             # Output directory
    ├── M0-L001.md
    ├── M0-L002.md
    └── ...
```

### Model Selection

**Default**: `gpt-4o-mini` (Batch API compatible)

**Note**: Existing scripts use `gpt-5-mini` which may not exist. Adjust in script if needed:

```python
MODEL_NAME = "gpt-4o-mini"  # Batch API compatible
```

### Error Handling

1. **API Key**: Check `openaiapikey.txt` or environment variable
2. **File Not Found**: Validate all required files exist
3. **Batch Status**: Handle all statuses (validating, in_progress, completed, failed, etc.)
4. **Results Parsing**: Handle JSON decode errors gracefully
5. **Content Cleaning**: Handle missing content gracefully

### Optimization Opportunities

1. **Resume Capability**: Save batch ID and resume later
2. **Incremental Processing**: Process results as they become available
3. **Parallel Processing**: Process results file in parallel (future enhancement)
4. **Caching**: Cache lessons map to avoid re-parsing
5. **Progress Tracking**: Better progress indicators for large batches

### Testing Strategy

1. **Small Batch Test**: Test with 1-5 lessons first
2. **Parsing Test**: Verify all lessons extracted correctly
3. **JSONL Validation**: Verify JSONL format is correct
4. **Results Processing**: Verify content extraction and cleaning
5. **File Output**: Verify saved files are correct

## Benefits vs Existing Approach

| Aspect | Async API (Existing) | Batch API (New) |
|--------|---------------------|-----------------|
| Cost | ~$0.15/lesson | ~$0.075/lesson (50% off) |
| Rate Limits | Need to handle | Built-in |
| Retries | Manual handling | Automatic |
| Scalability | Limited by rate limits | Handles hundreds easily |
| Wait Time | Wait for completion | Submit and resume later |
| Reliability | Custom retry logic | Built-in reliability |

## Trade-offs

### Advantages
- ✅ 50% cost savings
- ✅ Better scalability
- ✅ Built-in reliability
- ✅ Asynchronous processing

### Disadvantages
- ⚠️ Results available after completion (hours/days)
- ⚠️ 24-hour window to download results
- ⚠️ Less real-time feedback
- ⚠️ Need to poll for status

## Future Enhancements

1. **Resume Capability**: Save/load batch state
2. **Incremental Processing**: Process results as available
3. **Better Error Reporting**: Detailed error logs
4. **Progress Tracking**: Better progress indicators
5. **Batch Splitting**: Split large batches into smaller ones
6. **Model Selection**: Support multiple models
7. **Cost Estimation**: Pre-calculate estimated costs

## Conclusion

The Batch API approach provides significant cost savings and better scalability for generating large volumes of lessons. The workflow is well-suited for this use case where immediate results are not required, and the 50% discount makes it economically favorable.

The implementation follows existing patterns from the codebase while adapting for Batch API specifics, ensuring consistency and maintainability.

