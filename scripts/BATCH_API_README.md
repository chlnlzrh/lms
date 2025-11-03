# OpenAI Batch API for Lesson Generation

## Overview

This script uses OpenAI's **Batch API** to generate Data Governance lessons in bulk. The Batch API is **50% cheaper** than standard API calls and handles large volumes efficiently without rate limits.

## Design Thinking

### Why Batch API?

1. **Cost Efficiency**: 50% discount on OpenAI API calls
2. **Scalability**: Process hundreds of lessons without rate limit concerns
3. **Reliability**: Built-in retries for failed requests
4. **Asynchronous**: Submit batch and process results when ready

### Architecture

```
Content Structure.md
    ↓
Parse & Extract Lessons (M0-M14, L001-L999)
    ↓
Create JSONL File (one request per lesson)
    ↓
Upload to OpenAI
    ↓
Create Batch Job
    ↓
Poll for Completion (hours/days)
    ↓
Download Results
    ↓
Process & Save Lessons to src/data/data_gov/lessons/
```

### Workflow

1. **Parse** `src/data/data_gov/Content Structure.md` to extract all lessons
2. **Create JSONL** file with OpenAI chat completion requests (one per lesson)
3. **Upload** JSONL file to OpenAI
4. **Create** batch job
5. **Poll** for completion status
6. **Download** results when complete
7. **Process** and save each lesson as `M{module}-L{lesson}.md`

## Files

- `scripts/generate_datagov_lessons_batch.py` - Main script for batch generation
- `scripts/check_batch_status.py` - Helper to check batch status
- `scripts/process_batch_results.py` - Process downloaded results independently

## Usage

### 1. Generate Lessons (Full Workflow)

```bash
python scripts/generate_datagov_lessons_batch.py
```

This will:
- Parse Content Structure.md
- Filter out already-generated lessons
- Create JSONL input file
- Upload to OpenAI
- Create batch job
- Poll until completion (may take hours)
- Download and process results

### 2. Check Batch Status

If you need to check status manually or resume:

```bash
# Check status
python scripts/check_batch_status.py <batch_id>

# Check status and download results
python scripts/check_batch_status.py <batch_id> --download
```

### 3. Process Results Separately

If you've already downloaded results:

```bash
python scripts/process_batch_results.py <results.jsonl>
```

## Configuration

### Model

Default: `gpt-4o-mini` (Batch API compatible)

To change, edit `MODEL_NAME` in `generate_datagov_lessons_batch.py`:

```python
MODEL_NAME = "gpt-4o-mini"  # Change to your preferred model
```

### Paths

- **Content Structure**: `src/data/data_gov/Content Structure.md`
- **Prompt Template**: `LESSON_GENERATION_PROMPT_GENERIC.md`
- **Output Directory**: `src/data/data_gov/lessons/`
- **Batch Files**: `scripts/batch_api/`

## Batch Job Lifecycle

1. **validating** - Initial validation
2. **in_progress** - Processing requests
3. **finalizing** - Finalizing results
4. **completed** - Ready for download
5. **expired** - Results expired (24h window)
6. **cancelled** - Manually cancelled
7. **failed** - Batch failed

## Cost Estimation

- **Standard API**: ~$0.15 per lesson (estimated)
- **Batch API**: ~$0.075 per lesson (50% discount)
- **For 336 lessons**: ~$25.20 vs ~$50.40

## Troubleshooting

### Batch Stuck in "in_progress"

- Normal for large batches (hundreds of lessons)
- Can take hours or days depending on queue
- Check periodically with `check_batch_status.py`

### Results Expired

- Batch results expire after 24 hours
- Download immediately when completed
- Can't re-download expired results

### Missing Lessons

- Check `failed_count` in results summary
- Review error messages in results file
- Re-run failed lessons individually if needed

## Notes

- **Existing scripts unchanged**: This is a new implementation, existing Python programs are not modified
- **Same model**: Uses the same model configuration as existing scripts (adjustable)
- **Filtering**: Automatically skips already-generated lessons
- **Cleanup**: Post-processes content to remove prompt artifacts

## Example Output

```
================================================================================
Data Governance Lessons - Batch API Generator
================================================================================

Step 1: Parsing Content Structure.md...
✓ Found 336 lessons across 15 modules

Already generated: 0
Remaining to generate: 336

Step 2: Creating JSONL input file...
================================================================================
Creating JSONL input file: scripts/batch_api/datagov_lessons_input.jsonl
Total lessons to process: 336
================================================================================

  Prepared 10/336 lessons...
  Prepared 20/336 lessons...
  ...
✓ Created JSONL file with 336 requests
  File: scripts/batch_api/datagov_lessons_input.jsonl
  Size: 1250.3 KB

Step 3: Uploading JSONL file to OpenAI...
✓ Uploaded: file-abc123xyz

Step 4: Creating batch job...
✓ Batch created: batch_abc123xyz
  Status: validating

================================================================================
Batch Job Information:
  Batch ID: batch_abc123xyz
  Status: validating
  Input file: file-abc123xyz

You can check status with:
  batch = client.batches.retrieve('batch_abc123xyz')
================================================================================

Step 5: Waiting for batch completion...
...
```

