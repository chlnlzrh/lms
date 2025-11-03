# Curriculum Builder Agent

An agent that generates comprehensive curriculum lesson maps for any subject using Claude Haiku 4.5 Model.

## Overview

The Curriculum Builder Agent:
- Uses Claude Haiku 4.5 (`claude-haiku-4-5-20251001`) via Anthropic API
- Generates structured lesson maps with 15 sections, each containing 3 subsections with 8 lessons
- Automatically creates subject folders with lowercase abbreviations
- Saves curriculum to `src/data/{subject_short}/Content Structure.md`

## API Endpoint

**POST** `/api/curriculum-builder`

### Request Body
```json
{
  "subject": "Your Subject Name"
}
```

### Response
```json
{
  "success": true,
  "subject": "Machine Learning",
  "folderName": "ml",
  "path": "src/data/ml/Content Structure.md",
  "fileExists": false,
  "overwritten": false,
  "usage": {
    "input_tokens": 1234,
    "output_tokens": 5678
  }
}
```

## Usage

### 1. Environment Setup

Ensure `ANTHROPIC_API_KEY` is set in your environment or `.env.local`:

```bash
# .env.local
ANTHROPIC_API_KEY=your_api_key_here
```

Or the API will read from `claudeAPIkey.txt` if present (though using env vars is recommended).

### 2. Start the Development Server

```bash
npm run dev
```

The API will be available at `http://localhost:3000/api/curriculum-builder`

### 3. Call the API

#### Using curl:
```bash
curl -X POST http://localhost:3000/api/curriculum-builder \
  -H "Content-Type: application/json" \
  -d '{"subject": "Machine Learning"}'
```

#### Using Python test script:
```bash
python scripts/test_curriculum_builder.py "Machine Learning"
```

#### Using JavaScript/TypeScript:
```typescript
const response = await fetch('/api/curriculum-builder', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ subject: 'Machine Learning' })
})
const result = await response.json()
```

## Folder Name Generation

The agent automatically generates lowercase folder names:

- **Data Engineering** → `de`
- **Machine Learning** → `ml`
- **Salesforce** → `sfdc`
- **Data Governance** → `datagov`
- **Master Data Management** → `mdm`
- **Robotic Process Automation** → `rpa`
- **Single Word Subject** → First 6 consonants or first 6 chars

## Output Format

The generated curriculum follows this structure:

```
Complexity Legend

[F] Foundational — Basic concepts, prerequisites
[I] Intermediate — Practical application, integration of concepts
[A] Advanced — Complex problem-solving, experience-dependent
[E] Expert — Specialized knowledge, architectural decisions

1) Subject Fundamentals & Landscape — Lesson Map

1.1 Core Concepts & Vocabulary

1.1.1 Example Lesson Title A [F]
1.1.2 Example Lesson Title B [F]
... (8 lessons per subsection)

1.2 Second Subsection Title
... (3 subsections per section)

... (15 sections total)

15) Capstone — Role-Based Tracks — Lesson Map
15.1 Foundations Track (New Grad → Junior)
15.2 Technical Track (Mid → Senior)
15.3 Architect/Strategic Track (Staff → Principal)
```

## Curriculum Structure

- **15 Sections**: Each covers a major topic area
- **3 Subsections per Section**: Breaking down topics into focused areas
- **8 Lessons per Subsection**: Specific, actionable lessons
- **Complexity Markers**: Each lesson tagged [F], [I], [A], or [E]
- **Capstone Section**: Role-based tracks for different career levels

## Error Handling

The API returns appropriate HTTP status codes:

- **400**: Invalid request (missing or empty subject)
- **500**: Server error (file system, generation failures)
- **503**: Service unavailable (API key issues)

## Notes

- If a `Content Structure.md` already exists, it will be overwritten
- Generation typically takes 30-120 seconds depending on complexity
- The API uses `max_tokens: 32768` to ensure complete curriculum generation
- All output is saved in UTF-8 encoding

## Examples

### Generate curriculum for "Cloud Security"
```bash
curl -X POST http://localhost:3000/api/curriculum-builder \
  -H "Content-Type: application/json" \
  -d '{"subject": "Cloud Security"}'
```

Result: Creates `src/data/cloudsec/Content Structure.md` (or similar abbreviation)

### Generate curriculum for "DevOps"
```bash
python scripts/test_curriculum_builder.py "DevOps"
```

Result: Creates `src/data/devops/Content Structure.md`




