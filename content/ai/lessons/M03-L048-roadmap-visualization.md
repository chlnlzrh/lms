# Roadmap Visualization: Strategic Planning with LLM-Enhanced Workflows

## Core Concepts

Roadmap visualization transforms abstract planning data into structured visual representations that communicate direction, dependencies, and timelines. For engineers building AI-powered systems, the fundamental shift isn't just automating diagram generation—it's using LLMs to extract structured planning data from unstructured sources, maintain consistency across evolving requirements, and generate multiple visualization formats from a single source of truth.

### Traditional vs. LLM-Enhanced Approach

**Traditional roadmap creation:**

```python
# Manual extraction and structuring
roadmap_data = {
    "Q1": [
        {"feature": "User Authentication", "team": "Backend", "priority": "high"},
        {"feature": "API Rate Limiting", "team": "Backend", "priority": "medium"}
    ],
    "Q2": [
        {"feature": "Mobile App", "team": "Frontend", "priority": "high"}
    ]
}

# Manual diagram generation with hardcoded layout
def generate_gantt_chart(data):
    # 50+ lines of layout logic
    # Brittle to changes
    # Single output format
    pass
```

**LLM-enhanced approach:**

```python
from typing import List, Dict, Optional
import json
from dataclasses import dataclass, asdict
from datetime import datetime

@dataclass
class RoadmapItem:
    title: str
    timeline: str
    dependencies: List[str]
    team: str
    priority: str
    status: str

class RoadmapExtractor:
    """Extract structured roadmap data from unstructured text."""
    
    def __init__(self, llm_client):
        self.client = llm_client
    
    def extract_from_text(self, planning_doc: str) -> List[RoadmapItem]:
        """Convert meeting notes, docs into structured roadmap."""
        prompt = f"""Extract roadmap items from this planning document.
Return JSON array with: title, timeline (Q1/Q2/etc), dependencies (array), 
team, priority (high/medium/low), status (planned/in-progress/completed).

Document:
{planning_doc}

Output only valid JSON array."""

        response = self.client.generate(prompt, temperature=0.1)
        items_data = json.loads(response)
        
        return [RoadmapItem(**item) for item in items_data]
    
    def generate_mermaid(self, items: List[RoadmapItem]) -> str:
        """Generate Mermaid diagram from structured data."""
        items_json = json.dumps([asdict(item) for item in items], indent=2)
        
        prompt = f"""Generate a Mermaid gantt chart from this roadmap data:

{items_json}

Use this format:
- Section by team
- Show dependencies with 'after' keyword
- Include priority in task names
- Output only the mermaid code block."""

        return self.client.generate(prompt, temperature=0.1)
    
    def generate_timeline_svg(self, items: List[RoadmapItem]) -> str:
        """Generate different visualization from same data."""
        # Can generate: SVG, ASCII, HTML timeline, etc.
        # Single source of truth, multiple outputs
        pass
```

### Key Engineering Insights

**1. Separation of Extraction and Rendering**

The critical insight is treating roadmap visualization as a two-phase pipeline: (1) structured data extraction and (2) format-specific rendering. LLMs excel at phase 1—converting natural language planning documents into consistent JSON schemas. Traditional tools handle phase 2. This separation enables:

- Version control on structured data (git-friendly JSON)
- Multiple visualization outputs without re-extraction
- Programmatic validation and consistency checks
- Easy integration with project management APIs

**2. Schema Enforcement via Few-Shot Examples**

LLMs don't naturally output consistent schemas. Engineers must provide explicit structure:

```python
SCHEMA_EXAMPLE = {
    "items": [
        {
            "id": "auth-system",
            "title": "User Authentication System",
            "timeline": "Q1 2024",
            "dependencies": [],
            "team": "platform",
            "priority": "high",
            "status": "in-progress",
            "effort_weeks": 6
        }
    ]
}
```

Including this in prompts increases schema compliance from ~60% to ~95%.

**3. Dependency Graph Validation**

LLMs may hallucinate dependencies. Always validate:

```python
def validate_dependencies(items: List[RoadmapItem]) -> List[str]:
    """Return list of validation errors."""
    errors = []
    item_ids = {item.title for item in items}
    
    for item in items:
        for dep in item.dependencies:
            if dep not in item_ids:
                errors.append(f"{item.title} depends on non-existent {dep}")
    
    # Check for circular dependencies
    graph = {item.title: item.dependencies for item in items}
    if has_cycle(graph):
        errors.append("Circular dependency detected")
    
    return errors

def has_cycle(graph: Dict[str, List[str]]) -> bool:
    """Detect cycles in dependency graph."""
    visited = set()
    rec_stack = set()
    
    def visit(node):
        if node in rec_stack:
            return True
        if node in visited:
            return False
        
        visited.add(node)
        rec_stack.add(node)
        
        for neighbor in graph.get(node, []):
            if visit(neighbor):
                return True
        
        rec_stack.remove(node)
        return False
    
    return any(visit(node) for node in graph)
```

### Why This Matters Now

Planning documents exist in Slack threads, meeting notes, email chains, and slide decks. Manually consolidating these into roadmaps creates:

- **2-4 hour weekly overhead** for engineering managers
- **Synchronization delays** (roadmaps lag reality by 1-2 weeks)
- **Format lock-in** (switching from Gantt to timeline requires full rebuild)

LLM-enhanced pipelines reduce extraction time by 80-90% and enable near-real-time roadmap updates from living documents.

## Technical Components

### 1. Structured Data Extraction

**Technical Explanation:**

The extraction component uses LLMs to parse unstructured text and output JSON conforming to a predefined schema. This requires careful prompt engineering to specify field types, constraints, and relationships.

**Implementation:**

```python
from typing import TypedDict, Literal
from enum import Enum

class Priority(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

class RoadmapItemSchema(TypedDict):
    id: str
    title: str
    timeline: str
    dependencies: List[str]
    team: str
    priority: Priority
    status: Literal["planned", "in-progress", "completed", "blocked"]
    confidence: Literal["high", "medium", "low"]

class StructuredExtractor:
    EXTRACTION_PROMPT = """Extract roadmap items from the following text.
Output a JSON object with an "items" array. Each item must have:
- id: lowercase-hyphenated unique identifier
- title: descriptive name
- timeline: quarter (Q1 2024, Q2 2024) or month (Jan 2024)
- dependencies: array of other item IDs this depends on
- team: engineering team name
- priority: "high", "medium", or "low"
- status: "planned", "in-progress", "completed", or "blocked"
- confidence: "high", "medium", "low" (how certain is this estimate?)

Example output:
{
  "items": [
    {
      "id": "payment-gateway",
      "title": "Integrate payment gateway",
      "timeline": "Q2 2024",
      "dependencies": ["user-auth"],
      "team": "payments",
      "priority": "high",
      "status": "planned",
      "confidence": "medium"
    }
  ]
}

Text to extract from:
{text}

Output only valid JSON:"""
    
    def extract(self, text: str) -> Dict:
        prompt = self.EXTRACTION_PROMPT.format(text=text)
        response = self.llm_client.generate(prompt, temperature=0.1)
        
        # Clean response (LLMs sometimes add markdown)
        cleaned = response.strip()
        if cleaned.startswith("```json"):
            cleaned = cleaned.split("```json")[1].split("```")[0]
        elif cleaned.startswith("```"):
            cleaned = cleaned.split("```")[1].split("```")[0]
        
        data = json.loads(cleaned)
        self._validate_schema(data)
        return data
    
    def _validate_schema(self, data: Dict) -> None:
        """Validate extracted data matches schema."""
        if "items" not in data:
            raise ValueError("Missing 'items' key")
        
        for item in data["items"]:
            required = {"id", "title", "timeline", "dependencies", 
                       "team", "priority", "status", "confidence"}
            missing = required - set(item.keys())
            if missing:
                raise ValueError(f"Item missing fields: {missing}")
            
            if item["priority"] not in ["high", "medium", "low"]:
                raise ValueError(f"Invalid priority: {item['priority']}")
```

**Practical Implications:**

- **Temperature 0.1-0.2** for extraction tasks (need consistency)
- **Validation catches ~15% of LLM output** that violates schema
- **Re-extraction with corrections** handles validation failures

**Trade-offs:**

- Stricter schemas → more validation failures but cleaner data
- Looser schemas → higher success rate but requires downstream handling
- **Recommended:** Medium strictness with automatic retry on failure

### 2. Multi-Format Rendering

**Technical Explanation:**

Once structured data exists, generate different visualizations by prompting the LLM to output format-specific syntax (Mermaid, Graphviz, SVG paths, ASCII art).

**Implementation:**

```python
class MultiFormatRenderer:
    """Generate multiple visualization formats from structured data."""
    
    def to_mermaid_gantt(self, data: Dict) -> str:
        """Generate Mermaid Gantt chart."""
        items_json = json.dumps(data, indent=2)
        
        prompt = f"""Convert this roadmap data to a Mermaid gantt chart.

Rules:
- Group by team using 'section' keyword
- Format tasks as: TASK_ID: TASK_NAME :PRIORITY, TIMELINE
- Show dependencies with :after DEPENDENCY_ID
- Use priority tags: crit for high, active for in-progress

Data:
{items_json}

Output ONLY the mermaid code (no markdown fences):"""

        return self.llm_client.generate(prompt, temperature=0.0)
    
    def to_graphviz(self, data: Dict) -> str:
        """Generate Graphviz dependency graph."""
        prompt = f"""Convert this roadmap to Graphviz DOT format.

Create a dependency graph showing:
- Nodes colored by status (green=completed, yellow=in-progress, blue=planned)
- Edges showing dependencies
- Node labels include title and timeline
- Use rankdir=LR (left to right)

Data:
{json.dumps(data, indent=2)}

Output only valid DOT syntax:"""

        return self.llm_client.generate(prompt, temperature=0.0)
    
    def to_ascii_timeline(self, data: Dict) -> str:
        """Generate ASCII art timeline."""
        prompt = f"""Create an ASCII art timeline from this roadmap.

Format:
Q1 2024  |==AUTH==|  |==API==|
Q2 2024              |====MOBILE====|
Q3 2024                        |=ANALYTICS=|

Use | for boundaries, = for work, show item titles abbreviated.
Group concurrent items on same line.

Data:
{json.dumps(data, indent=2)}

Output ASCII art:"""

        return self.llm_client.generate(prompt, temperature=0.0)
```

**Real Constraints:**

- **Mermaid rendering** works best for <20 items (readability limit)
- **Graphviz** handles dependencies well but requires `dot` binary installed
- **ASCII art** limited to ~80 character width for terminal display

**Example Output Comparison:**

For the same structured data:
- **Mermaid:** Best for executive presentations (renders in GitHub, Notion)
- **Graphviz:** Best for dependency analysis (clear graph layout)
- **ASCII:** Best for CLI tools, CI/CD pipeline output

### 3. Incremental Update Pipeline

**Technical Explanation:**

Roadmaps evolve continuously. Instead of full re-extraction, implement incremental updates that merge new information with existing structured data.

**Implementation:**

```python
from datetime import datetime
from typing import Optional

class RoadmapUpdateManager:
    """Manage incremental roadmap updates."""
    
    def __init__(self, storage_path: str, llm_client):
        self.storage_path = storage_path
        self.llm_client = llm_client
        self.current_roadmap = self._load_roadmap()
    
    def _load_roadmap(self) -> Dict:
        """Load existing roadmap or create new."""
        try:
            with open(self.storage_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return {
                "items": [],
                "last_updated": None,
                "version": 1
            }
    
    def update_from_text(self, new_info: str, source: str) -> Dict:
        """Update roadmap with new information."""
        # Extract new items
        extractor = StructuredExtractor(self.llm_client)
        new_data = extractor.extract(new_info)
        
        # Merge with existing
        merged = self._merge_roadmaps(
            self.current_roadmap,
            new_data,
            source
        )
        
        # Save with version bump
        merged["version"] = self.current_roadmap["version"] + 1
        merged["last_updated"] = datetime.utcnow().isoformat()
        
        with open(self.storage_path, 'w') as f:
            json.dump(merged, f, indent=2)
        
        self.current_roadmap = merged
        return merged
    
    def _merge_roadmaps(self, existing: Dict, new: Dict, source: str) -> Dict:
        """Intelligently merge roadmap updates."""
        existing_items = {item["id"]: item for item in existing["items"]}
        
        for new_item in new["items"]:
            item_id = new_item["id"]
            
            if item_id in existing_items:
                # Update existing item
                merged_item = self._merge_items(
                    existing_items[item_id],
                    new_item,
                    source
                )
                existing_items[item_id] = merged_item
            else:
                # Add new item
                new_item["added_by"] = source
                new_item["added_at"] = datetime.utcnow().isoformat()
                existing_items[item_id] = new_item
        
        return {
            "items": list(existing_items.values()),
            "last_updated": existing["last_updated"],
            "version": existing["version"]
        }
    
    def _merge_items(self, existing: Dict, new: Dict, source: str) -> Dict:
        """Merge individual roadmap items with conflict resolution."""
        prompt = f"""Two versions of a roadmap item exist. Merge them intelligently.

Existing version:
{json.dumps(existing, indent=2)}

New version (from {source}):
{json.dumps(new, indent=2)}

Rules:
- If status changed to more complete (planned→in-progress→completed), use new
- If timeline changed, use new value but add note about change
- Merge dependencies (union of both)
- Keep most recent priority if different
- Add "updated_fields" array listing what changed

Output merged item as JSON:"""

        response = self.llm_client.generate(prompt, temperature=0.1)
        merged = json.loads(response.strip())
        merged["last_updated_by"] = source
        merged["last_updated_at"] = datetime.utcnow().isoformat()
        
        return merged
```

**Practical Benefits:**

- **Version history** enables rollback if bad data introduce