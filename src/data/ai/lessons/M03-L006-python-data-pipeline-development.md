# Python Data Pipeline Development for AI/LLM Systems

## Core Concepts

A data pipeline is a series of processing steps that transform raw data into a format suitable for machine learning models. In AI/LLM contexts, pipelines handle everything from initial data extraction through cleaning, transformation, chunking, embedding generation, and loading into vector stores or training datasets.

### Traditional vs. Modern Pipeline Architecture

**Traditional ETL (pre-LLM era):**

```python
import pandas as pd
from typing import List

def traditional_etl_pipeline(csv_path: str) -> pd.DataFrame:
    """Simple ETL: Extract CSV, Transform columns, Load to database"""
    # Extract
    df = pd.read_csv(csv_path)
    
    # Transform
    df['date'] = pd.to_datetime(df['date'])
    df['amount'] = df['amount'].astype(float)
    df = df.dropna()
    
    # Load (simulated)
    # db.insert(df)
    
    return df
```

**Modern AI Pipeline (LLM-aware):**

```python
import pandas as pd
from typing import List, Dict, Iterator
from dataclasses import dataclass
import hashlib
import json

@dataclass
class Document:
    content: str
    metadata: Dict
    embedding: List[float] = None
    chunk_id: str = None

def modern_ai_pipeline(csv_path: str) -> Iterator[Document]:
    """
    AI-aware pipeline: streaming, chunking, deduplication, metadata preservation
    """
    # Extract with streaming (memory efficient)
    for chunk_df in pd.read_csv(csv_path, chunksize=1000):
        
        # Transform: combine relevant fields into document format
        for _, row in chunk_df.iterrows():
            content = f"Date: {row['date']}\nAmount: {row['amount']}\nDescription: {row['description']}"
            
            # Generate deterministic ID for deduplication
            chunk_id = hashlib.sha256(content.encode()).hexdigest()
            
            # Preserve metadata for filtering/routing
            metadata = {
                'source': csv_path,
                'date': str(row['date']),
                'amount': float(row['amount']),
                'record_type': 'transaction'
            }
            
            yield Document(
                content=content,
                metadata=metadata,
                chunk_id=chunk_id
            )
```

**Key Differences:**
1. **Streaming vs. Batch**: Modern pipelines process data incrementally to handle large datasets (10GB+) without loading everything into memory
2. **Document-Oriented**: Data transformed into text documents with rich metadata, not just normalized tables
3. **Deduplication**: Content hashing prevents duplicate processing (critical when costs are per-token)
4. **Metadata Preservation**: Keeps filterable attributes separate from embedding content

### Why This Matters NOW

LLM systems have fundamentally different data requirements than traditional ML:

1. **Token Economics**: Every character costs money in embedding APIs and storage. A poorly designed pipeline that doesn't deduplicate can cost 3-10x more.

2. **Context Window Constraints**: Documents must be chunked intelligently. A 50-page PDF processed naively creates 200+ chunks; semantic chunking reduces this to 40 chunks with better retrieval accuracy.

3. **Retrieval Quality**: How you split and structure data during pipeline processing directly determines RAG system accuracy. Poor chunking = 40-60% drop in answer quality.

4. **Scale Mismatch**: Traditional pipelines assumed structured data. LLM pipelines process unstructured text from PDFs, HTML, Slack messages—requiring different validation and quality checks.

### Engineering Insight That Changes Perspective

**The Pipeline IS the Feature Engineering**: In traditional ML, pipelines delivered clean data, then separate feature engineering created model inputs. In LLM systems, pipeline decisions (chunking strategy, metadata extraction, document structure) ARE your feature engineering. There's no separate step—your pipeline output quality directly determines model performance.

---

## Technical Components

### 1. Data Extraction with Format Handling

**Technical Explanation:**

Extraction must handle diverse source formats (PDF, DOCX, HTML, JSON, databases) while preserving semantic structure. Unlike traditional ETL that standardizes to tables, AI pipelines preserve document semantics—headings, lists, tables—because LLMs leverage this structure.

**Practical Implementation:**

```python
from pathlib import Path
from typing import Iterator, Union
import json
import pypdf
from dataclasses import dataclass

@dataclass
class RawDocument:
    content: str
    source_path: str
    format: str
    metadata: dict

class UniversalExtractor:
    """Extract text while preserving structure from multiple formats"""
    
    def extract(self, path: Union[str, Path]) -> RawDocument:
        path = Path(path)
        
        extractors = {
            '.pdf': self._extract_pdf,
            '.txt': self._extract_text,
            '.json': self._extract_json,
            '.md': self._extract_text,
        }
        
        extractor = extractors.get(path.suffix.lower())
        if not extractor:
            raise ValueError(f"Unsupported format: {path.suffix}")
        
        return extractor(path)
    
    def _extract_pdf(self, path: Path) -> RawDocument:
        """Extract PDF preserving page boundaries"""
        reader = pypdf.PdfReader(str(path))
        
        pages = []
        for i, page in enumerate(reader.pages):
            text = page.extract_text()
            pages.append(f"[Page {i+1}]\n{text}")
        
        content = "\n\n---\n\n".join(pages)
        
        return RawDocument(
            content=content,
            source_path=str(path),
            format='pdf',
            metadata={
                'page_count': len(reader.pages),
                'title': reader.metadata.get('/Title', ''),
            }
        )
    
    def _extract_text(self, path: Path) -> RawDocument:
        with open(path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        return RawDocument(
            content=content,
            source_path=str(path),
            format=path.suffix[1:],
            metadata={'size_bytes': path.stat().st_size}
        )
    
    def _extract_json(self, path: Path) -> RawDocument:
        """Convert JSON to readable text format"""
        with open(path, 'r') as f:
            data = json.load(f)
        
        # Convert to readable format (not just json.dumps)
        content = self._json_to_text(data)
        
        return RawDocument(
            content=content,
            source_path=str(path),
            format='json',
            metadata={'record_count': len(data) if isinstance(data, list) else 1}
        )
    
    def _json_to_text(self, obj, indent=0) -> str:
        """Convert JSON to human-readable text for LLM processing"""
        lines = []
        prefix = "  " * indent
        
        if isinstance(obj, dict):
            for key, value in obj.items():
                if isinstance(value, (dict, list)):
                    lines.append(f"{prefix}{key}:")
                    lines.append(self._json_to_text(value, indent + 1))
                else:
                    lines.append(f"{prefix}{key}: {value}")
        elif isinstance(obj, list):
            for item in obj:
                lines.append(self._json_to_text(item, indent))
                lines.append("")
        else:
            lines.append(f"{prefix}{obj}")
        
        return "\n".join(lines)
```

**Real Constraints:**
- PDFs can have embedded images/tables that extract as garbage text—requires OCR fallback or table detection
- Character encoding issues cause 5-10% of extractions to fail; need error handling and logging
- Large files (100MB+) require streaming extraction to avoid memory issues

**Trade-offs:**
- **Preserving structure** (markdown formatting, page breaks) increases token count by 10-15% but improves retrieval precision by 20-30%
- **Universal extractors** are slower than format-specific libraries but reduce maintenance burden

---

### 2. Semantic Chunking Strategy

**Technical Explanation:**

Chunking splits long documents into smaller pieces that fit within embedding model limits (typically 512-8192 tokens). Naive chunking (every N characters) breaks sentences and ideas mid-thought. Semantic chunking respects document structure—splitting on section boundaries, paragraphs, or natural topic shifts.

**Practical Implementation:**

```python
from typing import List
import re

class SemanticChunker:
    """Chunk text respecting semantic boundaries"""
    
    def __init__(self, max_chunk_size: int = 1000, overlap: int = 200):
        self.max_chunk_size = max_chunk_size
        self.overlap = overlap
    
    def chunk(self, text: str, metadata: dict = None) -> List[Document]:
        """Split text on semantic boundaries with overlap"""
        # First, split on major boundaries (headers, page breaks)
        sections = self._split_on_boundaries(text)
        
        chunks = []
        for section in sections:
            if len(section) <= self.max_chunk_size:
                chunks.append(section)
            else:
                # Split large sections on paragraph boundaries
                chunks.extend(self._split_large_section(section))
        
        # Add overlap between chunks for context continuity
        overlapped_chunks = self._add_overlap(chunks)
        
        # Convert to Document objects
        documents = []
        for i, chunk in enumerate(overlapped_chunks):
            doc_metadata = {
                **(metadata or {}),
                'chunk_index': i,
                'total_chunks': len(overlapped_chunks)
            }
            
            documents.append(Document(
                content=chunk,
                metadata=doc_metadata,
                chunk_id=hashlib.sha256(chunk.encode()).hexdigest()
            ))
        
        return documents
    
    def _split_on_boundaries(self, text: str) -> List[str]:
        """Split on headers and major section breaks"""
        # Patterns for semantic boundaries (in priority order)
        patterns = [
            r'\n#{1,3}\s+.+\n',  # Markdown headers
            r'\n\[Page \d+\]\n',  # Page breaks
            r'\n\n\n+',           # Multiple blank lines
        ]
        
        sections = [text]
        for pattern in patterns:
            new_sections = []
            for section in sections:
                splits = re.split(f'({pattern})', section)
                # Recombine delimiter with following text
                for i in range(0, len(splits)-1, 2):
                    if i+1 < len(splits):
                        new_sections.append(splits[i] + splits[i+1])
                    else:
                        new_sections.append(splits[i])
            sections = new_sections
        
        return [s.strip() for s in sections if s.strip()]
    
    def _split_large_section(self, section: str) -> List[str]:
        """Split section that exceeds max size on paragraph boundaries"""
        paragraphs = section.split('\n\n')
        
        chunks = []
        current_chunk = []
        current_size = 0
        
        for para in paragraphs:
            para_size = len(para)
            
            if current_size + para_size > self.max_chunk_size and current_chunk:
                # Finalize current chunk
                chunks.append('\n\n'.join(current_chunk))
                current_chunk = [para]
                current_size = para_size
            else:
                current_chunk.append(para)
                current_size += para_size
        
        if current_chunk:
            chunks.append('\n\n'.join(current_chunk))
        
        return chunks
    
    def _add_overlap(self, chunks: List[str]) -> List[str]:
        """Add overlapping text between chunks for context continuity"""
        if len(chunks) <= 1:
            return chunks
        
        overlapped = [chunks[0]]
        
        for i in range(1, len(chunks)):
            prev_chunk = chunks[i-1]
            current_chunk = chunks[i]
            
            # Take last N characters from previous chunk as overlap
            overlap_text = prev_chunk[-self.overlap:] if len(prev_chunk) > self.overlap else prev_chunk
            
            # Add overlap marker for debugging
            overlapped_chunk = f"[...continued from previous chunk...]\n{overlap_text}\n\n{current_chunk}"
            overlapped.append(overlapped_chunk)
        
        return overlapped
```

**Real Constraints:**
- Overlap increases total token count by 15-25% but improves retrieval recall for queries spanning chunk boundaries
- Fixed-size chunking is 10x faster than semantic chunking but produces 30-40% worse retrieval results
- Token counting (not character counting) is necessary for accurate chunk sizing; character counts are 20-30% off for code/special characters

**Trade-offs:**
- **Larger chunks** (1500+ tokens): Better context preservation, fewer total chunks, but worse specificity in retrieval
- **Smaller chunks** (300-500 tokens): More precise retrieval, but may lose important context and increase infrastructure costs

---

### 3. Deduplication and Content Hashing

**Technical Explanation:**

Duplicate or near-duplicate content wastes embedding API calls and storage. Content hashing creates deterministic fingerprints to detect exact duplicates. Near-duplicate detection uses MinHash/SimHash algorithms to find content that's 80-95% similar without comparing every pair.

**Practical Implementation:**

```python
from typing import Set, Dict
import hashlib
from collections import defaultdict

class DeduplicationEngine:
    """Detect and remove duplicate/near-duplicate documents"""
    
    def __init__(self):
        self.seen_hashes: Set[str] = set()
        self.near_duplicate_buckets: Dict[int, List[str]] = defaultdict(list)
    
    def is_duplicate(self, doc: Document) -> bool:
        """Check if document is exact duplicate"""
        content_hash = self._hash_content(doc.content)
        
        if content_hash in self.seen_hashes:
            return True
        
        self.seen_hashes.add(content_hash)
        return False
    
    def is_near_duplicate(self, doc: Document, similarity_threshold: float = 0.85) -> bool:
        """Check if document is near-duplicate using MinHash approximation"""
        # Generate shingles (n-grams) from content
        shingles = self._generate_shingles(doc.content, n=3)
        
        # Create MinHash signature
        signature = self._minhash(shingles)
        
        # Check against existing signatures in buckets
        bucket_key = signature % 100  # Simple bucketing
        
        for existing_sig in self.near_duplicate_buckets[bucket_key]:
            similarity = self._estimate_similarity(signature, existing_sig)
            if similarity >= similarity_threshold:
                return True
        
        self.near_duplicate_buckets[bucket_key].append(signature)
        return False
    
    def _hash_content(self, content: str) -> str:
        """Create deterministic hash of normalized content"""
        # Normalize: lowercase, strip whitespace, remove special chars
        normalized = ' '.join(content.lower().split())
        normalized = ''.join(c for c in normalized if c.isalnum() or c.isspace())
        
        return hashlib.sha256(normalized.encode()).hexdigest()
    
    def _generate_shingles(self, text: str, n: int = 3) -> Set[str]:
        """Generate n-gram shingles from text"""
        words = text.lower().split()
        shingles = set()
        
        for i in range(len(words) - n + 1):
            shingle = ' '.join(words[i:i+n])
            shingles.add(shingle)
        
        return shingles
    
    def _minhash(self, shingles: Set[str], num_hashes: int = 100) -> int:
        """Create MinHash signature (simplified version)"""
        # In production, use datasketch library for proper MinHash
        # This is a simplified educational version
        min_hash =