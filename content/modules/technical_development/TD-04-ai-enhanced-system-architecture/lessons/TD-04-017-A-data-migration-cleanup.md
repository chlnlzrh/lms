# Data Migration & Cleanup for AI Systems

## Core Concepts

Data migration and cleanup in AI systems involves transforming raw, inconsistent data into structured formats that language models can effectively process. Unlike traditional database migrations where schema changes are the primary concern, AI data migration focuses on semantic consistency, context preservation, and format standardization that directly impacts model performance.

### Traditional vs. Modern Approach

```python
# Traditional Database Migration
# Focus: Schema transformation, referential integrity
import sqlite3

def traditional_migration(old_db: str, new_db: str) -> None:
    """Migrate customer records between databases."""
    old_conn = sqlite3.connect(old_db)
    new_conn = sqlite3.connect(new_db)
    
    # Direct column mapping
    old_conn.execute("SELECT id, name, email FROM customers")
    for row in old_conn.fetchall():
        new_conn.execute(
            "INSERT INTO users (user_id, full_name, email_address) VALUES (?, ?, ?)",
            row
        )
    
    new_conn.commit()
```

```python
# AI-Focused Data Migration
# Focus: Context preservation, semantic consistency, token efficiency
from typing import List, Dict, Any
import json
import re

def ai_migration(raw_data: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """
    Migrate customer data for AI processing.
    Preserves semantic meaning while standardizing format.
    """
    migrated = []
    
    for record in raw_data:
        # Normalize and enrich for AI context
        cleaned = {
            "context": f"Customer: {normalize_name(record['name'])}",
            "contact": standardize_email(record['email']),
            "history": consolidate_interactions(record.get('notes', [])),
            "metadata": json.dumps({
                "account_age_days": calculate_age(record['created_at']),
                "interaction_count": len(record.get('notes', []))
            })
        }
        
        # Validate semantic completeness
        if has_sufficient_context(cleaned):
            migrated.append(cleaned)
    
    return migrated

def normalize_name(name: str) -> str:
    """Remove extra whitespace, standardize capitalization."""
    return ' '.join(name.strip().split()).title()

def standardize_email(email: str) -> str:
    """Lowercase and validate basic format."""
    email = email.lower().strip()
    if not re.match(r'^[^@]+@[^@]+\.[^@]+$', email):
        return "invalid_email"
    return email

def consolidate_interactions(notes: List[str]) -> str:
    """Combine historical notes into concise summary."""
    if not notes:
        return "No previous interactions"
    
    # Remove duplicates, limit length for token efficiency
    unique_notes = list(dict.fromkeys(notes))
    summary = "; ".join(unique_notes[:5])  # Limit to 5 most recent
    return summary[:500]  # Cap at 500 chars

def has_sufficient_context(record: Dict[str, str]) -> bool:
    """Ensure record has minimum viable context for AI."""
    return (
        len(record['context']) > 10 and
        record['contact'] != "invalid_email" and
        len(record['history']) > 0
    )
```

### Key Engineering Insights

**Token economics matter**: Every character in your migrated data consumes tokens. A customer record with 50 redundant whitespace characters across 10,000 records wastes 500,000 tokens—real cost at scale.

**Context collapse is silent failure**: Traditional migrations throw errors on schema violations. AI migrations silently degrade when context is lost. A customer name split across fields (`first_name: "John"`, `last_name: "Smith"`) becomes meaningless fragments without explicit concatenation and labeling.

**Garbage persists differently**: In traditional systems, bad data causes query failures. In AI systems, bad data trains bad patterns. A single dataset with inconsistent date formats (MM/DD/YYYY vs. DD/MM/YYYY) can confuse a model across all future date interpretations.

### Why This Matters Now

Vector databases and retrieval-augmented generation (RAG) systems have made data migration a recurring task rather than a one-time event. You're no longer migrating data once for a new database—you're continuously transforming data for embeddings, updating knowledge bases, and reprocessing content as models improve. The engineering discipline required for clean, consistent data directly determines your model's ability to retrieve relevant context and generate accurate responses.

## Technical Components

### 1. Format Standardization

**Technical Explanation**: LLMs perform best with consistent structural patterns. Variation in formatting—even semantically equivalent data—creates noise that degrades model performance. Standardization means establishing canonical representations for dates, names, addresses, and domain-specific entities.

**Practical Implications**: A model trained on inconsistent date formats may fail to correctly extract temporal information. If 40% of your data uses "Jan 15, 2024", 30% uses "2024-01-15", and 30% uses "01/15/2024", the model learns three different patterns for the same concept.

**Real Constraints**: Over-standardization can lose important semantic information. Converting all text to lowercase improves consistency but loses proper noun signals. Balance uniformity with information preservation.

```python
from datetime import datetime
from typing import Optional
import re

class FormatStandardizer:
    """Standardize common data formats for AI processing."""
    
    @staticmethod
    def standardize_date(date_str: str) -> Optional[str]:
        """
        Convert various date formats to ISO 8601.
        Returns None if parsing fails.
        """
        formats = [
            "%m/%d/%Y",      # 01/15/2024
            "%d/%m/%Y",      # 15/01/2024
            "%Y-%m-%d",      # 2024-01-15
            "%b %d, %Y",     # Jan 15, 2024
            "%B %d, %Y",     # January 15, 2024
            "%m-%d-%Y",      # 01-15-2024
        ]
        
        for fmt in formats:
            try:
                dt = datetime.strptime(date_str.strip(), fmt)
                return dt.strftime("%Y-%m-%d")  # ISO 8601
            except ValueError:
                continue
        
        return None
    
    @staticmethod
    def standardize_phone(phone: str) -> str:
        """
        Convert phone numbers to E.164 format (assuming US).
        """
        # Remove all non-digit characters
        digits = re.sub(r'\D', '', phone)
        
        # Handle 10-digit US numbers
        if len(digits) == 10:
            return f"+1{digits}"
        elif len(digits) == 11 and digits[0] == '1':
            return f"+{digits}"
        
        return phone  # Return original if can't standardize
    
    @staticmethod
    def standardize_currency(amount: str) -> Optional[float]:
        """
        Extract numeric value from currency strings.
        """
        # Remove currency symbols and commas
        cleaned = re.sub(r'[,$€£¥]', '', amount.strip())
        
        try:
            return float(cleaned)
        except ValueError:
            return None

# Example usage
standardizer = FormatStandardizer()

sample_dates = ["01/15/2024", "Jan 15, 2024", "2024-01-15"]
for date in sample_dates:
    print(f"{date} -> {standardizer.standardize_date(date)}")
# Output:
# 01/15/2024 -> 2024-01-15
# Jan 15, 2024 -> 2024-01-15
# 2024-01-15 -> 2024-01-15

sample_phones = ["(555) 123-4567", "555.123.4567", "5551234567"]
for phone in sample_phones:
    print(f"{phone} -> {standardizer.standardize_phone(phone)}")
# Output:
# (555) 123-4567 -> +15551234567
# 555.123.4567 -> +15551234567
# 5551234567 -> +15551234567
```

### 2. Deduplication with Semantic Awareness

**Technical Explanation**: Traditional deduplication uses exact matching or simple fuzzy logic. AI systems require semantic deduplication—identifying records that mean the same thing even with different wording. This prevents embedding spaces from being polluted with redundant vectors.

**Practical Implications**: Duplicate data in a RAG system means wasted storage and retrieval slots. If your system retrieves top-5 results and 3 are duplicates, you've reduced effective context by 60%.

**Real Constraints**: Semantic similarity detection requires embeddings, which is computationally expensive. For large datasets, implement tiered deduplication: exact matching first, then fuzzy matching, then semantic similarity only for remaining candidates.

```python
from typing import List, Set, Tuple
from difflib import SequenceMatcher
import hashlib

class SemanticDeduplicator:
    """Multi-tier deduplication strategy."""
    
    def __init__(self, fuzzy_threshold: float = 0.85):
        self.fuzzy_threshold = fuzzy_threshold
        self.seen_hashes: Set[str] = set()
    
    def deduplicate(self, records: List[str]) -> List[str]:
        """
        Apply three-tier deduplication:
        1. Exact match (hash-based)
        2. Fuzzy match (character similarity)
        3. Semantic match (would use embeddings in production)
        """
        unique_records = []
        
        for record in records:
            # Tier 1: Exact deduplication via hash
            record_hash = self._hash_record(record)
            if record_hash in self.seen_hashes:
                continue
            
            # Tier 2: Fuzzy deduplication
            if self._is_fuzzy_duplicate(record, unique_records):
                continue
            
            # Tier 3: Semantic deduplication (simplified here)
            # In production, use embeddings and cosine similarity
            if self._is_semantic_duplicate(record, unique_records):
                continue
            
            self.seen_hashes.add(record_hash)
            unique_records.append(record)
        
        return unique_records
    
    def _hash_record(self, record: str) -> str:
        """Create hash of normalized record."""
        normalized = record.lower().strip()
        normalized = ' '.join(normalized.split())  # Normalize whitespace
        return hashlib.md5(normalized.encode()).hexdigest()
    
    def _is_fuzzy_duplicate(self, record: str, existing: List[str]) -> bool:
        """Check if record is very similar to existing records."""
        for existing_record in existing:
            similarity = SequenceMatcher(None, record, existing_record).ratio()
            if similarity >= self.fuzzy_threshold:
                return True
        return False
    
    def _is_semantic_duplicate(self, record: str, existing: List[str]) -> bool:
        """
        Simplified semantic check.
        In production, use actual embeddings and cosine similarity.
        """
        # Simplified: check if key terms overlap significantly
        record_terms = set(record.lower().split())
        
        for existing_record in existing:
            existing_terms = set(existing_record.lower().split())
            
            if len(record_terms) == 0 or len(existing_terms) == 0:
                continue
            
            overlap = len(record_terms & existing_terms)
            union = len(record_terms | existing_terms)
            
            # Jaccard similarity
            if overlap / union >= 0.7:
                return True
        
        return False

# Example usage
deduplicator = SemanticDeduplicator(fuzzy_threshold=0.85)

customer_notes = [
    "Customer called about billing issue with invoice #1234",
    "Customer called about billing issue with invoice #1234",  # Exact duplicate
    "Customer phoned regarding billing problem invoice #1234",  # Fuzzy duplicate
    "Customer reported shipping delay",  # Unique
    "Customer reported shipping delays",  # Fuzzy duplicate
]

unique_notes = deduplicator.deduplicate(customer_notes)
print(f"Original: {len(customer_notes)} records")
print(f"After dedup: {len(unique_notes)} records")
print("\nUnique records:")
for note in unique_notes:
    print(f"  - {note}")

# Output:
# Original: 5 records
# After dedup: 2 records
# 
# Unique records:
#   - Customer called about billing issue with invoice #1234
#   - Customer reported shipping delay
```

### 3. Null Handling and Imputation

**Technical Explanation**: LLMs interpret missing data differently than databases. A NULL value in SQL is explicitly absent. In text data fed to an LLM, emptiness can be represented as empty strings, placeholder text, or omitted fields—each with different semantic implications.

**Practical Implications**: An empty string in a customer feedback field might mean "no feedback provided" or "feedback pending" or represent a data collection failure. The model can't distinguish without explicit signals.

**Real Constraints**: Imputation (filling missing values) must preserve truthfulness. Inventing plausible data to fill gaps creates hallucination patterns. Better to explicitly mark missing data than fabricate.

```python
from typing import Dict, Any, Optional
from enum import Enum

class MissingDataStrategy(Enum):
    """Strategies for handling missing data."""
    OMIT = "omit"              # Remove field entirely
    EXPLICIT = "explicit"       # Mark as explicitly missing
    DEFAULT = "default"         # Use safe default value
    CONTEXT = "context"         # Derive from related fields

class NullHandler:
    """Handle missing data with context-aware strategies."""
    
    @staticmethod
    def handle_missing(
        record: Dict[str, Any],
        field: str,
        strategy: MissingDataStrategy = MissingDataStrategy.EXPLICIT,
        default_value: Optional[Any] = None
    ) -> Dict[str, Any]:
        """
        Apply missing data strategy to a specific field.
        """
        if field in record and record[field] not in [None, "", "null", "NULL"]:
            return record  # Field has valid value
        
        if strategy == MissingDataStrategy.OMIT:
            record.pop(field, None)
        
        elif strategy == MissingDataStrategy.EXPLICIT:
            record[field] = f"[No {field.replace('_', ' ')} provided]"
        
        elif strategy == MissingDataStrategy.DEFAULT:
            record[field] = default_value if default_value is not None else "Unknown"
        
        elif strategy == MissingDataStrategy.CONTEXT:
            # Derive from other fields if possible
            record[field] = NullHandler._derive_from_context(record, field)
        
        return record
    
    @staticmethod
    def _derive_from_context(record: Dict[str, Any], field: str) -> str:
        """
        Attempt to derive missing value from related fields.
        """
        # Example: derive full_name from first_name and last_name
        if field == "full_name":
            first = record.get("first_name", "")
            last = record.get("last_name", "")
            if first or last:
                return f"{first} {last}".strip()
        
        # Example: derive status from last_activity_date
        if field == "status":
            last_activity = record.get("last_activity_date")
            if last_activity:
                # In production, parse date and apply business logic
                return "Active"
            return "Inactive"
        
        return "[Data not available]"
    
    @staticmethod
    def clean_record(record: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply appropriate strategies to all fields in a record.
        """
        field_strategies = {
            "email": MissingDataStrategy.EXPLICIT,
            "phone": MissingDataStrategy.OMIT,
            "address": MissingDataStrategy.EXPLICIT,
            "notes": MissingDataStrategy.DEFAULT,
            "full_name": MissingDataStrategy.CONTEXT,
            "status": MissingDataStrategy.CONTEXT,
        }
        
        cleaned = record.copy()
        
        for field, strategy in field_