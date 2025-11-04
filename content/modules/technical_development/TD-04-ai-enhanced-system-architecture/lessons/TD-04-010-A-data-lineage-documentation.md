# Data Lineage Documentation for AI Systems

## Core Concepts

### Technical Definition

Data lineage documentation tracks the complete lifecycle of data through AI systems: from raw sources through transformations, preprocessing, training, inference, and storage. It creates an immutable, queryable record of what data was used, how it was modified, when operations occurred, and which versions of code and models touched it.

Unlike traditional data lineage in analytics pipelines (tracking SQL transforms and ETL jobs), AI data lineage must capture:

- **Stochastic operations**: Random sampling, augmentation with seed states
- **Model versioning dependencies**: Which model weights processed which data batches
- **Feature drift**: How feature distributions change across pipeline versions
- **Feedback loops**: When model outputs become training inputs
- **Multi-modal transformations**: Text→embeddings, images→vectors, audio→spectrograms

### Engineering Analogy: Traditional vs. AI Data Lineage

```python
# Traditional ETL Lineage (Deterministic)
class TraditionalLineage:
    """Tracks deterministic transformations"""
    def __init__(self):
        self.operations = []
    
    def log_transform(self, input_table: str, output_table: str, 
                      sql_query: str) -> None:
        """Simple append-only log of table transformations"""
        self.operations.append({
            'input': input_table,
            'output': output_table,
            'operation': sql_query,
            'timestamp': datetime.now()
        })
    
    def trace_origin(self, table: str) -> List[str]:
        """Walk backward to find source tables"""
        return [op['input'] for op in self.operations 
                if op['output'] == table]

# AI Pipeline Lineage (Stochastic + Versioned)
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from hashlib import sha256
import json

@dataclass
class DataArtifact:
    """Immutable reference to data at a point in time"""
    artifact_id: str
    content_hash: str
    schema_version: str
    row_count: int
    feature_stats: Dict[str, Any]
    created_at: datetime
    
@dataclass
class LineageNode:
    """Captures non-deterministic transformation context"""
    input_artifacts: List[str]  # artifact_ids
    output_artifact: str
    operation_type: str  # 'augment', 'train', 'inference'
    code_version: str  # git commit hash
    model_version: Optional[str]  # model weights hash
    hyperparameters: Dict[str, Any]
    random_seed: Optional[int]
    execution_metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Generate deterministic node ID from inputs"""
        content = json.dumps({
            'inputs': sorted(self.input_artifacts),
            'operation': self.operation_type,
            'code': self.code_version,
            'model': self.model_version,
            'hyperparams': self.hyperparameters,
            'seed': self.random_seed
        }, sort_keys=True)
        self.node_id = sha256(content.encode()).hexdigest()[:16]

class AILineageTracker:
    """Production-grade lineage with reproducibility guarantees"""
    def __init__(self, storage_backend: str):
        self.artifacts: Dict[str, DataArtifact] = {}
        self.lineage_graph: Dict[str, LineageNode] = {}
        self.storage = storage_backend
    
    def register_artifact(self, data: Any, metadata: Dict) -> str:
        """Create immutable artifact with content-based ID"""
        content_hash = self._hash_data(data)
        artifact = DataArtifact(
            artifact_id=f"artifact_{content_hash[:12]}",
            content_hash=content_hash,
            schema_version=metadata['schema_version'],
            row_count=len(data),
            feature_stats=self._compute_stats(data),
            created_at=datetime.now()
        )
        self.artifacts[artifact.artifact_id] = artifact
        return artifact.artifact_id
    
    def log_transformation(self, node: LineageNode) -> str:
        """Record transformation with full reproducibility context"""
        self.lineage_graph[node.node_id] = node
        self._persist_to_storage(node)
        return node.node_id
    
    def trace_to_source(self, artifact_id: str) -> List[LineageNode]:
        """Reconstruct full provenance chain"""
        chain = []
        current = artifact_id
        
        while True:
            # Find nodes that produced current artifact
            producers = [n for n in self.lineage_graph.values() 
                        if n.output_artifact == current]
            if not producers:
                break
            
            node = producers[0]  # Should be unique
            chain.append(node)
            
            # Continue tracing from inputs
            if node.input_artifacts:
                current = node.input_artifacts[0]
            else:
                break
        
        return chain
    
    def verify_reproducibility(self, artifact_id: str) -> bool:
        """Check if artifact can be reproduced from lineage"""
        chain = self.trace_to_source(artifact_id)
        
        for node in chain:
            # Verify all dependencies exist
            for input_id in node.input_artifacts:
                if input_id not in self.artifacts:
                    return False
            
            # Verify code version is accessible
            if not self._code_exists(node.code_version):
                return False
            
            # Verify model weights if applicable
            if node.model_version and not self._model_exists(node.model_version):
                return False
        
        return True
```

**Key difference**: Traditional lineage tracks *what happened*. AI lineage must track *how to reproduce what happened*, including all sources of non-determinism.

### Why This Matters NOW

1. **Regulatory Compliance**: EU AI Act, GDPR, and sector-specific regulations require provenance documentation for high-risk AI systems. You must prove which data trained which models.

2. **Model Debugging**: When a model fails in production, you need to trace back through: inference inputs → model version → training data → preprocessing → raw sources. Without lineage, debugging takes weeks instead of hours.

3. **Data Contamination**: Training/test leakage is endemic in ML. Lineage lets you verify data partitions never mixed across pipeline versions.

4. **Cost Attribution**: Cloud ML training costs $10K-$1M per run. Lineage enables cost tracking per data source, team, and experiment lineage branch.

5. **Reproducibility Crisis**: 70% of ML papers have non-reproducible results. Production systems need higher standards. Lineage is the engineering discipline that makes reproduction possible.

## Technical Components

### 1. Artifact Versioning and Content Addressing

**Technical Explanation**

Content-addressed artifacts use cryptographic hashes of data content as identifiers, making artifacts immutable and deduplication automatic. Unlike location-based versioning (S3 paths with version numbers), content addressing guarantees bit-for-bit identity.

```python
import hashlib
import pickle
from typing import Protocol, Any
import pandas as pd
import numpy as np

class HashableArtifact(Protocol):
    """Interface for artifacts that can be content-addressed"""
    def to_bytes(self) -> bytes:
        ...

class DatasetArtifact:
    """Content-addressed dataset with schema tracking"""
    
    def __init__(self, df: pd.DataFrame, schema_version: str):
        self.df = df
        self.schema_version = schema_version
        self._content_hash: Optional[str] = None
        self._canonical_form: Optional[bytes] = None
    
    def to_bytes(self) -> bytes:
        """Convert to canonical byte representation"""
        if self._canonical_form is None:
            # Normalize: sort columns, reset index, deterministic pickle
            normalized = self.df.copy()
            normalized = normalized.sort_index(axis=1)  # Sort columns
            normalized = normalized.reset_index(drop=True)  # Remove index
            
            # Deterministic serialization
            self._canonical_form = pickle.dumps(
                normalized,
                protocol=pickle.HIGHEST_PROTOCOL,
                fix_imports=False
            )
        
        return self._canonical_form
    
    @property
    def content_hash(self) -> str:
        """Compute deterministic content hash"""
        if self._content_hash is None:
            hasher = hashlib.sha256()
            hasher.update(self.to_bytes())
            hasher.update(self.schema_version.encode())
            self._content_hash = hasher.hexdigest()
        
        return self._content_hash
    
    def __eq__(self, other: 'DatasetArtifact') -> bool:
        """Content-based equality"""
        return (isinstance(other, DatasetArtifact) and 
                self.content_hash == other.content_hash)

class ModelArtifact:
    """Content-addressed model weights with architecture hash"""
    
    def __init__(self, state_dict: Dict, architecture_hash: str):
        self.state_dict = state_dict
        self.architecture_hash = architecture_hash
    
    def to_bytes(self) -> bytes:
        """Convert model weights to canonical bytes"""
        # Sort parameters by name for determinism
        sorted_params = sorted(self.state_dict.items())
        
        # Serialize each tensor deterministically
        bytes_list = []
        for name, tensor in sorted_params:
            bytes_list.append(name.encode())
            
            # Convert to numpy for deterministic serialization
            if hasattr(tensor, 'cpu'):  # PyTorch tensor
                arr = tensor.cpu().numpy()
            else:
                arr = np.array(tensor)
            
            bytes_list.append(arr.tobytes())
        
        return b''.join(bytes_list)
    
    @property
    def content_hash(self) -> str:
        """Hash of model weights + architecture"""
        hasher = hashlib.sha256()
        hasher.update(self.architecture_hash.encode())
        hasher.update(self.to_bytes())
        return hasher.hexdigest()

# Usage example
df1 = pd.DataFrame({'feature_a': [1, 2, 3], 'feature_b': [4, 5, 6]})
artifact1 = DatasetArtifact(df1, schema_version='v1')

# Same data in different column order - same hash
df2 = pd.DataFrame({'feature_b': [4, 5, 6], 'feature_a': [1, 2, 3]})
artifact2 = DatasetArtifact(df2, schema_version='v1')

print(f"Hash 1: {artifact1.content_hash[:12]}")
print(f"Hash 2: {artifact2.content_hash[:12]}")
print(f"Equal: {artifact1 == artifact2}")  # True
```

**Practical Implications**

- **Automatic Deduplication**: Store each unique dataset once. If five experiments use the same preprocessing output, it's stored once and referenced five times.
- **Immutability Guarantee**: Content hash changes if data changes. Impossible to silently modify artifacts.
- **Cache Invalidation**: Transformations can be memoized. If input hash and operation are unchanged, return cached output.

**Real Constraints**

- **Computation Cost**: Hashing 100GB datasets takes 5-10 minutes. Mitigate with incremental hashing or chunk-level hashing.
- **Floating Point Precision**: `0.1 + 0.2 != 0.3` in floating point. Use fixed precision (e.g., round to 6 decimals) or relative epsilon comparisons.
- **Non-Deterministic Libraries**: Some libraries (e.g., older pandas versions) have non-deterministic serialization. Test thoroughly.

### 2. Transformation Provenance with Seed Tracking

**Technical Explanation**

Stochastic operations (data augmentation, sampling, dropout) must record random seeds to be reproducible. Seed tracking extends beyond `random.seed(42)` to capture numpy, torch, tensorflow, and system-level entropy sources.

```python
import random
import numpy as np
from typing import Dict, Callable, Any
from contextlib import contextmanager
from dataclasses import dataclass
import sys

@dataclass
class SeedState:
    """Captures all sources of randomness"""
    python_seed: int
    numpy_seed: int
    torch_seed: Optional[int]
    cuda_seed: Optional[int]
    system_seed: int
    
    def apply(self) -> None:
        """Set all random number generators"""
        random.seed(self.python_seed)
        np.random.seed(self.numpy_seed)
        
        try:
            import torch
            torch.manual_seed(self.torch_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self.cuda_seed)
        except ImportError:
            pass
    
    @staticmethod
    def from_master_seed(master_seed: int) -> 'SeedState':
        """Derive all RNG seeds from master seed"""
        # Use hash to derive independent seeds
        def derive_seed(name: str) -> int:
            return hash((master_seed, name)) & 0xFFFFFFFF
        
        return SeedState(
            python_seed=derive_seed('python'),
            numpy_seed=derive_seed('numpy'),
            torch_seed=derive_seed('torch'),
            cuda_seed=derive_seed('cuda'),
            system_seed=master_seed
        )

class ReproducibleTransform:
    """Wrapper that makes any transform reproducible"""
    
    def __init__(self, transform_fn: Callable, seed: int):
        self.transform_fn = transform_fn
        self.seed_state = SeedState.from_master_seed(seed)
        self.execution_count = 0
    
    def __call__(self, *args, **kwargs) -> Any:
        """Execute transform with seed isolation"""
        # Save current RNG state
        python_state = random.getstate()
        numpy_state = np.random.get_state()
        
        try:
            # Apply reproducible seed
            self.seed_state.apply()
            
            # Execute transform
            result = self.transform_fn(*args, **kwargs)
            
            self.execution_count += 1
            return result
            
        finally:
            # Restore original RNG state
            random.setstate(python_state)
            np.random.set_state(numpy_state)
    
    def get_lineage_metadata(self) -> Dict[str, Any]:
        """Return reproducibility metadata"""
        return {
            'transform': self.transform_fn.__name__,
            'seed_state': {
                'python': self.seed_state.python_seed,
                'numpy': self.seed_state.numpy_seed,
                'torch': self.seed_state.torch_seed,
            },
            'execution_count': self.execution_count,
            'reproducibility_level': 'exact'
        }

# Practical example: Reproducible data augmentation
def augment_images(images: np.ndarray, flip_prob: float = 0.5) -> np.ndarray:
    """Apply random augmentation to images"""
    augmented = images.copy()
    
    for i in range(len(augmented)):
        # Random horizontal flip
        if np.random.random() < flip_prob:
            augmented[i] = np.fliplr(augmented[i])
        
        # Random rotation
        angle = np.random.uniform(-15, 15)
        # ... rotation logic ...
    
    return augmented

# Make it reproducible
images = np.random.rand(10, 64, 64, 3)  # 10 sample images

augment_reproducible = ReproducibleTransform(augment_images, seed=42)

# These produce identical results
output1 = augment_reproducible(images)
output2 = augment_reproducible(images)

print(f"Outputs identical: {np.allclose(output1, output2)}")  # True
print(f"Lineage: {augment_reproducible.get_lineage_metadata()}")
```

**Practical Implications**

- **Exact Reproduction