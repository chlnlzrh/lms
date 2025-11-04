# Incident Response Procedures for AI/LLM Systems

## Core Concepts

Incident response in AI/LLM systems requires fundamentally different procedures than traditional software systems. While conventional applications fail predictably—null pointer exceptions, connection timeouts, out-of-memory errors—LLM systems exhibit non-deterministic failure modes: hallucinations that pass validation checks, prompt injection attacks that look like legitimate inputs, cost overruns from retry loops, and semantic drift that degrades over hours without throwing errors.

### Traditional vs. AI-Native Incident Response

```python
# Traditional incident response: Binary failure detection
class TraditionalMonitor:
    def check_health(self, response_time: float, error_rate: float) -> bool:
        """Clear thresholds, deterministic behavior"""
        if response_time > 500:  # ms
            self.alert("HIGH_LATENCY")
            return False
        if error_rate > 0.01:  # 1%
            self.alert("ERROR_THRESHOLD_EXCEEDED")
            return False
        return True
    
    def remediate(self, issue: str) -> None:
        """Deterministic remediation"""
        if issue == "HIGH_LATENCY":
            self.scale_up()
        elif issue == "ERROR_THRESHOLD_EXCEEDED":
            self.rollback_deployment()


# AI-native incident response: Probabilistic failure detection
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import numpy as np

@dataclass
class LLMIncident:
    timestamp: datetime
    incident_type: str
    severity: float  # 0.0-1.0, computed from multiple signals
    context: Dict
    sample_outputs: List[str]
    
class AIIncidentDetector:
    def __init__(self):
        self.baseline_metrics = {}
        self.semantic_drift_threshold = 0.15
        self.cost_anomaly_std = 3.0
        
    def check_health(
        self, 
        outputs: List[str],
        latencies: List[float],
        token_counts: List[int],
        user_feedback: List[Optional[bool]]
    ) -> Tuple[bool, Optional[LLMIncident]]:
        """Multi-dimensional health assessment"""
        
        # Traditional metrics
        avg_latency = np.mean(latencies)
        
        # AI-specific metrics
        semantic_consistency = self._measure_semantic_drift(outputs)
        cost_anomaly = self._detect_cost_anomaly(token_counts)
        quality_degradation = self._assess_quality(outputs, user_feedback)
        
        # Composite severity score
        severity = self._compute_severity({
            'latency': avg_latency,
            'semantic_drift': semantic_consistency,
            'cost_anomaly': cost_anomaly,
            'quality': quality_degradation
        })
        
        if severity > 0.7:
            incident = LLMIncident(
                timestamp=datetime.now(),
                incident_type=self._classify_incident(
                    semantic_consistency, cost_anomaly, quality_degradation
                ),
                severity=severity,
                context={
                    'latency_p95': np.percentile(latencies, 95),
                    'avg_tokens': np.mean(token_counts),
                    'drift_score': semantic_consistency
                },
                sample_outputs=outputs[-5:]  # Last 5 for analysis
            )
            return False, incident
        
        return True, None
    
    def _measure_semantic_drift(self, outputs: List[str]) -> float:
        """Detect when outputs diverge from expected patterns"""
        # Simplified: real implementation uses embeddings
        if not self.baseline_metrics.get('output_length'):
            self.baseline_metrics['output_length'] = np.mean([len(o) for o in outputs])
            return 0.0
        
        current_avg = np.mean([len(o) for o in outputs])
        baseline_avg = self.baseline_metrics['output_length']
        drift = abs(current_avg - baseline_avg) / baseline_avg
        return min(drift, 1.0)
    
    def _detect_cost_anomaly(self, token_counts: List[int]) -> float:
        """Detect runaway token usage"""
        if len(token_counts) < 10:
            return 0.0
            
        mean = np.mean(token_counts)
        std = np.std(token_counts)
        recent_mean = np.mean(token_counts[-5:])
        
        if std == 0:
            return 0.0
            
        z_score = abs(recent_mean - mean) / std
        return min(z_score / self.cost_anomaly_std, 1.0)
    
    def _assess_quality(
        self, 
        outputs: List[str], 
        feedback: List[Optional[bool]]
    ) -> float:
        """Assess output quality degradation"""
        if not any(f is not None for f in feedback):
            return 0.0
        
        valid_feedback = [f for f in feedback if f is not None]
        negative_rate = sum(1 for f in valid_feedback if not f) / len(valid_feedback)
        return negative_rate
    
    def _compute_severity(self, metrics: Dict[str, float]) -> float:
        """Weighted severity computation"""
        weights = {
            'latency': 0.2,
            'semantic_drift': 0.3,
            'cost_anomaly': 0.25,
            'quality': 0.25
        }
        
        normalized = {
            'latency': min(metrics['latency'] / 5000, 1.0),  # 5s max
            'semantic_drift': metrics['semantic_drift'],
            'cost_anomaly': metrics['cost_anomaly'],
            'quality': metrics['quality']
        }
        
        return sum(normalized[k] * weights[k] for k in weights)
    
    def _classify_incident(
        self, 
        drift: float, 
        cost: float, 
        quality: float
    ) -> str:
        """Classify incident type for targeted remediation"""
        if quality > 0.5:
            return "QUALITY_DEGRADATION"
        elif cost > 0.7:
            return "COST_RUNAWAY"
        elif drift > self.semantic_drift_threshold:
            return "SEMANTIC_DRIFT"
        else:
            return "COMPOSITE_FAILURE"
```

### Why AI Incident Response Differs

**Non-deterministic failures**: The same input can produce different outputs, some acceptable, some not. You can't simply replay requests to reproduce issues.

**Delayed manifestation**: Problems compound over time. A prompt that works 95% of the time creates 5% garbage that corrupts downstream systems, which manifests as failures hours later.

**Cascading semantic failures**: Unlike network partitions or database locks, AI failures cascade semantically—bad outputs become inputs to other systems, multiplying error states exponentially.

**No clear error boundaries**: A hallucinated API endpoint might be syntactically valid JSON that passes schema validation but calls a non-existent service. Traditional monitoring sees success; the application sees silent failure.

## Technical Components

### 1. Multi-Layered Detection System

AI incidents require detection at multiple abstraction layers because failures manifest differently depending on where you measure.

```python
from enum import Enum
from typing import Protocol
import hashlib
import json

class DetectionLayer(Enum):
    INFRASTRUCTURE = "infrastructure"
    MODEL_OUTPUT = "model_output"
    SEMANTIC = "semantic"
    BUSINESS_LOGIC = "business_logic"
    USER_IMPACT = "user_impact"

class DetectorProtocol(Protocol):
    def detect(self, data: Dict) -> Optional[Dict]:
        """Returns anomaly details if detected, None otherwise"""
        ...

class InfrastructureDetector:
    """Detects provider-level issues"""
    def __init__(self):
        self.latency_p99_threshold = 10000  # ms
        self.error_rate_threshold = 0.05
        
    def detect(self, data: Dict) -> Optional[Dict]:
        metrics = data.get('infrastructure_metrics', {})
        
        issues = []
        if metrics.get('latency_p99', 0) > self.latency_p99_threshold:
            issues.append(f"P99 latency: {metrics['latency_p99']}ms")
        
        if metrics.get('error_rate', 0) > self.error_rate_threshold:
            issues.append(f"Error rate: {metrics['error_rate']:.2%}")
        
        if metrics.get('rate_limited', False):
            issues.append("Rate limiting detected")
            
        if issues:
            return {
                'layer': DetectionLayer.INFRASTRUCTURE.value,
                'issues': issues,
                'severity': 0.9  # Infrastructure issues are critical
            }
        return None

class ModelOutputDetector:
    """Detects malformed or unexpected model outputs"""
    def __init__(self):
        self.expected_format_patterns = {}
        self.max_output_tokens = 4096
        
    def detect(self, data: Dict) -> Optional[Dict]:
        output = data.get('model_output', '')
        expected_format = data.get('expected_format')
        
        issues = []
        
        # Token overflow
        token_count = len(output.split())  # Simplified tokenization
        if token_count > self.max_output_tokens:
            issues.append(f"Token overflow: {token_count} tokens")
        
        # Format validation
        if expected_format == 'json':
            try:
                json.loads(output)
            except json.JSONDecodeError as e:
                issues.append(f"Invalid JSON: {e}")
        
        # Repetition detection
        if self._detect_repetition(output):
            issues.append("Excessive repetition detected")
        
        # Refusal patterns
        if self._is_refusal(output):
            issues.append("Model refusal detected")
            
        if issues:
            return {
                'layer': DetectionLayer.MODEL_OUTPUT.value,
                'issues': issues,
                'severity': 0.7,
                'sample': output[:200]
            }
        return None
    
    def _detect_repetition(self, text: str, threshold: float = 0.4) -> bool:
        """Detect if text contains excessive repetition"""
        words = text.split()
        if len(words) < 20:
            return False
        
        # Check for repeated n-grams
        ngram_size = 5
        ngrams = [tuple(words[i:i+ngram_size]) 
                  for i in range(len(words)-ngram_size)]
        unique_ratio = len(set(ngrams)) / len(ngrams) if ngrams else 1.0
        
        return unique_ratio < threshold
    
    def _is_refusal(self, text: str) -> bool:
        """Detect model refusal patterns"""
        refusal_patterns = [
            "i cannot", "i can't", "i'm not able to",
            "i don't have access", "as an ai",
            "i'm not authorized"
        ]
        text_lower = text.lower()
        return any(pattern in text_lower for pattern in refusal_patterns)

class SemanticDetector:
    """Detects semantic drift and hallucinations"""
    def __init__(self):
        self.reference_embeddings = {}
        self.drift_threshold = 0.2
        
    def detect(self, data: Dict) -> Optional[Dict]:
        output = data.get('model_output', '')
        context = data.get('context', '')
        reference_key = data.get('reference_key')
        
        issues = []
        
        # Hallucination detection via fact checking
        if context and self._contains_hallucination(output, context):
            issues.append("Potential hallucination detected")
        
        # Semantic drift from baseline
        if reference_key:
            drift = self._measure_drift(output, reference_key)
            if drift > self.drift_threshold:
                issues.append(f"Semantic drift: {drift:.2f}")
        
        # Contradiction detection
        if self._contains_contradiction(output):
            issues.append("Internal contradiction detected")
            
        if issues:
            return {
                'layer': DetectionLayer.SEMANTIC.value,
                'issues': issues,
                'severity': 0.8  # Semantic issues are serious
            }
        return None
    
    def _contains_hallucination(self, output: str, context: str) -> bool:
        """Simplified hallucination detection"""
        # Real implementation would use NLI models or fact-checking
        # This is a placeholder showing the pattern
        
        # Check if output references information not in context
        output_entities = set(output.lower().split())
        context_entities = set(context.lower().split())
        
        # If output contains many words not in context (excluding common words)
        common_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'in', 'on'}
        novel_entities = output_entities - context_entities - common_words
        
        # High ratio of novel entities might indicate hallucination
        return len(novel_entities) / max(len(output_entities), 1) > 0.7
    
    def _measure_drift(self, output: str, reference_key: str) -> float:
        """Measure semantic drift from baseline"""
        # Simplified: real implementation uses embedding similarity
        output_hash = hashlib.md5(output.encode()).hexdigest()
        
        if reference_key not in self.reference_embeddings:
            self.reference_embeddings[reference_key] = output_hash
            return 0.0
        
        # Placeholder: compare hashes (not semantically meaningful)
        reference_hash = self.reference_embeddings[reference_key]
        return 0.0 if output_hash == reference_hash else 0.3

    def _contains_contradiction(self, text: str) -> bool:
        """Detect internal contradictions"""
        # Real implementation would use NLI model
        # Simplified version checks for obvious negation patterns
        sentences = text.split('.')
        if len(sentences) < 2:
            return False
        
        negation_words = ['not', 'no', 'never', 'neither', 'none']
        affirmation_words = ['yes', 'is', 'are', 'always', 'definitely']
        
        has_negation = any(any(neg in s.lower() for neg in negation_words) 
                          for s in sentences)
        has_affirmation = any(any(aff in s.lower() for aff in affirmation_words) 
                             for s in sentences)
        
        return has_negation and has_affirmation

class MultiLayerMonitor:
    """Orchestrates detection across all layers"""
    def __init__(self):
        self.detectors = [
            InfrastructureDetector(),
            ModelOutputDetector(),
            SemanticDetector()
        ]
        
    def scan(self, data: Dict) -> List[Dict]:
        """Run all detectors and return all detected issues"""
        detections = []
        for detector in self.detectors:
            result = detector.detect(data)
            if result:
                detections.append(result)
        return detections
    
    def assess_overall_severity(self, detections: List[Dict]) -> float:
        """Compute composite severity score"""
        if not detections:
            return 0.0
        
        # Infrastructure issues are most critical
        max_severity = max(d['severity'] for d in detections)
        
        # Multiple simultaneous issues increase severity
        count_multiplier = 1 + (len(detections) - 1) * 0.2
        
        return min(max_severity * count_multiplier, 1.0)
```

**Practical implications**: You need specialized detectors at each layer because a single metric can't capture AI failure modes. Infrastructure might be healthy while outputs are semantically broken.

**Trade-offs**: More detection layers increase false positive rates and monitoring costs. Start with infrastructure and model output layers, add semantic detection only for critical paths.

### 2. Incident Classification and Routing

Not all incidents require the same response. Classifying