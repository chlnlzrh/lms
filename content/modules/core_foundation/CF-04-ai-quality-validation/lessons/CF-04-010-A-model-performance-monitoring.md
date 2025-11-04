# Model Performance Monitoring for LLM Systems

## Core Concepts

Model performance monitoring in LLM systems involves tracking behavioral characteristics, response quality, and operational metrics of language models in production. Unlike traditional ML monitoring that focuses on prediction accuracy against labeled datasets, LLM monitoring addresses unique challenges: evaluation without ground truth, detecting subtle quality degradation, tracking emergent failures, and measuring user satisfaction signals.

### Traditional vs. LLM Monitoring Architecture

```python
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import json

# Traditional ML Monitoring (Classification Model)
@dataclass
class TraditionalModelMetrics:
    """Simple metrics with clear ground truth"""
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    prediction_latency_ms: float
    
    def is_degraded(self, baseline: 'TraditionalModelMetrics') -> bool:
        return self.accuracy < baseline.accuracy * 0.95  # Simple threshold

# LLM Monitoring (Generation Model)
@dataclass
class LLMMetrics:
    """Complex, multi-dimensional quality assessment"""
    # Response quality (no single ground truth)
    semantic_similarity_score: Optional[float]
    toxicity_score: float
    factual_consistency_score: Optional[float]
    hallucination_indicators: List[str]
    
    # Behavioral metrics
    response_length_tokens: int
    refusal_rate: float
    instruction_following_score: Optional[float]
    
    # User signals
    user_feedback_score: Optional[float]
    retry_rate: float
    edit_distance_on_retry: Optional[int]
    
    # Operational
    latency_p50_ms: float
    latency_p99_ms: float
    tokens_per_second: float
    cost_per_request: float
    
    # Drift detection
    topic_distribution_divergence: float
    output_format_compliance: float
    
    def detect_anomalies(self, baseline: 'LLMMetrics', 
                        context: Dict) -> List[str]:
        """Multi-signal anomaly detection"""
        anomalies = []
        
        # Behavioral drift
        if abs(self.response_length_tokens - baseline.response_length_tokens) > 100:
            anomalies.append("response_length_drift")
            
        # Quality degradation (composite signal)
        if self.toxicity_score > baseline.toxicity_score * 1.5:
            anomalies.append("toxicity_increase")
            
        # User dissatisfaction signals
        if self.retry_rate > baseline.retry_rate * 1.3:
            anomalies.append("high_retry_rate")
            
        # Performance regression
        if self.latency_p99_ms > baseline.latency_p99_ms * 1.2:
            anomalies.append("latency_degradation")
            
        return anomalies
```

### Engineering Insight: The Ground Truth Problem

Traditional ML monitoring assumes labeled data for continuous evaluation. LLMs operate in open-ended generation spaces where:

1. **Multiple valid outputs exist** - Same input can have dozens of acceptable responses
2. **Quality is contextual** - A "good" response depends on user intent, domain, and downstream usage
3. **Failure modes are subtle** - Model doesn't "crash," it produces plausible but wrong or unhelpful text
4. **Evaluation is expensive** - Human review doesn't scale; automated metrics are noisy proxies

This forces a shift from "accuracy monitoring" to "behavioral fingerprinting" - tracking patterns in model behavior rather than correctness against labels.

### Why This Matters Now

Production LLM systems fail silently and gradually. A model update might:
- Increase refusal rates for legitimate requests (over-alignment)
- Shift response style without changing "correctness"
- Degrade on specific demographic groups or domains
- Develop unexpected sensitivities to prompt phrasing

These issues don't trigger traditional alerts. Without comprehensive monitoring, you discover problems through user complaints, not metrics. The challenge: building observability for systems with fuzzy success criteria and no ground truth.

## Technical Components

### 1. Real-Time Quality Scoring Pipeline

LLM responses need multi-dimensional quality assessment at inference time, balancing signal quality with latency overhead.

```python
from typing import Protocol, Callable
import asyncio
from concurrent.futures import ThreadPoolExecutor
import hashlib

class QualityScorer(Protocol):
    """Interface for quality scoring functions"""
    def score(self, prompt: str, response: str, context: Dict) -> float:
        ...

class QualityScoringPipeline:
    """Parallel quality assessment with latency budgets"""
    
    def __init__(self, 
                 scorers: Dict[str, Tuple[QualityScorer, int]],
                 executor_pool_size: int = 4):
        """
        Args:
            scorers: Dict of {metric_name: (scorer_fn, timeout_ms)}
            executor_pool_size: Thread pool for blocking scorers
        """
        self.scorers = scorers
        self.executor = ThreadPoolExecutor(max_workers=executor_pool_size)
        self._cache = {}  # Response hash -> scores
        
    async def score_response(self, 
                            prompt: str, 
                            response: str, 
                            context: Dict) -> Dict[str, Optional[float]]:
        """Execute all scorers in parallel with timeouts"""
        
        # Cache check for identical responses
        cache_key = hashlib.md5(f"{prompt}:{response}".encode()).hexdigest()
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        # Launch all scorers concurrently
        tasks = []
        for metric_name, (scorer, timeout_ms) in self.scorers.items():
            task = asyncio.create_task(
                self._score_with_timeout(
                    scorer, prompt, response, context, 
                    timeout_ms, metric_name
                )
            )
            tasks.append((metric_name, task))
        
        # Collect results
        scores = {}
        for metric_name, task in tasks:
            try:
                scores[metric_name] = await task
            except asyncio.TimeoutError:
                scores[metric_name] = None  # Timeout -> missing metric
            except Exception as e:
                scores[metric_name] = None
                # Log error but don't block request
                
        self._cache[cache_key] = scores
        return scores
    
    async def _score_with_timeout(self,
                                  scorer: QualityScorer,
                                  prompt: str,
                                  response: str,
                                  context: Dict,
                                  timeout_ms: int,
                                  metric_name: str) -> float:
        """Execute scorer with timeout"""
        loop = asyncio.get_event_loop()
        
        # Run in thread pool if blocking
        score_future = loop.run_in_executor(
            self.executor,
            scorer.score,
            prompt, response, context
        )
        
        return await asyncio.wait_for(
            score_future, 
            timeout=timeout_ms / 1000.0
        )

# Example scorer implementations
class ToxicityScorer:
    """Fast regex-based toxicity detection"""
    
    def __init__(self):
        # In production: use model like Detoxify or Perspective API
        self.toxic_patterns = [
            r'\b(profanity|slur|offensive_term)\b',
            # ... more patterns
        ]
        
    def score(self, prompt: str, response: str, context: Dict) -> float:
        import re
        toxic_count = sum(
            len(re.findall(pattern, response.lower()))
            for pattern in self.toxic_patterns
        )
        return min(1.0, toxic_count / 10.0)  # 0-1 scale

class SemanticCoherenceScorer:
    """Embedding-based prompt-response alignment"""
    
    def __init__(self, embedding_model):
        self.model = embedding_model
        
    def score(self, prompt: str, response: str, context: Dict) -> float:
        # Compute embeddings
        prompt_emb = self.model.encode(prompt)
        response_emb = self.model.encode(response)
        
        # Cosine similarity
        similarity = (prompt_emb @ response_emb) / (
            (prompt_emb @ prompt_emb) ** 0.5 * 
            (response_emb @ response_emb) ** 0.5
        )
        return float(similarity)

class ResponseLengthScorer:
    """Check if response meets length expectations"""
    
    def __init__(self, expected_min: int = 50, expected_max: int = 500):
        self.min = expected_min
        self.max = expected_max
        
    def score(self, prompt: str, response: str, context: Dict) -> float:
        length = len(response.split())
        if length < self.min:
            return 0.5 * (length / self.min)
        elif length > self.max:
            return 0.5 * (self.max / length)
        return 1.0
```

**Practical Implications:**
- Parallel execution keeps latency overhead under 100ms for multiple scorers
- Timeouts prevent slow scorers from blocking user requests
- Caching eliminates redundant scoring for identical responses
- Missing scores (timeouts/errors) degrade gracefully without breaking monitoring

**Trade-offs:**
- Fast heuristic scorers (regex, length) are noisy but cheap
- Model-based scorers (embeddings, classifiers) are accurate but add 50-200ms latency
- Balancing signal quality vs. request latency requires careful scorer selection

### 2. Behavioral Fingerprinting for Drift Detection

Track statistical distributions of model outputs to detect subtle behavioral changes without ground truth labels.

```python
from collections import defaultdict
from scipy.stats import ks_2samp, entropy
import numpy as np
from typing import List

class BehavioralFingerprint:
    """Statistical signature of model behavior"""
    
    def __init__(self, window_size: int = 1000):
        self.window_size = window_size
        
        # Response characteristics
        self.response_lengths: List[int] = []
        self.token_distributions: Dict[str, int] = defaultdict(int)
        self.sentiment_scores: List[float] = []
        self.refusal_indicators: List[bool] = []
        
        # Prompt-response relationships
        self.similarity_scores: List[float] = []
        self.response_times_ms: List[float] = []
        
    def update(self, 
               prompt: str, 
               response: str, 
               metadata: Dict):
        """Add observation to fingerprint"""
        
        # Response length distribution
        tokens = response.split()
        self.response_lengths.append(len(tokens))
        
        # Token usage patterns (top-k vocabulary)
        for token in tokens[:100]:  # Limit for efficiency
            self.token_distributions[token] += 1
            
        # Refusal patterns
        refusal_phrases = [
            "I cannot", "I'm unable", "I don't have access",
            "I can't", "I apologize, but"
        ]
        is_refusal = any(phrase in response for phrase in refusal_phrases)
        self.refusal_indicators.append(is_refusal)
        
        # Metadata
        if 'similarity_score' in metadata:
            self.similarity_scores.append(metadata['similarity_score'])
        if 'response_time_ms' in metadata:
            self.response_times_ms.append(metadata['response_time_ms'])
            
        # Sliding window
        self._truncate_window()
        
    def _truncate_window(self):
        """Keep only recent observations"""
        if len(self.response_lengths) > self.window_size:
            overflow = len(self.response_lengths) - self.window_size
            self.response_lengths = self.response_lengths[overflow:]
            self.sentiment_scores = self.sentiment_scores[overflow:]
            self.refusal_indicators = self.refusal_indicators[overflow:]
            self.similarity_scores = self.similarity_scores[overflow:]
            self.response_times_ms = self.response_times_ms[overflow:]
            
    def compare(self, baseline: 'BehavioralFingerprint') -> Dict[str, float]:
        """Statistical comparison between fingerprints"""
        
        divergences = {}
        
        # Length distribution shift (Kolmogorov-Smirnov test)
        if self.response_lengths and baseline.response_lengths:
            ks_stat, p_value = ks_2samp(
                self.response_lengths,
                baseline.response_lengths
            )
            divergences['length_distribution_ks'] = ks_stat
            divergences['length_distribution_p_value'] = p_value
            
        # Token distribution shift (Jensen-Shannon divergence)
        all_tokens = set(self.token_distributions.keys()) | \
                     set(baseline.token_distributions.keys())
        
        current_dist = np.array([
            self.token_distributions.get(t, 0) for t in all_tokens
        ])
        baseline_dist = np.array([
            baseline.token_distributions.get(t, 0) for t in all_tokens
        ])
        
        # Normalize
        current_dist = current_dist / (current_dist.sum() + 1e-10)
        baseline_dist = baseline_dist / (baseline_dist.sum() + 1e-10)
        
        # JS divergence
        m = 0.5 * (current_dist + baseline_dist)
        js_div = 0.5 * (entropy(current_dist, m) + entropy(baseline_dist, m))
        divergences['token_distribution_js'] = float(js_div)
        
        # Refusal rate change
        current_refusal_rate = np.mean(self.refusal_indicators)
        baseline_refusal_rate = np.mean(baseline.refusal_indicators)
        divergences['refusal_rate_delta'] = \
            current_refusal_rate - baseline_refusal_rate
            
        # Response time shift
        if self.response_times_ms and baseline.response_times_ms:
            divergences['latency_p50_delta'] = \
                np.percentile(self.response_times_ms, 50) - \
                np.percentile(baseline.response_times_ms, 50)
            divergences['latency_p99_delta'] = \
                np.percentile(self.response_times_ms, 99) - \
                np.percentile(baseline.response_times_ms, 99)
                
        return divergences
    
    def detect_drift(self, 
                     baseline: 'BehavioralFingerprint',
                     thresholds: Dict[str, float]) -> List[str]:
        """Detect significant behavioral drift"""
        
        divergences = self.compare(baseline)
        alerts = []
        
        # Check each metric against thresholds
        if divergences.get('length_distribution_p_value', 1.0) < 0.01:
            alerts.append(
                f"Significant length distribution shift "
                f"(KS={divergences['length_distribution_ks']:.3f})"
            )
            
        if divergences.get('token_distribution_js', 0) > \
           thresholds.get('token_js_threshold', 0.1):
            alerts.append(
                f"Token distribution drift "
                f"(JS={divergences['token_distribution_js']:.3f})"
            )
            
        refusal_delta = abs(divergences.get('refusal_rate_delta', 0))
        if refusal_delta > thresholds.get('refusal_rate_threshold', 0.05):
            alerts.append(
                f"Refusal rate change: {refusal_delta:+.1%}"
            )
            
        return alerts
```

**Practical Implications:**
- Detects model behavior changes without labeled data
- Statistical tests (KS, JS divergence) provide objective drift signals
- Sliding windows adapt to gradual distribution shifts
- Catches subtle issues like vocabulary narrowing or style changes

**Constraints:**
- Requires sufficient sample size (