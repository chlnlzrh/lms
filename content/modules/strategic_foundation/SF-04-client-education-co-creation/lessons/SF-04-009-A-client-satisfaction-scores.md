# Client Satisfaction Scores: Engineering LLM Quality Measurement Systems

## Core Concepts

Client satisfaction scores in LLM systems are automated quality metrics that measure how well model outputs meet user requirements. Unlike traditional software testing where correctness is binary (test passes or fails), LLM outputs exist on a quality spectrum requiring probabilistic evaluation frameworks.

### Traditional vs. Modern Quality Measurement

```python
# Traditional Software Testing (Deterministic)
def validate_api_response(response: dict) -> bool:
    """Binary validation: passes or fails"""
    required_fields = ['id', 'name', 'email']
    return all(field in response for field in required_fields)

# Result: True or False - unambiguous

# LLM Quality Measurement (Probabilistic)
from typing import Dict, Any
import numpy as np

def evaluate_llm_response(
    response: str,
    criteria: Dict[str, float]
) -> Dict[str, Any]:
    """Multi-dimensional quality assessment"""
    scores = {
        'relevance': assess_relevance(response),      # 0.0-1.0
        'accuracy': assess_accuracy(response),        # 0.0-1.0
        'completeness': assess_completeness(response), # 0.0-1.0
        'coherence': assess_coherence(response),      # 0.0-1.0
        'safety': assess_safety(response)             # 0.0-1.0
    }
    
    weighted_score = sum(
        scores[k] * criteria.get(k, 0.2) 
        for k in scores
    )
    
    return {
        'overall_score': weighted_score,
        'dimension_scores': scores,
        'pass_threshold': weighted_score >= 0.7,
        'confidence': calculate_confidence(scores)
    }

# Result: Multi-dimensional quality profile with uncertainty
```

### Key Engineering Insights

**1. Quality is Multi-Dimensional:** A single score masks critical quality dimensions. Production systems require decomposed metrics tracking relevance, accuracy, safety, and user preference independently.

**2. Ground Truth is Expensive:** Unlike unit tests with known outputs, LLM evaluation often lacks absolute ground truth. You'll build evaluation systems that themselves use LLMs or human judgment.

**3. Metrics Drive Behavior:** What you measure becomes what you optimize. Poorly designed satisfaction scores create perverse incentives—models that game metrics rather than serving users.

**4. Evaluation Latency Matters:** Real-time satisfaction scoring adds 100-500ms to response time. Asynchronous evaluation patterns are essential for production systems.

### Why This Matters Now

Production LLM systems fail silently. A model can generate plausible-sounding garbage with perfect API response codes. Without automated satisfaction measurement:

- Silent quality degradation costs 3-10x more to fix after user complaints
- A/B testing requires 10-100x more traffic without quality pre-filtering
- Model updates ship blind—no automated regression detection
- Cost optimization is impossible without quality-cost trade-off data

Engineering teams shipping LLM products without satisfaction scoring infrastructure are flying blind through production.

## Technical Components

### 1. Scoring Model Architecture

Satisfaction scores come from three architectural patterns, each with distinct trade-offs:

**LLM-as-Judge Pattern:**

```python
from openai import OpenAI
from typing import Literal
import json

client = OpenAI()

def llm_judge_score(
    prompt: str,
    response: str,
    criteria: str
) -> Dict[str, Any]:
    """Use LLM to evaluate another LLM's output"""
    
    judge_prompt = f"""Evaluate this AI response against the criteria.

User Prompt: {prompt}

AI Response: {response}

Evaluation Criteria: {criteria}

Provide scores (0-100) for:
1. Relevance: Does it address the prompt?
2. Accuracy: Is information correct?
3. Completeness: Are all aspects covered?
4. Clarity: Is it well-structured and clear?

Return JSON: {{"relevance": X, "accuracy": X, "completeness": X, "clarity": X, "reasoning": "brief explanation"}}"""

    try:
        result = client.chat.completions.create(
            model="gpt-4o-mini",  # Fast, cheap evaluator
            messages=[{"role": "user", "content": judge_prompt}],
            temperature=0.0,  # Deterministic evaluation
            response_format={"type": "json_object"}
        )
        
        scores = json.loads(result.choices[0].message.content)
        overall = np.mean([
            scores['relevance'],
            scores['accuracy'],
            scores['completeness'],
            scores['clarity']
        ])
        
        return {
            'overall_score': overall / 100.0,
            'dimensions': {k: v/100.0 for k, v in scores.items() if k != 'reasoning'},
            'reasoning': scores.get('reasoning', ''),
            'eval_latency_ms': result.usage.total_tokens * 0.5  # Estimate
        }
    except Exception as e:
        return {'error': str(e), 'overall_score': 0.0}
```

**Trade-offs:**
- Latency: 200-800ms per evaluation
- Cost: $0.0001-0.001 per evaluation
- Quality: Correlates 0.7-0.85 with human judgment
- Scalability: Limited by API rate limits

**Embedding Similarity Pattern:**

```python
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

client = OpenAI()

def embedding_similarity_score(
    response: str,
    reference_responses: list[str],
    semantic_weight: float = 0.7
) -> Dict[str, Any]:
    """Fast semantic similarity to known-good responses"""
    
    # Get embeddings
    texts = [response] + reference_responses
    result = client.embeddings.create(
        model="text-embedding-3-small",
        input=texts
    )
    
    embeddings = np.array([e.embedding for e in result.data])
    response_emb = embeddings[0].reshape(1, -1)
    reference_embs = embeddings[1:]
    
    # Calculate similarities
    similarities = cosine_similarity(response_emb, reference_embs)[0]
    max_similarity = np.max(similarities)
    avg_similarity = np.mean(similarities)
    
    # Lexical overlap (simple token intersection)
    response_tokens = set(response.lower().split())
    lexical_scores = []
    for ref in reference_responses:
        ref_tokens = set(ref.lower().split())
        overlap = len(response_tokens & ref_tokens) / len(response_tokens | ref_tokens)
        lexical_scores.append(overlap)
    
    lexical_score = np.max(lexical_scores)
    
    # Combined score
    overall = (semantic_weight * max_similarity + 
               (1 - semantic_weight) * lexical_score)
    
    return {
        'overall_score': float(overall),
        'max_semantic_similarity': float(max_similarity),
        'avg_semantic_similarity': float(avg_similarity),
        'lexical_overlap': float(lexical_score),
        'eval_latency_ms': 50  # Typically 30-100ms
    }
```

**Trade-offs:**
- Latency: 30-100ms per evaluation
- Cost: $0.000001-0.00001 per evaluation (100x cheaper)
- Quality: Correlates 0.5-0.7 with human judgment
- Scalability: Handles 1000+ QPS easily

**Learned Classifier Pattern:**

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import pickle

class LearnedSatisfactionClassifier:
    """Train custom classifier on historical ratings"""
    
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            max_features=500,
            ngram_range=(1, 3)
        )
        self.classifier = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        self.is_trained = False
    
    def train(
        self,
        responses: list[str],
        satisfaction_scores: list[float],  # 0.0-1.0 from human ratings
        prompts: list[str] = None
    ):
        """Train on historical human-rated data"""
        
        # Feature engineering
        features = []
        for i, response in enumerate(responses):
            text_features = self._extract_features(response)
            if prompts:
                text_features.update(
                    self._extract_prompt_response_features(
                        prompts[i], response
                    )
                )
            features.append(text_features)
        
        # Vectorize
        feature_names = list(features[0].keys())
        X_custom = np.array([[f[k] for k in feature_names] for f in features])
        X_text = self.vectorizer.fit_transform(responses).toarray()
        X = np.hstack([X_custom, X_text])
        
        # Convert scores to classes (low/med/high satisfaction)
        y = np.digitize(satisfaction_scores, bins=[0.4, 0.7])
        
        self.classifier.fit(X, y)
        self.feature_names = feature_names
        self.is_trained = True
    
    def score(
        self,
        response: str,
        prompt: str = None
    ) -> Dict[str, Any]:
        """Predict satisfaction score for new response"""
        
        if not self.is_trained:
            raise ValueError("Model not trained")
        
        # Extract same features
        custom_features = self._extract_features(response)
        if prompt:
            custom_features.update(
                self._extract_prompt_response_features(prompt, response)
            )
        
        X_custom = np.array([[custom_features[k] for k in self.feature_names]])
        X_text = self.vectorizer.transform([response]).toarray()
        X = np.hstack([X_custom, X_text])
        
        # Predict
        class_probs = self.classifier.predict_proba(X)[0]
        predicted_score = np.dot(class_probs, [0.2, 0.6, 0.9])  # Map classes to scores
        
        return {
            'overall_score': float(predicted_score),
            'confidence': float(np.max(class_probs)),
            'class_distribution': {
                'low': float(class_probs[0]),
                'medium': float(class_probs[1]),
                'high': float(class_probs[2])
            },
            'eval_latency_ms': 5  # Extremely fast
        }
    
    def _extract_features(self, response: str) -> Dict[str, float]:
        """Domain-specific feature engineering"""
        return {
            'length': len(response),
            'word_count': len(response.split()),
            'avg_word_length': np.mean([len(w) for w in response.split()]),
            'sentence_count': response.count('.') + response.count('!') + response.count('?'),
            'has_code': 1.0 if '```' in response else 0.0,
            'has_list': 1.0 if any(c in response for c in ['- ', '1.', '* ']) else 0.0,
            'question_marks': response.count('?') / max(len(response), 1),
        }
    
    def _extract_prompt_response_features(
        self, prompt: str, response: str
    ) -> Dict[str, float]:
        """Features relating prompt to response"""
        prompt_tokens = set(prompt.lower().split())
        response_tokens = set(response.lower().split())
        
        return {
            'token_overlap': len(prompt_tokens & response_tokens) / len(prompt_tokens),
            'length_ratio': len(response) / max(len(prompt), 1),
        }
```

**Trade-offs:**
- Latency: 1-10ms per evaluation
- Cost: Nearly zero at scale
- Quality: Correlates 0.6-0.8 with human judgment (domain-dependent)
- Scalability: 10,000+ QPS on single CPU

### 2. Satisfaction Score Dimensions

Production systems track multiple satisfaction dimensions independently:

```python
from dataclasses import dataclass
from typing import Optional
import re

@dataclass
class SatisfactionProfile:
    """Multi-dimensional quality assessment"""
    relevance: float  # Addresses the prompt
    accuracy: float   # Factually correct
    completeness: float  # Covers all aspects
    clarity: float    # Well-structured
    safety: float     # No harmful content
    efficiency: float  # Concise, no fluff
    
    overall: Optional[float] = None
    
    def __post_init__(self):
        if self.overall is None:
            # Weighted average - tune weights for your domain
            self.overall = (
                0.25 * self.relevance +
                0.25 * self.accuracy +
                0.20 * self.completeness +
                0.15 * self.clarity +
                0.10 * self.safety +
                0.05 * self.efficiency
            )

def comprehensive_evaluation(
    prompt: str,
    response: str,
    context: Optional[Dict[str, Any]] = None
) -> SatisfactionProfile:
    """Evaluate response across all dimensions"""
    
    # Relevance: semantic similarity to prompt intent
    relevance = evaluate_relevance(prompt, response)
    
    # Accuracy: check claims against knowledge base or search
    accuracy = evaluate_accuracy(response, context)
    
    # Completeness: covers all prompt aspects
    completeness = evaluate_completeness(prompt, response)
    
    # Clarity: structure, readability
    clarity = evaluate_clarity(response)
    
    # Safety: harmful content detection
    safety = evaluate_safety(response)
    
    # Efficiency: information density
    efficiency = evaluate_efficiency(response)
    
    return SatisfactionProfile(
        relevance=relevance,
        accuracy=accuracy,
        completeness=completeness,
        clarity=clarity,
        safety=safety,
        efficiency=efficiency
    )

def evaluate_relevance(prompt: str, response: str) -> float:
    """Quick heuristic-based relevance check"""
    # Extract key terms from prompt
    prompt_terms = set(re.findall(r'\b\w{4,}\b', prompt.lower()))
    response_terms = set(re.findall(r'\b\w{4,}\b', response.lower()))
    
    if not prompt_terms:
        return 1.0
    
    overlap = len(prompt_terms & response_terms) / len(prompt_terms)
    return min(1.0, overlap * 1.5)  # Boost score, cap at 1.0

def evaluate_accuracy(
    response: str,
    context: Optional[Dict[str, Any]]
) -> float:
    """Check factual accuracy against provided context"""
    if not context or 'facts' not in context:
        return 0.8  # Neutral score when can't verify
    
    # Simple fact-checking: verify key claims appear in context
    claims = extract_claims(response)
    verified = 0
    
    for claim in claims:
        if any(fact.lower() in claim.lower() for fact in context['facts']):
            verified += 1
    
    return verified / max(len(claims), 1)

def evaluate_completeness(prompt: str, response: str) -> float:
    """Check if all prompt aspects addressed"""
    # Detect questions in prompt
    questions = [q.strip() for q in re.split(r'[.!?]', prompt) if '?' in q]
    
    if not questions:
        return 1.0  # No specific questions to answer
    
    # Check if response addresses each question area
    addressed = 0
    for question in questions:
        key_terms = set(re.findall(r'\b\w{4,}\b', question.lower()))
        response_terms = set(re.findall(r'\b\w{4,