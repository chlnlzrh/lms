# Confidence Scoring & Uncertainty Quantification

## Core Concepts

Large language models don't think—they predict token probability distributions. Every output is a series of weighted guesses, yet most production systems treat these guesses as deterministic outputs. This creates a fundamental engineering problem: how do you build reliable systems on top of probabilistic foundations?

Confidence scoring and uncertainty quantification transform LLM outputs from opaque predictions into measurable statistical artifacts you can reason about, route, and act upon. This isn't about making models "more confident"—it's about extracting and operationalizing the uncertainty signal that already exists in the model's internal state.

### Traditional vs. Modern Approach

```python
# Traditional approach: Binary trust
def extract_entities_old(text: str, llm_client) -> list[str]:
    """Treat LLM output as ground truth."""
    response = llm_client.generate(
        f"Extract all person names from: {text}"
    )
    entities = response.split(",")
    return entities  # Hope it's correct

# Modern approach: Quantified uncertainty
from typing import TypedDict
import numpy as np

class EntityWithConfidence(TypedDict):
    entity: str
    confidence: float
    token_logprobs: list[float]
    method: str

def extract_entities_modern(
    text: str, 
    llm_client,
    confidence_threshold: float = 0.7
) -> tuple[list[EntityWithConfidence], list[str]]:
    """Route outputs based on measured uncertainty."""
    response = llm_client.generate(
        f"Extract all person names from: {text}",
        logprobs=5,  # Get top-5 token probabilities
        temperature=0.0
    )
    
    entities_high_conf = []
    entities_review = []
    
    for entity_span in parse_entities(response):
        # Aggregate token-level uncertainty
        token_probs = [
            token.logprob for token in entity_span.tokens
        ]
        mean_logprob = np.mean(token_probs)
        confidence = np.exp(mean_logprob)
        
        entity_obj: EntityWithConfidence = {
            "entity": entity_span.text,
            "confidence": confidence,
            "token_logprobs": token_probs,
            "method": "token_aggregation"
        }
        
        if confidence >= confidence_threshold:
            entities_high_conf.append(entity_obj)
        else:
            entities_review.append(entity_obj["entity"])
    
    return entities_high_conf, entities_review
```

The modern approach exposes three critical dimensions: the prediction itself, its uncertainty level, and a routing mechanism. This enables differential treatment—auto-accept high-confidence outputs, flag uncertain cases for review, and route edge cases to fallback strategies.

### Key Engineering Insights

**Insight 1: Logprobs are your ground truth.** Token-level log probabilities are the only direct measurement of model uncertainty. Every other metric (verbal confidence, repetition, self-consistency) is a proxy that must be validated against logprobs.

**Insight 2: Uncertainty compounds across tokens.** A response with 50 tokens at 90% confidence each has only 0.9^50 = 0.5% chance of being entirely correct. Sequence-level confidence must account for error propagation.

**Insight 3: Calibration beats raw probability.** A model reporting 80% confidence might be empirically correct 60% or 95% of the time. Calibration curves map stated confidence to observed accuracy—essential for production decision-making.

**Insight 4: Semantic uncertainty differs from lexical uncertainty.** Low token probability might indicate genuinely uncertain predictions OR valid but uncommon phrasings. Semantic clustering reveals whether uncertainty represents knowledge gaps or stylistic variation.

### Why This Matters Now

Production LLM systems fail predictably at the uncertainty boundary. Without quantified confidence, you're forced into binary decisions: trust everything (high error rates) or trust nothing (manual review bottlenecks). Modern systems demand granular routing—automatic handling for confident predictions, human-in-the-loop for uncertain cases, and fallback strategies for low-confidence outputs.

The economics are stark. Manual review costs $50-200 per hour. LLM inference costs $0.001-0.10 per request. A calibrated confidence system that auto-routes 70% of requests (vs. 20% with binary trust) saves 50% of review costs while improving accuracy on routed cases. This transforms LLMs from expensive assistants to scalable automation.

## Technical Components

### 1. Token-Level Log Probability Extraction

Log probabilities represent the model's raw uncertainty signal—the negative log-likelihood of each generated token given preceding context. These values range from 0 (certainty) to negative infinity (impossibility), with practical values typically between -0.01 and -10.

```python
from dataclasses import dataclass
import math

@dataclass
class TokenInfo:
    token: str
    logprob: float
    linear_prob: float
    rank: int  # Rank among all possible tokens
    top_alternatives: list[tuple[str, float]]

def extract_token_confidence(
    response_obj, 
    include_alternatives: bool = True
) -> list[TokenInfo]:
    """
    Extract per-token uncertainty from API response.
    
    Practical implications:
    - Tokens with logprob < -2.0 indicate high uncertainty
    - Large gaps between top choice and alternatives suggest
      multiple plausible completions
    - Rank position reveals whether choice was obvious or marginal
    """
    tokens = []
    
    for token_data in response_obj.choices[0].logprobs.content:
        logprob = token_data.logprob
        linear_prob = math.exp(logprob)
        
        alternatives = []
        if include_alternatives and token_data.top_logprobs:
            alternatives = [
                (alt.token, math.exp(alt.logprob))
                for alt in token_data.top_logprobs[1:6]
            ]
        
        tokens.append(TokenInfo(
            token=token_data.token,
            logprob=logprob,
            linear_prob=linear_prob,
            rank=compute_rank(token_data),
            top_alternatives=alternatives
        ))
    
    return tokens

def compute_rank(token_data) -> int:
    """Estimate rank from top-k logprobs."""
    if not token_data.top_logprobs:
        return 1  # Assume top if no alternatives
    
    # Count how many alternatives have higher probability
    selected_logprob = token_data.logprob
    rank = 1
    for alt in token_data.top_logprobs:
        if alt.logprob > selected_logprob:
            rank += 1
    return rank
```

**Practical constraints:**
- Not all APIs expose logprobs (proprietary models often hide them)
- Top-k logprobs (typically k=5-10) show only most likely alternatives
- Special tokens (BOS, EOS, padding) may have misleading probabilities
- Temperature/sampling affects probability distribution shape

**Real-world trade-off:** Requesting logprobs increases response size by 3-10x and latency by 5-15%. Cache token statistics for repeated patterns to amortize cost.

### 2. Sequence-Level Confidence Aggregation

Individual token probabilities must be aggregated into sequence-level confidence scores. Different aggregation strategies expose different uncertainty dimensions.

```python
from typing import Literal
import numpy as np
from scipy import stats

AggregationMethod = Literal[
    "geometric_mean", "min", "harmonic_mean", "entropy"
]

def compute_sequence_confidence(
    token_logprobs: list[float],
    method: AggregationMethod = "geometric_mean",
    normalize: bool = True
) -> float:
    """
    Aggregate token-level uncertainty into sequence score.
    
    Method selection guide:
    - geometric_mean: Balanced, standard choice for most cases
    - min: Conservative, flags sequences with any uncertain token
    - harmonic_mean: Emphasizes worst cases more than geometric
    - entropy: Measures predictability of distribution
    """
    if not token_logprobs:
        return 0.0
    
    probs = np.exp(token_logprobs)
    
    if method == "geometric_mean":
        # Product of probabilities, nth root
        # P(sequence) = P(t1) * P(t2) * ... * P(tn)
        log_prob_sum = np.sum(token_logprobs)
        score = np.exp(log_prob_sum / len(token_logprobs))
        
    elif method == "min":
        # Most uncertain token determines sequence confidence
        score = np.min(probs)
        
    elif method == "harmonic_mean":
        # More sensitive to low values than geometric mean
        score = len(probs) / np.sum(1.0 / (probs + 1e-10))
        
    elif method == "entropy":
        # Measure predictability of token distribution
        # Higher entropy = more uniform = more uncertain
        entropy = -np.sum(probs * np.log(probs + 1e-10))
        max_entropy = np.log(len(probs))
        score = 1.0 - (entropy / max_entropy)
    
    if normalize:
        # Scale to [0, 1] range accounting for sequence length
        length_penalty = np.exp(-len(token_logprobs) / 100)
        score = score * (1.0 - length_penalty * 0.3)
    
    return float(score)

# Example: Compare methods on real sequence
token_logprobs = [-0.1, -0.05, -2.5, -0.08, -0.12]  # One uncertain token

results = {
    method: compute_sequence_confidence(token_logprobs, method)
    for method in ["geometric_mean", "min", "harmonic_mean", "entropy"]
}
# geometric_mean: 0.42  (balanced)
# min: 0.08             (flags single uncertain token)
# harmonic_mean: 0.29   (emphasizes uncertainty)
# entropy: 0.65         (overall distribution predictability)
```

**Key insight:** Geometric mean balances individual token confidence with sequence length. Min confidence is appropriate for extraction tasks where any error invalidates the result. Entropy captures distribution shape but loses positional information.

### 3. Calibration and Temperature Scaling

Raw model probabilities are often miscalibrated—a stated 80% confidence might correspond to 60% or 95% empirical accuracy. Calibration transforms raw scores into actionable probabilities.

```python
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
import numpy as np

class ConfidenceCalibrator:
    """
    Calibrate raw model confidence scores to empirical accuracy.
    
    Two strategies:
    - Platt scaling: Fit logistic regression (parametric)
    - Isotonic regression: Fit monotonic function (non-parametric)
    """
    
    def __init__(self, method: Literal["platt", "isotonic"] = "isotonic"):
        self.method = method
        self.calibrator = None
        self.bins = 20
        self.calibration_curve = None
    
    def fit(self, confidences: np.ndarray, correctness: np.ndarray):
        """
        Train calibrator on validation set.
        
        Args:
            confidences: Raw model confidence scores [0, 1]
            correctness: Binary accuracy labels (1=correct, 0=wrong)
        """
        confidences = np.array(confidences).reshape(-1, 1)
        correctness = np.array(correctness)
        
        if self.method == "platt":
            # Fit logistic regression on logit-transformed scores
            self.calibrator = LogisticRegression()
            self.calibrator.fit(confidences, correctness)
            
        elif self.method == "isotonic":
            # Fit isotonic (monotonically increasing) regression
            self.calibrator = IsotonicRegression(out_of_bounds="clip")
            self.calibrator.fit(
                confidences.ravel(), 
                correctness
            )
        
        # Compute calibration curve for visualization
        self._compute_calibration_curve(confidences, correctness)
    
    def _compute_calibration_curve(
        self, 
        confidences: np.ndarray, 
        correctness: np.ndarray
    ):
        """Bin predictions and compute empirical accuracy per bin."""
        bins = np.linspace(0, 1, self.bins + 1)
        bin_centers = (bins[:-1] + bins[1:]) / 2
        
        empirical_probs = []
        for i in range(self.bins):
            mask = (confidences >= bins[i]) & (confidences < bins[i+1])
            if np.sum(mask) > 0:
                empirical_probs.append(np.mean(correctness[mask]))
            else:
                empirical_probs.append(np.nan)
        
        self.calibration_curve = (bin_centers, empirical_probs)
    
    def calibrate(self, raw_confidence: float) -> float:
        """Transform raw confidence into calibrated probability."""
        if self.calibrator is None:
            raise ValueError("Calibrator not fitted. Call fit() first.")
        
        conf_array = np.array([[raw_confidence]])
        
        if self.method == "platt":
            calibrated = self.calibrator.predict_proba(conf_array)[0, 1]
        else:
            calibrated = self.calibrator.predict(conf_array.ravel())[0]
        
        return float(np.clip(calibrated, 0.0, 1.0))
    
    def expected_calibration_error(
        self, 
        confidences: np.ndarray, 
        correctness: np.ndarray
    ) -> float:
        """
        Compute ECE: weighted average of |confidence - accuracy| per bin.
        Lower is better (0 = perfect calibration).
        """
        bins = np.linspace(0, 1, self.bins + 1)
        ece = 0.0
        
        for i in range(self.bins):
            mask = (confidences >= bins[i]) & (confidences < bins[i+1])
            if np.sum(mask) == 0:
                continue
            
            bin_confidence = np.mean(confidences[mask])
            bin_accuracy = np.mean(correctness[mask])
            bin_weight = np.sum(mask) / len(confidences)
            
            ece += bin_weight * abs(bin_confidence - bin_accuracy)
        
        return ece

# Example: Calibrate overconfident model
np.random.seed(42)
n_samples = 1000

# Simulate overconfident model (reports higher confidence than warranted)
raw_confidences = np.random.beta(8, 2, n_samples)  # Skewed toward high values
true_accuracy = raw_confidences * 0.7 + np.random.normal(0, 0.1, n_samples)
correctness = (np.random.random(n_samples) < true_accuracy).astype(int)

calibrator = ConfidenceCalibrator(method="isotonic")
calibrator.fit(raw_confidences, correctness)

# Before calibration
print(f"ECE (before): {calibrator.expected_calibration_error(raw_confidences, correctness):.3f}")

# After calibration
calibrated_confidences = np.array([
    calibrator.calibrate(c) for c in raw_confidences
])
print(f"ECE (after): {calibrator.expected_calibration_error(calibrated_confidences, correctness):.3f}")
# Typical improvement: ECE drops from 0.15-0.25 to 0.03-0.08
```

**Critical trade-off:** Calibration requires labeled validation data (100-1000 examples minimum). Isotonic regression is more flexible but risks overfitting on small datasets. Platt scaling is more stable but assumes sigmoid relationship.

### 4. Self-Consistency and Ensemble Uncertainty

When logprobs aren't available, self-consistency measures uncertainty by sampling multiple responses and measuring agreement. High variance indicates epistemic uncertainty (model doesn't know the answer).

```python
from collections import Counter
from typing import Any
import hashlib

def compute_self_consistency(
    prompt: str,
    llm_client,
    n