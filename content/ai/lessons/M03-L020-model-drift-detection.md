# Model Drift Detection: Engineering Reliable AI Systems in Production

## Core Concepts

Model drift is the degradation of a machine learning model's predictive performance over time due to changes in the statistical properties of input data, target variables, or their relationships. Unlike software bugs that fail immediately, drift causes silent, gradual failures that compound over weeks or months.

### Traditional vs. Modern Approach

**Traditional Software Monitoring:**
```python
# Traditional API monitoring - binary success/failure
def monitor_api_endpoint(response):
    if response.status_code != 200:
        alert("API failure detected")
    if response.latency > 1000:
        alert("Performance degradation")
    # Clear thresholds, immediate detection
```

**ML Model Drift Detection:**
```python
from typing import Dict, List, Tuple
import numpy as np
from scipy import stats
from datetime import datetime, timedelta

class DriftDetector:
    """Detects gradual degradation in model performance"""
    
    def __init__(self, baseline_data: np.ndarray, 
                 baseline_predictions: np.ndarray,
                 sensitivity: float = 0.05):
        self.baseline_data = baseline_data
        self.baseline_predictions = baseline_predictions
        self.baseline_distribution = self._compute_distribution(baseline_data)
        self.sensitivity = sensitivity
        
    def detect_drift(self, current_data: np.ndarray,
                    current_predictions: np.ndarray) -> Dict[str, any]:
        """
        Multi-faceted drift detection - no single threshold
        Returns drift scores across multiple dimensions
        """
        return {
            'data_drift': self._detect_data_drift(current_data),
            'prediction_drift': self._detect_prediction_drift(current_predictions),
            'concept_drift': self._detect_concept_drift(current_data, current_predictions),
            'timestamp': datetime.now()
        }
    
    def _detect_data_drift(self, current_data: np.ndarray) -> Dict[str, float]:
        """Kolmogorov-Smirnov test for distribution shifts"""
        drift_scores = {}
        for feature_idx in range(current_data.shape[1]):
            baseline_feature = self.baseline_data[:, feature_idx]
            current_feature = current_data[:, feature_idx]
            
            # KS test for distribution difference
            statistic, p_value = stats.ks_2samp(baseline_feature, current_feature)
            drift_scores[f'feature_{feature_idx}'] = {
                'ks_statistic': statistic,
                'p_value': p_value,
                'drifted': p_value < self.sensitivity
            }
        return drift_scores
    
    def _detect_prediction_drift(self, current_predictions: np.ndarray) -> Dict[str, any]:
        """Monitor prediction distribution changes"""
        baseline_mean = np.mean(self.baseline_predictions)
        current_mean = np.mean(current_predictions)
        
        # Population Stability Index (PSI)
        psi = self._calculate_psi(self.baseline_predictions, current_predictions)
        
        return {
            'baseline_mean': baseline_mean,
            'current_mean': current_mean,
            'mean_shift': abs(current_mean - baseline_mean),
            'psi': psi,
            'drifted': psi > 0.2  # PSI > 0.2 indicates significant drift
        }
    
    def _calculate_psi(self, baseline: np.ndarray, 
                       current: np.ndarray, bins: int = 10) -> float:
        """Population Stability Index calculation"""
        baseline_hist, bin_edges = np.histogram(baseline, bins=bins)
        current_hist, _ = np.histogram(current, bins=bin_edges)
        
        # Avoid division by zero
        baseline_pct = (baseline_hist + 1e-6) / np.sum(baseline_hist)
        current_pct = (current_hist + 1e-6) / np.sum(current_hist)
        
        psi = np.sum((current_pct - baseline_pct) * np.log(current_pct / baseline_pct))
        return psi
    
    def _compute_distribution(self, data: np.ndarray) -> Dict:
        """Store baseline distribution statistics"""
        return {
            'mean': np.mean(data, axis=0),
            'std': np.std(data, axis=0),
            'quantiles': np.quantile(data, [0.25, 0.5, 0.75], axis=0)
        }
    
    def _detect_concept_drift(self, data: np.ndarray, 
                            predictions: np.ndarray) -> Dict[str, any]:
        """Detect changes in data-prediction relationships"""
        # This requires ground truth labels for full implementation
        # Placeholder for monitoring prediction patterns given inputs
        return {
            'implemented': False,
            'note': 'Requires labeled data stream'
        }
```

The fundamental difference: **traditional monitoring detects immediate failures; drift detection identifies gradual, statistical degradation**.

### Key Engineering Insights

**1. Drift is inevitable, not exceptional.** Unlike bugs you fix once, drift is a continuous process. Production data distributions always diverge from training data over time due to:
- Seasonal patterns (e-commerce traffic during holidays)
- User behavior changes (new features, competitors, world events)
- Data pipeline changes (new data sources, preprocessing updates)
- Adversarial adaptation (fraud patterns evolving to evade detection)

**2. Multiple drift types require different strategies:**
- **Data drift (covariate shift):** Input distribution changes, but P(Y|X) stays constant
- **Concept drift:** Relationship between inputs and outputs changes
- **Label drift (prior probability shift):** Target distribution changes
- **Prediction drift:** Model outputs change without labeled data to confirm accuracy

**3. Detection without ground truth is possible but limited.** Most production scenarios lack immediate labels. You can detect data and prediction drift statistically, but confirming actual performance degradation requires eventual ground truth.

### Why This Matters NOW

Modern AI systems deployed at scale face drift continuously:

- **LLM applications:** User query patterns shift as product evolves
- **Recommendation systems:** User preferences change with trends
- **Fraud detection:** Attackers adapt to existing models
- **Computer vision:** Camera hardware updates, lighting conditions change

Without drift detection, models silently degrade. A model that was 95% accurate at deployment might drop to 75% after six months—users experience worse results, but you don't know until manual review catches it.

The cost is measurable: a recommendation system with 10% drift might see 15-20% revenue decrease; a fraud detector with drift might miss 30% more fraudulent transactions.

## Technical Components

### 1. Statistical Distance Metrics

**Technical Explanation:**

Statistical distance metrics quantify how different two probability distributions are. For drift detection, you compare baseline (training or early production) distributions against current production distributions.

**Critical Metrics:**

```python
from typing import Tuple
import numpy as np
from scipy.spatial.distance import jensenshannon
from scipy.stats import ks_2samp, wasserstein_distance

class DistanceMetrics:
    """Core statistical distance calculations for drift detection"""
    
    @staticmethod
    def kolmogorov_smirnov(baseline: np.ndarray, 
                          current: np.ndarray) -> Tuple[float, float]:
        """
        KS test: Maximum distance between cumulative distributions
        Good for: Continuous features, univariate analysis
        Limitation: Univariate only, sensitive to sample size
        """
        statistic, p_value = ks_2samp(baseline, current)
        return statistic, p_value
    
    @staticmethod
    def jensen_shannon_divergence(baseline: np.ndarray, 
                                  current: np.ndarray,
                                  bins: int = 30) -> float:
        """
        JSD: Symmetric measure of distribution similarity (0-1 scale)
        Good for: All feature types, interpretable
        Limitation: Requires binning for continuous features
        """
        # Create histograms with same bins
        min_val, max_val = min(baseline.min(), current.min()), max(baseline.max(), current.max())
        baseline_hist, _ = np.histogram(baseline, bins=bins, range=(min_val, max_val))
        current_hist, _ = np.histogram(current, bins=bins, range=(min_val, max_val))
        
        # Normalize to probabilities
        baseline_prob = baseline_hist / baseline_hist.sum()
        current_prob = current_hist / current_hist.sum()
        
        # Calculate JS divergence
        jsd = jensenshannon(baseline_prob, current_prob)
        return jsd ** 2  # Square to get actual JSD (not distance)
    
    @staticmethod
    def wasserstein(baseline: np.ndarray, current: np.ndarray) -> float:
        """
        Wasserstein distance: "Earth mover's distance"
        Good for: Capturing magnitude of distribution shift
        Limitation: Computationally expensive for large datasets
        """
        return wasserstein_distance(baseline, current)
    
    @staticmethod
    def population_stability_index(baseline: np.ndarray,
                                  current: np.ndarray,
                                  bins: int = 10) -> float:
        """
        PSI: Industry standard for score monitoring
        Thresholds: <0.1 (no drift), 0.1-0.2 (moderate), >0.2 (significant)
        Good for: Model score distributions
        """
        # Equal-frequency binning on baseline
        percentiles = np.linspace(0, 100, bins + 1)
        bin_edges = np.percentile(baseline, percentiles)
        bin_edges[-1] = np.inf  # Handle edge case
        
        baseline_hist, _ = np.histogram(baseline, bins=bin_edges)
        current_hist, _ = np.histogram(current, bins=bin_edges)
        
        # Convert to percentages
        baseline_pct = (baseline_hist + 1e-6) / baseline_hist.sum()
        current_pct = (current_hist + 1e-6) / current_hist.sum()
        
        # PSI formula
        psi = np.sum((current_pct - baseline_pct) * np.log(current_pct / baseline_pct))
        return psi

# Example usage
np.random.seed(42)
baseline_data = np.random.normal(0, 1, 10000)
drifted_data = np.random.normal(0.5, 1.2, 10000)  # Shifted mean and variance

metrics = DistanceMetrics()

print("No Drift vs Drifted Comparison:")
print(f"KS Statistic: {metrics.kolmogorov_smirnov(baseline_data, drifted_data)[0]:.4f}")
print(f"JS Divergence: {metrics.jensen_shannon_divergence(baseline_data, drifted_data):.4f}")
print(f"Wasserstein: {metrics.wasserstein(baseline_data, drifted_data):.4f}")
print(f"PSI: {metrics.population_stability_index(baseline_data, drifted_data):.4f}")
```

**Practical Implications:**

- **Choose metrics based on feature type:** KS for continuous, JSD for categorical
- **PSI is interpretable for stakeholders:** "PSI of 0.25 means significant drift"
- **Computational cost matters at scale:** KS is O(n log n), Wasserstein is O(n³) without optimization

**Real Constraint:** Different metrics can disagree. A feature might show high KS but low PSI. Use multiple metrics and tune thresholds based on validation data where you have ground truth.

### 2. Windowing Strategies

**Technical Explanation:**

Windowing determines how you aggregate data for drift comparison. Poor windowing causes false positives (detecting noise as drift) or false negatives (missing real drift).

```python
from collections import deque
from typing import Optional, List
import numpy as np
from datetime import datetime, timedelta

class WindowManager:
    """Manages data windows for drift detection"""
    
    def __init__(self, window_type: str = 'sliding',
                 window_size: int = 1000,
                 comparison_type: str = 'static'):
        """
        window_type: 'sliding', 'tumbling', or 'exponential'
        comparison_type: 'static' (vs baseline) or 'adaptive' (vs recent window)
        """
        self.window_type = window_type
        self.window_size = window_size
        self.comparison_type = comparison_type
        self.baseline_window: Optional[np.ndarray] = None
        self.current_window = deque(maxlen=window_size)
        self.historical_windows: List[np.ndarray] = []
        
    def update(self, data_point: np.ndarray) -> bool:
        """
        Add data point and return True if window is full
        """
        self.current_window.append(data_point)
        return len(self.current_window) == self.window_size
    
    def get_comparison_windows(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns (reference_window, current_window) for comparison
        """
        current = np.array(list(self.current_window))
        
        if self.comparison_type == 'static':
            # Always compare to initial baseline
            if self.baseline_window is None:
                self.baseline_window = current.copy()
            return self.baseline_window, current
        
        elif self.comparison_type == 'adaptive':
            # Compare to previous window(s)
            if len(self.historical_windows) == 0:
                self.historical_windows.append(current.copy())
                return current, current
            
            reference = self.historical_windows[-1]
            self.historical_windows.append(current.copy())
            
            # Keep only last N windows for memory management
            if len(self.historical_windows) > 10:
                self.historical_windows.pop(0)
            
            return reference, current
    
    def reset_window(self):
        """Reset for tumbling window strategy"""
        if self.window_type == 'tumbling':
            self.current_window.clear()

class ExponentialWeightedWindow:
    """Time-decay weighted drift detection"""
    
    def __init__(self, alpha: float = 0.1, feature_dim: int = 1):
        """
        alpha: decay factor (higher = more weight on recent data)
        """
        self.alpha = alpha
        self.mean = np.zeros(feature_dim)
        self.variance = np.ones(feature_dim)
        self.count = 0
        
    def update(self, data_point: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Update running statistics with exponential decay
        Returns current mean and variance
        """
        if self.count == 0:
            self.mean = data_point
            self.count = 1
            return {'mean': self.mean, 'variance': self.variance}
        
        # Exponential moving average
        delta = data_point - self.mean
        self.mean += self.alpha * delta
        self.variance = (1 - self.alpha) * (self.variance + self.alpha * delta**2)
        self.count += 1
        
        return {'mean': self.mean, 'variance': self.variance}
    
    def detect_drift(self, current_point: np.ndarray, 
                    threshold_std: float = 3.0) -> Dict[str, any]:
        """
        Detect if current point deviates from expected distribution
        """
        stats = self.update(current_point)
        z_score = np.abs((current_point - stats['mean']) / np.sqrt(stats['variance']))
        
        return {
            'z_score': z_score,
            'drifted': np.any(z_score > threshold_std),
            'drifted_features': np.where(z_score > threshold_std)[0]
        }

# Comparative example
np.random.seed(42)

# Simulate data stream with drift at point 5000
normal_data = np.random.normal(0, 1, 5000)
drifted_data =