# Ethical AI Testing Frameworks

## Core Concepts

Ethical AI testing frameworks are systematic, programmatic approaches to evaluating AI systems for bias, fairness, safety, and alignment with human values. Unlike traditional software testing that validates functional correctness and performance, ethical AI testing validates behavioral characteristics that emerge from statistical learning—properties that can't be proven correct but must be empirically measured across distributions.

### Engineering Analogy: Traditional vs. Ethical Testing

```python
# Traditional Software Testing
def test_calculate_interest():
    """Test deterministic financial calculation"""
    principal = 1000
    rate = 0.05
    expected = 50
    actual = calculate_interest(principal, rate)
    assert actual == expected  # Binary pass/fail

# Ethical AI Testing
def test_loan_approval_fairness():
    """Test statistical fairness across demographic groups"""
    test_cases = load_test_cases(stratified_by=['race', 'gender', 'age'])
    predictions = model.predict(test_cases)
    
    # Can't assert exact equality—must measure distributional properties
    metrics = {
        'demographic_parity': calculate_demographic_parity(predictions, test_cases),
        'equalized_odds': calculate_equalized_odds(predictions, test_cases),
        'disparate_impact': calculate_disparate_impact(predictions, test_cases)
    }
    
    # Multiple competing definitions of "fair"
    # Thresholds are policy decisions, not mathematical truths
    assert metrics['disparate_impact'] > 0.8  # Legal threshold, not technical requirement
```

The fundamental difference: traditional testing verifies deterministic mappings (input X always produces output Y), while ethical AI testing measures emergent statistical properties across data distributions where no single "correct" answer exists.

### Why This Matters Now

Three converging factors make ethical AI testing critical:

1. **Regulatory enforcement**: EU AI Act, algorithmic accountability laws, and anti-discrimination regulations now require documented testing evidence
2. **Model opacity**: As models grow (GPT-4: ~1.8T parameters), interpretability decreases while potential for hidden biases increases
3. **Production consequences**: AI systems now make high-stakes decisions (hiring, lending, healthcare) where biases cause measurable harm

Engineers can no longer treat fairness as a post-deployment concern. Ethical testing must be integrated into CI/CD pipelines with the same rigor as security testing.

### Key Insights That Change Engineering Thinking

**Insight 1: Fairness definitions are mutually exclusive.** You mathematically cannot satisfy demographic parity and equalized odds simultaneously except in trivial cases. Testing requires choosing which fairness criterion matches your deployment context.

**Insight 2: Test data distribution is the hardest problem.** Unlike functional tests where you control inputs, ethical testing requires representative samples of protected groups—often unavailable or illegal to collect.

**Insight 3: Thresholds are policy, not engineering.** When a test shows 15% performance disparity between groups, whether that's "acceptable" is a business/legal decision, not a technical one. Your framework must separate measurement from judgment.

## Technical Components

### 1. Stratified Test Set Construction

Test sets must contain sufficient samples across demographic groups, capability levels, and adversarial scenarios—not just random samples from production data.

```python
from typing import Dict, List, Tuple
import pandas as pd
import numpy as np
from dataclasses import dataclass

@dataclass
class StratificationConfig:
    protected_attributes: List[str]
    min_samples_per_stratum: int
    capability_levels: List[str]
    adversarial_ratio: float = 0.1

class StratifiedTestSetBuilder:
    """Build test sets with guaranteed representation across critical dimensions"""
    
    def __init__(self, config: StratificationConfig):
        self.config = config
        
    def build_test_set(
        self, 
        raw_data: pd.DataFrame,
        target_size: int
    ) -> Tuple[pd.DataFrame, Dict[str, int]]:
        """
        Construct test set with stratified sampling ensuring representation.
        
        Returns:
            test_set: Stratified test data
            coverage_report: Samples per stratum
        """
        # Calculate required samples per stratum
        strata = raw_data.groupby(self.config.protected_attributes)
        n_strata = len(strata)
        
        samples_per_stratum = max(
            self.config.min_samples_per_stratum,
            target_size // n_strata
        )
        
        # Stratified sampling
        test_samples = []
        coverage = {}
        
        for stratum_key, stratum_data in strata:
            # Oversample minority groups if needed
            if len(stratum_data) < samples_per_stratum:
                sampled = stratum_data.sample(
                    n=samples_per_stratum, 
                    replace=True,  # Oversample with replacement
                    random_state=42
                )
            else:
                sampled = stratum_data.sample(
                    n=samples_per_stratum,
                    random_state=42
                )
            
            test_samples.append(sampled)
            coverage[str(stratum_key)] = len(sampled)
        
        test_set = pd.concat(test_samples, ignore_index=True)
        
        # Add adversarial examples
        adversarial_size = int(len(test_set) * self.config.adversarial_ratio)
        adversarial_examples = self._generate_adversarial_examples(
            test_set, 
            adversarial_size
        )
        test_set = pd.concat([test_set, adversarial_examples], ignore_index=True)
        
        return test_set, coverage
    
    def _generate_adversarial_examples(
        self, 
        base_data: pd.DataFrame, 
        n: int
    ) -> pd.DataFrame:
        """Generate edge cases that probe model boundaries"""
        # Example: demographic information with unusual feature combinations
        adversarial = base_data.sample(n=n, random_state=42).copy()
        
        # Inject boundary-probing perturbations
        for attr in self.config.protected_attributes:
            if attr in adversarial.columns:
                # Swap protected attribute while keeping other features constant
                adversarial[attr] = adversarial[attr].sample(frac=1).values
        
        return adversarial

# Usage
config = StratificationConfig(
    protected_attributes=['race', 'gender', 'age_group'],
    min_samples_per_stratum=100,
    capability_levels=['basic', 'intermediate', 'expert']
)

builder = StratifiedTestSetBuilder(config)
test_set, coverage = builder.build_test_set(raw_data, target_size=5000)
print(f"Coverage per stratum: {coverage}")
```

**Practical Implications**: Production datasets are typically imbalanced—minority groups may have 100x fewer samples. Without stratification, your test suite may have zero coverage for important demographic combinations, missing critical biases.

**Real Constraints**: 
- Oversampling introduces statistical bias (artificially inflated minority performance)
- Some demographic information may be legally protected and unavailable
- Intersectional groups (e.g., "Black women over 50") may have too few samples even with oversampling

### 2. Multi-Metric Fairness Evaluation

No single metric captures "fairness." Frameworks must compute multiple metrics and expose the trade-offs.

```python
from typing import Optional
import numpy as np
from sklearn.metrics import confusion_matrix

class FairnessMetrics:
    """Compute multiple fairness metrics with clear trade-off visualization"""
    
    @staticmethod
    def demographic_parity(
        predictions: np.ndarray,
        protected_attribute: np.ndarray
    ) -> Dict[str, float]:
        """
        Measures whether positive prediction rates are equal across groups.
        
        Demographic Parity: P(Ŷ=1|A=0) ≈ P(Ŷ=1|A=1)
        
        Use when: You want equal representation in positive outcomes regardless
        of ground truth (e.g., interview callbacks, ad exposure)
        """
        results = {}
        groups = np.unique(protected_attribute)
        
        for group in groups:
            group_mask = protected_attribute == group
            positive_rate = predictions[group_mask].mean()
            results[f"group_{group}_positive_rate"] = positive_rate
        
        # Calculate maximum disparity
        rates = list(results.values())
        results['max_disparity'] = max(rates) - min(rates)
        results['disparate_impact'] = min(rates) / max(rates) if max(rates) > 0 else 0
        
        return results
    
    @staticmethod
    def equalized_odds(
        y_true: np.ndarray,
        predictions: np.ndarray,
        protected_attribute: np.ndarray
    ) -> Dict[str, float]:
        """
        Measures whether TPR and FPR are equal across groups.
        
        Equalized Odds: 
            P(Ŷ=1|Y=1,A=0) ≈ P(Ŷ=1|Y=1,A=1) (TPR)
            P(Ŷ=1|Y=0,A=0) ≈ P(Ŷ=1|Y=0,A=1) (FPR)
        
        Use when: You want equal error rates across groups for high-stakes
        decisions (lending, criminal justice)
        """
        results = {}
        groups = np.unique(protected_attribute)
        
        tpr_by_group = []
        fpr_by_group = []
        
        for group in groups:
            group_mask = protected_attribute == group
            y_true_group = y_true[group_mask]
            y_pred_group = predictions[group_mask]
            
            tn, fp, fn, tp = confusion_matrix(
                y_true_group, 
                y_pred_group
            ).ravel()
            
            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
            
            tpr_by_group.append(tpr)
            fpr_by_group.append(fpr)
            
            results[f"group_{group}_tpr"] = tpr
            results[f"group_{group}_fpr"] = fpr
        
        results['tpr_disparity'] = max(tpr_by_group) - min(tpr_by_group)
        results['fpr_disparity'] = max(fpr_by_group) - min(fpr_by_group)
        
        return results
    
    @staticmethod
    def calibration_by_group(
        y_true: np.ndarray,
        predicted_probabilities: np.ndarray,
        protected_attribute: np.ndarray,
        n_bins: int = 10
    ) -> Dict[str, np.ndarray]:
        """
        Measures whether predicted probabilities match actual outcomes per group.
        
        Calibration: For predictions p, approximately p% should be positive.
        
        Use when: Probability estimates are used for decision-making
        (risk scores, confidence thresholds)
        """
        results = {}
        groups = np.unique(protected_attribute)
        
        for group in groups:
            group_mask = protected_attribute == group
            y_true_group = y_true[group_mask]
            y_prob_group = predicted_probabilities[group_mask]
            
            # Bin predictions
            bins = np.linspace(0, 1, n_bins + 1)
            bin_indices = np.digitize(y_prob_group, bins) - 1
            
            predicted_probs = []
            actual_rates = []
            
            for bin_idx in range(n_bins):
                bin_mask = bin_indices == bin_idx
                if bin_mask.sum() > 0:
                    predicted_probs.append(y_prob_group[bin_mask].mean())
                    actual_rates.append(y_true_group[bin_mask].mean())
            
            results[f"group_{group}_calibration_error"] = np.mean(
                np.abs(np.array(predicted_probs) - np.array(actual_rates))
            )
        
        return results

# Usage demonstrating metric trade-offs
def evaluate_model_fairness(
    model,
    test_data: pd.DataFrame,
    protected_attr: str = 'race'
) -> Dict[str, Dict]:
    """Comprehensive fairness evaluation exposing trade-offs"""
    X = test_data.drop(['label', protected_attr], axis=1)
    y_true = test_data['label'].values
    protected = test_data[protected_attr].values
    
    predictions = model.predict(X)
    probabilities = model.predict_proba(X)[:, 1]
    
    metrics = FairnessMetrics()
    
    results = {
        'demographic_parity': metrics.demographic_parity(predictions, protected),
        'equalized_odds': metrics.equalized_odds(y_true, predictions, protected),
        'calibration': metrics.calibration_by_group(y_true, probabilities, protected)
    }
    
    # Highlight fundamental trade-off
    dp_violation = results['demographic_parity']['max_disparity'] > 0.1
    eo_violation = results['equalized_odds']['tpr_disparity'] > 0.1
    
    if dp_violation and not eo_violation:
        results['interpretation'] = (
            "Model satisfies equalized odds but violates demographic parity. "
            "This is expected when base rates differ between groups. "
            "Decision: Choose based on whether equal representation or equal "
            "accuracy is your priority."
        )
    
    return results
```

**Practical Implications**: A model can satisfy equalized odds (equal TPR/FPR across groups) while violating demographic parity (unequal positive prediction rates) when groups have different base rates. This isn't a bug—it's a fundamental mathematical constraint.

**Trade-off Example**: In loan approval, if Group A historically has 60% repayment rate and Group B has 40%, a model satisfying demographic parity must approve equal percentages from each group. But this means accepting more high-risk applicants from Group B, increasing losses. Equalized odds would allow different approval rates if the model is equally accurate for both groups.

### 3. Automated Bias Detection Pipelines

Ethical testing must run automatically on every model update, like security scans.

```python
from typing import Callable, List
from dataclasses import dataclass
import json

@dataclass
class FairnessTestCase:
    name: str
    metric_fn: Callable
    threshold: float
    protected_attributes: List[str]
    severity: str  # 'critical', 'warning', 'info'

class BiasDetectionPipeline:
    """Automated bias detection for CI/CD integration"""
    
    def __init__(self, test_cases: List[FairnessTestCase]):
        self.test_cases = test_cases
        self.results = []
    
    def run(
        self,
        model,
        test_data: pd.DataFrame,
        fail_on_critical: bool = True
    ) -> Dict:
        """
        Execute all bias detection tests.
        
        Returns:
            results: Test outcomes with pass/fail status
            
        Raises:
            BiasDetectionError: If critical tests fail and fail_on_critical=True
        """
        all_passed = True
        critical_failures = []
        
        for test_case in self.test_cases:
            result = self._run_single_test(model, test_data, test_case)
            self.results.append(result)
            
            if not result['passed']:
                if test_case.severity == 'critical':
                    critical_failures.append(result)
                    all_passed = False
        
        summary = {
            'all_passed': all_passed,
            'total_tests': len(self.test_cases),
            'passed': sum(1 for r in self.results if r['passed']),
            'failed': sum(1 for r in self.results if not r['passed']),
            'critical_failures': len(critical_failures),
            'details': self.results
        }
        
        if fail_on_critical and critical_