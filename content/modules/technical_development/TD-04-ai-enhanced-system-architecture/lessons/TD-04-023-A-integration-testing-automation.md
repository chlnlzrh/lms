# Integration Testing Automation for LLM-Based Systems

## Core Concepts

Integration testing for LLM-based systems requires fundamentally different approaches than traditional software testing. While conventional integration tests verify deterministic interactions between components with predictable outputs, LLM integration tests must validate probabilistic behaviors, handle non-deterministic responses, and assess semantic correctness rather than exact string matches.

### Traditional vs. LLM Integration Testing

```python
# Traditional Integration Test (Deterministic)
import requests
from typing import Dict, Any

def test_traditional_api_integration():
    """Traditional REST API integration test with exact assertions"""
    response = requests.post(
        "https://api.example.com/calculate",
        json={"operation": "add", "a": 5, "b": 3}
    )
    
    assert response.status_code == 200
    assert response.json() == {"result": 8, "operation": "add"}
    assert response.headers["Content-Type"] == "application/json"
    # Exact matches work because output is deterministic


# LLM Integration Test (Non-Deterministic)
import openai
from dataclasses import dataclass
from typing import Optional
import json

@dataclass
class LLMTestAssertion:
    """Semantic assertion for LLM outputs"""
    contains_concepts: list[str]
    min_length: int
    max_length: int
    expected_format: Optional[str] = None
    prohibited_content: list[str] = None

def test_llm_integration_pipeline():
    """LLM pipeline test requiring semantic validation"""
    client = openai.OpenAI()
    
    # Step 1: Document classification
    classification_response = client.chat.completions.create(
        model="gpt-4",
        messages=[{
            "role": "user",
            "content": "Classify this support ticket: 'My payment failed twice'"
        }],
        temperature=0.3
    )
    
    classification = classification_response.choices[0].message.content
    
    # Can't assert exact string, must validate semantically
    assert_semantic_match(
        output=classification,
        assertion=LLMTestAssertion(
            contains_concepts=["payment", "technical", "high priority"],
            min_length=20,
            max_length=200,
            prohibited_content=["user error", "low priority"]
        )
    )
    
    # Step 2: Response generation using classification
    response_generation = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": f"Category: {classification}"},
            {"role": "user", "content": "Generate support response"}
        ],
        temperature=0.7
    )
    
    response_text = response_generation.choices[0].message.content
    
    # Validate downstream integration worked correctly
    assert_semantic_match(
        output=response_text,
        assertion=LLMTestAssertion(
            contains_concepts=["payment", "resolve", "support"],
            min_length=50,
            max_length=500,
            expected_format="professional_tone"
        )
    )

def assert_semantic_match(output: str, assertion: LLMTestAssertion) -> bool:
    """Semantic assertion helper for LLM outputs"""
    # Length checks
    assert assertion.min_length <= len(output) <= assertion.max_length, \
        f"Length {len(output)} outside range [{assertion.min_length}, {assertion.max_length}]"
    
    # Concept presence (case-insensitive)
    output_lower = output.lower()
    for concept in assertion.contains_concepts:
        assert concept.lower() in output_lower, \
            f"Required concept '{concept}' not found in output"
    
    # Prohibited content
    if assertion.prohibited_content:
        for prohibited in assertion.prohibited_content:
            assert prohibited.lower() not in output_lower, \
                f"Prohibited content '{prohibited}' found in output"
    
    return True
```

### Key Engineering Insights

**1. Determinism vs. Stochasticity Trade-off**: Setting `temperature=0` provides more consistent outputs for testing but may not reflect production behavior. Advanced integration testing requires running test suites at multiple temperature settings to validate both consistency and variability handling.

**2. Assertion Pyramids**: LLM integration tests need layered assertions—syntax validation (JSON structure), semantic validation (concept presence), and behavioral validation (downstream effects). Each layer catches different failure modes.

**3. Test Data Contamination**: Public LLMs may have been trained on common test datasets. Integration tests must use synthetic data or recently created examples that couldn't be in training data, or validation becomes circular.

### Why This Matters Now

LLM-based systems are moving from prototypes to production at scale. The gap between "demo works" and "system is reliable" is filled by comprehensive integration testing. Without proper testing automation:

- **Silent degradation**: Model updates change behavior without detection
- **Cascade failures**: One component's semantic drift breaks downstream systems
- **Cost explosion**: Untested edge cases consume tokens inefficiently
- **Compliance risk**: Unvalidated outputs may violate regulatory requirements

Modern LLM applications integrate multiple models, retrieval systems, databases, and APIs. Integration testing automation is the only scalable way to ensure these complex pipelines maintain correctness, performance, and cost targets across deployments.

## Technical Components

### 1. Semantic Validation Framework

Traditional string comparison fails for LLM outputs. Semantic validation uses embedding similarity, concept extraction, and LLM-as-judge patterns to assess correctness.

```python
import numpy as np
from typing import List, Tuple
from sklearn.metrics.pairwise import cosine_similarity
import openai

class SemanticValidator:
    """Framework for validating LLM outputs semantically"""
    
    def __init__(self, embedding_model: str = "text-embedding-3-small"):
        self.client = openai.OpenAI()
        self.embedding_model = embedding_model
        self.similarity_threshold = 0.85
    
    def validate_similarity(
        self, 
        actual_output: str, 
        expected_output: str,
        threshold: float = None
    ) -> Tuple[bool, float]:
        """Compare semantic similarity using embeddings"""
        threshold = threshold or self.similarity_threshold
        
        # Generate embeddings
        actual_embedding = self._get_embedding(actual_output)
        expected_embedding = self._get_embedding(expected_output)
        
        # Calculate cosine similarity
        similarity = cosine_similarity(
            actual_embedding.reshape(1, -1),
            expected_embedding.reshape(1, -1)
        )[0][0]
        
        passed = similarity >= threshold
        return passed, similarity
    
    def validate_with_judge(
        self,
        actual_output: str,
        validation_criteria: str,
        context: str = ""
    ) -> Tuple[bool, str, dict]:
        """Use LLM-as-judge for complex validation"""
        judge_prompt = f"""You are a test validator. Evaluate if the output meets the criteria.

Context: {context}

Output to validate:
{actual_output}

Validation criteria:
{validation_criteria}

Respond in JSON format:
{{
    "passes": true/false,
    "reasoning": "explanation",
    "severity": "critical|major|minor",
    "specific_failures": ["list of specific issues"]
}}"""
        
        response = self.client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": judge_prompt}],
            temperature=0.1,
            response_format={"type": "json_object"}
        )
        
        result = json.loads(response.choices[0].message.content)
        return result["passes"], result["reasoning"], result
    
    def _get_embedding(self, text: str) -> np.ndarray:
        """Generate embedding vector for text"""
        response = self.client.embeddings.create(
            model=self.embedding_model,
            input=text
        )
        return np.array(response.data[0].embedding)

# Usage in integration tests
def test_customer_response_generation():
    validator = SemanticValidator()
    
    # Generate response
    actual = generate_customer_response("How do I reset my password?")
    
    # Expected semantic content (not exact string)
    expected = """Navigate to settings, click forgot password, 
    check your email for reset link, follow instructions to create new password"""
    
    # Validate semantic similarity
    passes, similarity = validator.validate_similarity(actual, expected)
    assert passes, f"Semantic similarity {similarity:.3f} below threshold"
    
    # Validate with judge for complex criteria
    criteria = """
    - Includes step-by-step instructions
    - Mentions email verification
    - Professional and helpful tone
    - No security anti-patterns (like sharing passwords)
    - References official support channels
    """
    
    passes, reasoning, details = validator.validate_with_judge(
        actual_output=actual,
        validation_criteria=criteria,
        context="Password reset support response"
    )
    
    assert passes, f"Judge validation failed: {reasoning}"
    assert details["severity"] != "critical", "Critical issues found"
```

**Practical Implications**: Semantic validation costs more (API calls for embeddings/judge) but catches actual failures that string matching misses. Budget 2-5x more tokens for validation than generation in test environments.

**Trade-offs**: Embedding similarity is fast but may miss logical errors. LLM-as-judge catches complex issues but adds latency and cost. Use embeddings for regression tests, judge for critical path validation.

### 2. Pipeline State Management

LLM pipelines maintain state across multiple calls. Integration tests must capture, replay, and assert on stateful interactions.

```python
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from datetime import datetime
import hashlib
import json

@dataclass
class PipelineState:
    """Captures complete pipeline execution state"""
    pipeline_id: str
    timestamp: datetime
    steps: List[Dict[str, Any]] = field(default_factory=list)
    context: Dict[str, Any] = field(default_factory=dict)
    
    def add_step(self, step_name: str, input_data: Any, output_data: Any, metadata: Dict = None):
        """Record a pipeline step"""
        self.steps.append({
            "step_name": step_name,
            "input": input_data,
            "output": output_data,
            "metadata": metadata or {},
            "timestamp": datetime.utcnow().isoformat()
        })
    
    def get_state_hash(self) -> str:
        """Generate hash of pipeline state for comparison"""
        state_json = json.dumps(self.steps, sort_keys=True)
        return hashlib.sha256(state_json.encode()).hexdigest()

class StatefulPipelineTest:
    """Test harness for stateful LLM pipelines"""
    
    def __init__(self):
        self.states: List[PipelineState] = []
        self.baseline_states: Dict[str, PipelineState] = {}
    
    def record_pipeline_execution(
        self,
        pipeline_id: str,
        execution_func: callable
    ) -> PipelineState:
        """Execute and record pipeline state"""
        state = PipelineState(
            pipeline_id=pipeline_id,
            timestamp=datetime.utcnow()
        )
        
        # Execute with state tracking
        result = execution_func(state)
        
        self.states.append(state)
        return state
    
    def assert_state_consistency(
        self,
        current_state: PipelineState,
        baseline_state: PipelineState,
        strict: bool = False
    ):
        """Validate pipeline state matches expected behavior"""
        assert len(current_state.steps) == len(baseline_state.steps), \
            f"Step count mismatch: {len(current_state.steps)} vs {len(baseline_state.steps)}"
        
        for i, (current_step, baseline_step) in enumerate(
            zip(current_state.steps, baseline_state.steps)
        ):
            # Step structure should match
            assert current_step["step_name"] == baseline_step["step_name"], \
                f"Step {i} name mismatch"
            
            if strict:
                # Strict mode: outputs must be identical
                assert current_step["output"] == baseline_step["output"], \
                    f"Step {i} output mismatch in strict mode"
            else:
                # Semantic mode: validate output compatibility
                self._validate_output_compatibility(
                    current_step["output"],
                    baseline_step["output"],
                    step_name=current_step["step_name"]
                )
    
    def _validate_output_compatibility(
        self,
        current_output: Any,
        baseline_output: Any,
        step_name: str
    ):
        """Validate outputs are semantically compatible"""
        # Type compatibility
        assert type(current_output) == type(baseline_output), \
            f"{step_name}: Type mismatch"
        
        # For structured data, validate schema
        if isinstance(current_output, dict):
            assert set(current_output.keys()) == set(baseline_output.keys()), \
                f"{step_name}: Schema mismatch"

# Example: Testing a multi-step RAG pipeline
def test_rag_pipeline_state():
    """Integration test for stateful RAG pipeline"""
    tester = StatefulPipelineTest()
    
    def execute_rag_pipeline(state: PipelineState) -> Any:
        client = openai.OpenAI()
        
        # Step 1: Query rewriting
        query = "What's our refund policy?"
        rewrite_response = client.chat.completions.create(
            model="gpt-4",
            messages=[{
                "role": "user",
                "content": f"Rewrite this query for search: {query}"
            }],
            temperature=0.3
        )
        rewritten = rewrite_response.choices[0].message.content
        state.add_step("query_rewrite", query, rewritten)
        
        # Step 2: Document retrieval (mocked for example)
        docs = retrieve_documents(rewritten)  # Your retrieval logic
        state.add_step("retrieval", rewritten, docs, {
            "num_docs": len(docs),
            "retrieval_scores": [d["score"] for d in docs]
        })
        
        # Step 3: Context assembly
        context = "\n\n".join([d["content"] for d in docs])
        state.add_step("context_assembly", docs, context, {
            "context_length": len(context)
        })
        
        # Step 4: Response generation
        final_response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": f"Context:\n{context}"},
                {"role": "user", "content": query}
            ],
            temperature=0.5
        )
        answer = final_response.choices[0].message.content
        state.add_step("generation", {"query": query, "context": context}, answer)
        
        return answer
    
    # Record baseline execution
    baseline_state = tester.record_pipeline_execution(
        "rag_refund_policy",
        execute_rag_pipeline
    )
    
    # Test execution matches baseline behavior
    test_state = tester.record_pipeline_execution(
        "rag_refund_policy",
        execute_rag_pipeline
    )
    
    tester.assert_state_consistency(test_state, baseline_state, strict=False)
    
    # Validate specific step properties
    retrieval_step = test_state.steps[1]
    assert retrieval_step["metadata"]["num_docs"] >= 3, \
        "Should retrieve at least 3 documents"
    assert all(score > 0.7 for score in retrieval_step["metadata"]["retrieval_scores"]), \
        "All retrieved documents should have high relevance"

def retrieve_documents(query: str) -> List[Dict[str, Any]]:
    """Mock document retrieval"""
    return [
        {"content": "Refund policy doc 1", "score": 0.92},
        {"content": "Refund policy doc 2", "score": 0.88},
        {"content": "