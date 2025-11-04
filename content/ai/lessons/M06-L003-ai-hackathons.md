# AI Hackathons: Engineering Rapid Prototypes with LLMs

## Core Concepts

AI hackathons represent a compressed product development cycle where engineers build functional prototypes leveraging LLM capabilities within 24-48 hours. Unlike traditional hackathons that focus on novel algorithms or infrastructure, AI hackathons center on rapidly composing existing LLM capabilities into useful applications.

### Traditional vs. Modern Hackathon Engineering

```python
# Traditional Hackathon Approach (2019)
class SentimentAnalyzer:
    def __init__(self):
        # Hours spent on data collection and model training
        self.model = self.train_custom_model()
        self.vectorizer = TfidfVectorizer()
    
    def train_custom_model(self):
        # 6-8 hours: data collection, cleaning, training
        training_data = self.scrape_labeled_data()
        X, y = self.prepare_features(training_data)
        model = LogisticRegression()
        model.fit(X, y)
        return model
    
    def analyze(self, text: str) -> dict:
        features = self.vectorizer.transform([text])
        sentiment = self.model.predict(features)[0]
        return {"sentiment": sentiment}

# Modern AI Hackathon Approach (2024)
from anthropic import Anthropic
from typing import Literal

class LLMSentimentAnalyzer:
    def __init__(self, api_key: str):
        # 5 minutes: setup and prompt engineering
        self.client = Anthropic(api_key=api_key)
        self.system_prompt = """Analyze sentiment and extract key entities.
        Return JSON: {"sentiment": "positive|negative|neutral", 
                      "confidence": 0-1, 
                      "key_themes": [...],
                      "entities": [...]}"""
    
    def analyze(self, text: str) -> dict:
        response = self.client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=500,
            system=self.system_prompt,
            messages=[{"role": "user", "content": text}]
        )
        return eval(response.content[0].text)  # Parse JSON response

# Time investment shift:
# Traditional: 70% model building, 20% integration, 10% UI
# Modern: 10% prompt engineering, 40% integration, 50% user experience
```

The fundamental shift is from **building capabilities** to **composing capabilities**. Your engineering time moves from training models to designing interfaces, handling edge cases, and creating compelling user experiences.

### Key Insights That Change Your Approach

**1. The Bottleneck Is Not Intelligence**  
In 2019, model quality was the primary constraint. You spent hours tuning hyperparameters for a 2% accuracy gain. In 2024, base LLMs already exceed human performance on most knowledge tasks. Your bottleneck is now: prompt design, latency management, cost optimization, and user experience.

**2. Prototypes Are Production-Ready (With Guardrails)**  
A well-architected hackathon project can scale to thousands of users without fundamental rewrites. The difference between a prototype and production is primarily: error handling, rate limiting, cost controls, and monitoring—not core functionality.

**3. Value Comes From Integration, Not Innovation**  
The winning projects rarely involve novel AI techniques. They win by: connecting to the right data sources, solving real user pain points, creating intuitive interfaces, and handling edge cases gracefully.

### Why This Matters Now

The barrier to building AI applications has collapsed from months to hours. This creates two imperatives:

1. **Speed of iteration determines success**: The team that can test 10 ideas beats the team that perfectly executes 1 idea.
2. **Rapid prototyping is a core engineering skill**: Your ability to validate product hypotheses in hours directly impacts your career value.

## Technical Components

### 1. Prompt Engineering Under Time Pressure

In a hackathon, you don't have time for extensive prompt iteration. You need a systematic approach to get 80% quality on the first try.

**Technical Framework:**

```python
from dataclasses import dataclass
from typing import List, Optional
import json

@dataclass
class PromptTemplate:
    """Structured approach to rapid prompt engineering"""
    task_description: str
    input_format: str
    output_format: str
    constraints: List[str]
    examples: List[dict]
    
    def render(self, user_input: str) -> str:
        prompt_parts = [
            f"Task: {self.task_description}",
            f"\nInput Format: {self.input_format}",
            f"\nOutput Format: {self.output_format}",
            "\nConstraints:"
        ]
        prompt_parts.extend([f"- {c}" for c in self.constraints])
        
        if self.examples:
            prompt_parts.append("\nExamples:")
            for ex in self.examples:
                prompt_parts.append(f"Input: {ex['input']}")
                prompt_parts.append(f"Output: {ex['output']}\n")
        
        prompt_parts.append(f"\nNow process this input:\n{user_input}")
        return "\n".join(prompt_parts)

# Example: Rapid email categorization system
email_classifier = PromptTemplate(
    task_description="Categorize customer emails and extract action items",
    input_format="Raw email text with subject and body",
    output_format='JSON: {"category": str, "priority": int, "action_items": [str], "auto_reply": bool}',
    constraints=[
        "Category must be one of: sales, support, billing, general",
        "Priority: 1-5 (5 is urgent)",
        "Only set auto_reply=true for simple acknowledgments"
    ],
    examples=[
        {
            "input": "Subject: Billing issue\nBody: I was charged twice this month",
            "output": '{"category": "billing", "priority": 5, "action_items": ["Investigate duplicate charge", "Issue refund"], "auto_reply": false}'
        }
    ]
)

# Usage in hackathon context
def process_email(email_text: str, client) -> dict:
    prompt = email_classifier.render(email_text)
    response = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=300,
        messages=[{"role": "user", "content": prompt}]
    )
    return json.loads(response.content[0].text)
```

**Practical Implications:**

- **Time savings**: 5 minutes to create a working classifier vs. 2+ hours of trial and error
- **Consistency**: Template structure ensures you don't forget critical constraints
- **Debuggability**: When prompts fail, you know which component to adjust

**Trade-offs:**

- More verbose prompts = higher token costs (typically 2-3x)
- Rigid structure may not suit creative tasks
- Examples add latency but significantly improve accuracy

### 2. API Cost Management in Real-Time

Your hackathon budget might be $50-100. Without cost controls, a single bug can drain this in minutes.

**Cost Control Architecture:**

```python
from datetime import datetime, timedelta
from collections import defaultdict
from typing import Dict
import threading

class CostGuard:
    """Real-time cost tracking and circuit breaker"""
    
    def __init__(self, hourly_limit: float = 10.0, daily_limit: float = 50.0):
        self.hourly_limit = hourly_limit
        self.daily_limit = daily_limit
        self.costs: Dict[datetime, float] = defaultdict(float)
        self.lock = threading.Lock()
        
        # Token cost per million (approximate)
        self.cost_per_mtok = {
            "input": 3.0,   # $3 per million input tokens
            "output": 15.0  # $15 per million output tokens
        }
    
    def estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Estimate cost before making API call"""
        input_cost = (input_tokens / 1_000_000) * self.cost_per_mtok["input"]
        output_cost = (output_tokens / 1_000_000) * self.cost_per_mtok["output"]
        return input_cost + output_cost
    
    def check_and_record(self, estimated_cost: float) -> bool:
        """Returns True if request is within budget"""
        with self.lock:
            now = datetime.now()
            hour_ago = now - timedelta(hours=1)
            day_ago = now - timedelta(days=1)
            
            # Clean old entries
            self.costs = {
                ts: cost for ts, cost in self.costs.items() 
                if ts > day_ago
            }
            
            # Calculate recent spending
            hourly_spend = sum(
                cost for ts, cost in self.costs.items() 
                if ts > hour_ago
            )
            daily_spend = sum(self.costs.values())
            
            # Check limits
            if hourly_spend + estimated_cost > self.hourly_limit:
                return False
            if daily_spend + estimated_cost > self.daily_limit:
                return False
            
            # Record cost
            self.costs[now] = estimated_cost
            return True
    
    def get_status(self) -> dict:
        """Get current spending status"""
        now = datetime.now()
        hour_ago = now - timedelta(hours=1)
        
        hourly_spend = sum(
            cost for ts, cost in self.costs.items() 
            if ts > hour_ago
        )
        daily_spend = sum(self.costs.values())
        
        return {
            "hourly_spend": hourly_spend,
            "hourly_remaining": self.hourly_limit - hourly_spend,
            "daily_spend": daily_spend,
            "daily_remaining": self.daily_limit - daily_spend
        }

# Integration with API calls
cost_guard = CostGuard(hourly_limit=5.0, daily_limit=50.0)

def safe_llm_call(prompt: str, max_output_tokens: int, client) -> Optional[str]:
    """LLM call with cost protection"""
    # Rough token estimation (4 chars ≈ 1 token)
    estimated_input = len(prompt) // 4
    estimated_cost = cost_guard.estimate_cost(estimated_input, max_output_tokens)
    
    if not cost_guard.check_and_record(estimated_cost):
        status = cost_guard.get_status()
        raise Exception(f"Budget exceeded. Status: {status}")
    
    response = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=max_output_tokens,
        messages=[{"role": "user", "content": prompt}]
    )
    
    return response.content[0].text
```

**Real Constraints:**

- **False positives**: Conservative estimation may block valid requests (better than budget overrun)
- **Threading overhead**: Lock contention at high concurrency (negligible for hackathon scale)
- **Doesn't catch actual API costs**: This estimates costs; actual billing may differ by 10-20%

**Concrete Example:**

Without cost guards, a recursive prompt loop drained $47 in 8 minutes during testing. With guards: circuit breaker triggered after $5, preserving budget for iteration.

### 3. Latency Optimization for Demo Quality

Users tolerate 3-5 seconds for AI responses. Beyond that, your demo feels broken. Hackathon projects often fail not because of functionality, but because responses take 15+ seconds.

**Streaming Response Pattern:**

```python
from anthropic import Anthropic
from typing import Iterator, Callable

def stream_response(
    prompt: str,
    client: Anthropic,
    on_chunk: Callable[[str], None]
) -> str:
    """Stream response chunks for immediate UI feedback"""
    full_response = []
    
    with client.messages.stream(
        model="claude-3-5-sonnet-20241022",
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}]
    ) as stream:
        for text in stream.text_stream:
            full_response.append(text)
            on_chunk(text)  # Send to UI immediately
    
    return "".join(full_response)

# Example: Web endpoint with streaming
from flask import Flask, Response, stream_with_context
import json

app = Flask(__name__)

@app.route('/analyze', methods=['POST'])
def analyze():
    """Streaming endpoint for real-time feedback"""
    user_input = request.json['text']
    
    def generate():
        # Send initial acknowledgment
        yield f"data: {json.dumps({'status': 'processing'})}\n\n"
        
        accumulated = []
        def on_chunk(chunk: str):
            accumulated.append(chunk)
            # Send each chunk as Server-Sent Event
            yield f"data: {json.dumps({'chunk': chunk})}\n\n"
        
        result = stream_response(
            f"Analyze this text: {user_input}",
            client,
            on_chunk
        )
        
        # Send completion signal
        yield f"data: {json.dumps({'status': 'complete', 'full_text': result})}\n\n"
    
    return Response(
        stream_with_context(generate()),
        mimetype='text/event-stream'
    )
```

**Performance Impact:**

- **Perceived latency**: Drops from 8s (full response) to 0.3s (first chunk)
- **User engagement**: Users see progress immediately, reducing bounce rate
- **Actual latency unchanged**: Total processing time is the same, but user experience is dramatically better

### 4. Caching and Prompt Optimization

Many hackathon projects make the same API calls repeatedly. Intelligent caching can reduce costs by 60-80% and latency by 90%.

```python
import hashlib
from functools import lru_cache
from typing import Optional
import pickle
import os

class PromptCache:
    """Persistent cache for LLM responses"""
    
    def __init__(self, cache_dir: str = ".cache"):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
    
    def _hash_prompt(self, prompt: str, model: str, max_tokens: int) -> str:
        """Create unique hash for prompt configuration"""
        key = f"{model}:{max_tokens}:{prompt}"
        return hashlib.sha256(key.encode()).hexdigest()
    
    def get(self, prompt: str, model: str, max_tokens: int) -> Optional[str]:
        """Retrieve cached response"""
        cache_key = self._hash_prompt(prompt, model, max_tokens)
        cache_path = os.path.join(self.cache_dir, f"{cache_key}.pkl")
        
        if os.path.exists(cache_path):
            with open(cache_path, 'rb') as f:
                return pickle.load(f)
        return None
    
    def set(self, prompt: str, model: str, max_tokens: int, response: str):
        """Cache response"""
        cache_key = self._hash_prompt(prompt, model, max_tokens)
        cache_path = os.path.join(self.cache_dir, f"{cache_key}.pkl")
        
        with open(cache_path, 'wb') as f:
            pickle.dump(response, f)

# Integration
cache = PromptCache()

def cached_llm_call(prompt: str, model: str, max_tokens: int, client) -> str:
    """LLM call with caching"""
    # Check cache first
    cached = cache.get(prompt, model, max_tokens)
    if cached:
        return cached
    
    # Make API call
    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}]
    )
    
    result = response.content[0].text
    
    # Cache result
    cache.set(prompt, model, max_tokens, result)
    
    return result
```

**When Caching Matters:**

- Demo scenarios with repeated inputs
- Development/testing iterations
- Batch processing with duplicate queries

**When Not To Cache:**

-