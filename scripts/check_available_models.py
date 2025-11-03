"""
Check what OpenAI models are available
"""
import os
from openai import OpenAI

with open("openaiapikey.txt", "r", encoding="utf-8") as f:
    api_key = f.read().strip()

client = OpenAI(api_key=api_key)

print("="*80)
print("CHECKING AVAILABLE MODELS")
print("="*80)

# Test a few common model names
models_to_test = [
    "gpt-5-instant",
    "gpt-4o-mini",
    "gpt-4o",
    "gpt-4-turbo",
    "gpt-4",
    "gpt-3.5-turbo"
]

print("\nTesting model availability...\n")

for model_name in models_to_test:
    try:
        print(f"Testing {model_name}...", end=" ")
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": "test"}],
            max_tokens=5
        )
        print(f"✓ AVAILABLE")
    except Exception as e:
        error_msg = str(e)
        if "does not exist" in error_msg or "model_not_found" in error_msg:
            print(f"✗ NOT FOUND")
        elif "permission" in error_msg.lower() or "access" in error_msg.lower():
            print(f"✗ NO ACCESS")
        else:
            print(f"✗ ERROR: {type(e).__name__}")

print("\n" + "="*80)
print("RECOMMENDATION:")
print("="*80)
print("If gpt-5-instant is not available, use gpt-4o-mini as fallback")
print("(Fast, cost-effective, 128K context, 16K output tokens)")
print("="*80)

