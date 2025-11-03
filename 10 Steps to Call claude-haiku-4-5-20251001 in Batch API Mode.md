# 10 Steps to Call claude-haiku-4-5-20251001 in Batch API Mode

The Message Batches API is a powerful, cost-effective way to asynchronously process large volumes of Messages requests, with all usage charged at 50% of the standard API prices.

------

## **Step-by-Step Process**

### **Step 1: Prepare Your API Key**

Retrieve your Anthropic API key from the Console. This key is required in the header of all API requests to authenticate your account and access Anthropic's services.

```bash
export ANTHROPIC_API_KEY="your-api-key-here"
```

------

### **Step 2: Structure Your Batch Requests**

A Message Batch is composed of a list of requests, where each request has a unique custom_id and a params object with the standard Messages API parameters. Create a JSON structure with all requests:

```json
{
  "requests": [
    {
      "custom_id": "request-1",
      "params": {
        "model": "claude-haiku-4-5-20251001",
        "max_tokens": 1024,
        "messages": [
          {"role": "user", "content": "Summarize artificial intelligence in 50 words"}
        ]
      }
    },
    {
      "custom_id": "request-2",
      "params": {
        "model": "claude-haiku-4-5-20251001",
        "max_tokens": 1024,
        "messages": [
          {"role": "user", "content": "What is the capital of France?"}
        ]
      }
    }
  ]
}
```

------

### **Step 3: Validate Batch Size Constraints**

Verify that the total batch request size doesn't exceed 256 MB. Ensure each request in the batch has a unique custom_id. Ensure it has been less than 29 days since batch created_at time. Batches can contain up to 10,000 messages per batch or 32 MB in size.

------

### **Step 4: Submit the Batch Request**

Send a POST request to the Batch API endpoint with proper headers:

```bash
curl https://api.anthropic.com/v1/messages/batches \
  --header "x-api-key: $ANTHROPIC_API_KEY" \
  --header "anthropic-version: 2023-06-01" \
  --header "content-type: application/json" \
  --data @batch_requests.json
```

**Response (example):**

```json
{
  "id": "msgbatch_01HkcTjaV5uDC8jWR4ZsDV8d",
  "type": "message_batch",
  "processing_status": "in_progress",
  "request_counts": {
    "processing": 2,
    "succeeded": 0,
    "errored": 0,
    "canceled": 0,
    "expired": 0
  },
  "created_at": "2025-11-01T10:30:00Z",
  "expires_at": "2025-11-02T10:30:00Z",
  "archived_at": null,
  "request_counts": {...}
}
```

------

### **Step 5: Store the Batch ID**

Save the returned batch ID for monitoring and result retrieval:

```bash
BATCH_ID="msgbatch_01HkcTjaV5uDC8jWR4ZsDV8d"
```

------

### **Step 6: Poll for Batch Status**

You can monitor the state of your batch by using the retrieval endpoint and polling this endpoint to know when processing has ended. Check the status at intervals:

```bash
curl https://api.anthropic.com/v1/messages/batches/$BATCH_ID \
  --header "x-api-key: $ANTHROPIC_API_KEY" \
  --header "anthropic-version: 2023-06-01"
```

**Check for `processing_status`:**

- `in_progress`: Batch is still being processed
- `ended`: All requests have finished processing

------

### **Step 7: Wait for Processing to Complete**

Processing time: Responses are typically available within 24 hours, although they may be quicker depending on demand. Implement polling logic:

```bash
#!/bin/bash
BATCH_ID="msgbatch_01HkcTjaV5uDC8jWR4ZsDV8d"
MAX_ATTEMPTS=1440  # 24 hours with 1-minute intervals

for i in $(seq 1 $MAX_ATTEMPTS); do
  STATUS=$(curl -s https://api.anthropic.com/v1/messages/batches/$BATCH_ID \
    --header "x-api-key: $ANTHROPIC_API_KEY" \
    --header "anthropic-version: 2023-06-01" | jq -r '.processing_status')
  
  if [ "$STATUS" = "ended" ]; then
    echo "Batch processing completed"
    break
  fi
  
  echo "Attempt $i: Status is $STATUS. Waiting..."
  sleep 60
done
```

------

### **Step 8: Retrieve Results URL**

Results of the batch are available for download at the results_url property on the Message Batch. Once processing ends, extract the results URL:

```bash
RESULTS_URL=$(curl -s https://api.anthropic.com/v1/messages/batches/$BATCH_ID \
  --header "x-api-key: $ANTHROPIC_API_KEY" \
  --header "anthropic-version: 2023-06-01" | jq -r '.results_url')

echo "Results available at: $RESULTS_URL"
```

------

### **Step 9: Download and Stream Results**

Because of the potentially large size of the results, it's recommended to stream results back rather than download them all at once. Fetch results in streaming format:

```bash
curl -s "$RESULTS_URL" \
  --header "anthropic-version: 2023-06-01" \
  --header "x-api-key: $ANTHROPIC_API_KEY" | \
  while IFS= read -r line; do
    echo "$line" | jq '.'
  done > batch_results.jsonl
```

**Sample result entry:**

```json
{
  "custom_id": "request-1",
  "result": {
    "type": "succeeded",
    "message": {
      "id": "msg_abc123",
      "type": "message",
      "role": "assistant",
      "content": [{"type": "text", "text": "Artificial intelligence is technology enabling machines to learn, reason, and perform tasks intelligently without explicit programming, transforming industries and society."}],
      "model": "claude-haiku-4-5-20251001",
      "stop_reason": "end_turn",
      "usage": {"input_tokens": 15, "output_tokens": 28}
    }
  }
}
```

------

### **Step 10: Parse and Process Results**

Parse the JSONL results and extract responses by custom_id:

```bash
cat batch_results.jsonl | jq -r \
  'select(.result.type == "succeeded") | 
   "\(.custom_id): \(.result.message.content[0].text)"'
```

**Output:**

```
request-1: Artificial intelligence is technology enabling machines to learn, reason, and perform tasks intelligently...
request-2: The capital of France is Paris.
```

------

## **Key Benefits of Using Haiku 4.5 in Batch Mode**

- Costs 50% less than standard API calls
- Enhanced throughput with higher rate limits for much larger request volumes
- Ideal for non-time-sensitive bulk operations (data analysis, content generation, testing)
- Each request in the batch is processed independently, so you can mix different types of requests within a single batch

------

## **Important Constraints**

- Each request in the batch must have a unique custom_id
- Maximum 256 MB per batch request
- Processing window: up to 24 hours
- Results expire 29 days after batch creation

------

**Reference Documentation:** https://docs.anthropic.com/en/docs/build-with-claude/batch-processing