# Deployment Guide

Serve, export, and deploy your fine-tuned models using llm-forge's serving and export tools.

---

## Serving Backends

llm-forge supports three serving backends:

| Backend | Use Case | Features | Install Extra |
|---------|----------|----------|---------------|
| Gradio | Interactive chat UI | Streaming, RAG, parameter controls | `llm-forge[serve]` |
| FastAPI | REST API / production | OpenAI-compatible, SSE streaming, CORS | `llm-forge[serve]` |
| vLLM | High-throughput production | Continuous batching, PagedAttention | Separate install |

---

## Gradio Chat Interface

The Gradio backend provides an interactive web-based chat UI with streaming token generation, adjustable sampling parameters, and optional RAG integration.

### Quick Start

```bash
llm-forge serve --config config.yaml --model-path ./outputs/my-lora/merged
```

Open your browser at `http://localhost:7860`.

### Configuration

```yaml
serving:
  backend: "gradio"
  host: "0.0.0.0"
  port: 7860
  merge_adapter: true          # Merge LoRA before serving
```

### Features

- **Streaming generation** -- Tokens appear in real-time as the model generates them
- **Adjustable parameters** -- Sidebar controls for temperature (0.0-2.0), top-p (0.0-1.0), top-k (0-100), and max tokens (16-4096)
- **System prompt** -- Configurable system prompt per conversation
- **RAG integration** -- Toggle retrieval-augmented generation when a knowledge base is configured
- **Model info panel** -- Displays model architecture, parameter count, device, and dtype
- **Chat history** -- Multi-turn conversation support with message history
- **Stop button** -- Cancel generation mid-stream

### RAG-Enabled Serving

To enable RAG in the chat interface:

```yaml
rag:
  enabled: true
  knowledge_base_path: "./data/knowledge_base/"
  vectorstore: "chromadb"
  embedding_model: "sentence-transformers/all-MiniLM-L6-v2"
  top_k: 5
  chunk_strategy: "recursive"
  chunk_size: 512

serving:
  backend: "gradio"
```

The Gradio UI will display a "Enable RAG" checkbox that users can toggle on/off.

### Python API

```python
from llm_forge.serving.gradio_app import GradioApp

app = GradioApp(
    model_path="./outputs/my-lora/merged",
    config=config,      # Optional LLMForgeConfig instance
)

app.launch(
    host="0.0.0.0",
    port=7860,
    share=False,        # Set True for a public Gradio link
)
```

---

## FastAPI REST Server

The FastAPI backend provides a production-ready REST API with OpenAI-compatible endpoints.

### Quick Start

```bash
llm-forge serve --config config.yaml --model-path ./outputs/my-lora/merged --backend fastapi
```

### Configuration

```yaml
serving:
  backend: "fastapi"
  host: "0.0.0.0"
  port: 8000
```

### Endpoints

#### GET /health

Health check endpoint.

```bash
curl http://localhost:8000/health
```

Response:

```json
{
  "status": "healthy",
  "model_loaded": true
}
```

#### GET /model/info

Return model metadata.

```bash
curl http://localhost:8000/model/info
```

Response:

```json
{
  "model_path": "./outputs/my-lora/merged",
  "model_type": "llama",
  "parameters": "1,235,814,400",
  "dtype": "torch.float16",
  "device": "cuda:0",
  "max_position_embeddings": 2048,
  "vocab_size": 32000
}
```

#### POST /generate

Generate text from a prompt.

```bash
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Explain quantum computing in simple terms.",
    "max_tokens": 256,
    "temperature": 0.7,
    "top_p": 0.9,
    "top_k": 50,
    "stream": false
  }'
```

Response:

```json
{
  "id": "gen-a1b2c3d4e5f6",
  "text": "Quantum computing is a type of computation that...",
  "usage": {
    "prompt_tokens": 12,
    "completion_tokens": 87,
    "total_tokens": 99
  },
  "finish_reason": "stop"
}
```

#### POST /generate (Streaming)

Stream tokens via Server-Sent Events:

```bash
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Hello!", "max_tokens": 100, "stream": true}'
```

Each SSE event:

```
data: {"id": "gen-a1b2c3d4e5f6", "text": "Hello", "finish_reason": null}

data: {"id": "gen-a1b2c3d4e5f6", "text": "!", "finish_reason": null}

data: {"id": "gen-a1b2c3d4e5f6", "text": "", "finish_reason": "stop"}

data: [DONE]
```

#### POST /chat

OpenAI-compatible chat completion endpoint.

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llm-forge",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "What is the capital of France?"}
    ],
    "max_tokens": 256,
    "temperature": 0.7,
    "stream": false
  }'
```

Response:

```json
{
  "id": "chatcmpl-a1b2c3d4e5f6",
  "object": "chat.completion",
  "created": 1709312456,
  "model": "llm-forge",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "The capital of France is Paris."
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 25,
    "completion_tokens": 8,
    "total_tokens": 33
  }
}
```

#### POST /chat (Streaming)

Stream chat completions in OpenAI-compatible SSE format:

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llm-forge",
    "messages": [{"role": "user", "content": "Hello!"}],
    "stream": true
  }'
```

Each SSE event follows the OpenAI streaming format:

```
data: {"id":"chatcmpl-...","object":"chat.completion.chunk","created":...,"model":"llm-forge","choices":[{"index":0,"delta":{"content":"Hello"},"finish_reason":null}]}

data: {"id":"chatcmpl-...","object":"chat.completion.chunk","created":...,"model":"llm-forge","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}

data: [DONE]
```

### Using with OpenAI SDK

The `/chat` endpoint is compatible with the OpenAI Python SDK:

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000",
    api_key="not-needed",
)

response = client.chat.completions.create(
    model="llm-forge",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is machine learning?"},
    ],
    temperature=0.7,
    max_tokens=256,
)

print(response.choices[0].message.content)
```

### Python API

```python
from llm_forge.serving.fastapi_server import FastAPIServer

server = FastAPIServer(
    model_path="./outputs/my-lora/merged",
    config=config,
)

server.start(host="0.0.0.0", port=8000)
```

---

## vLLM Serving

For high-throughput production deployments, llm-forge supports serving via vLLM, which provides continuous batching and PagedAttention for optimal inference performance.

```yaml
serving:
  backend: "vllm"
  host: "0.0.0.0"
  port: 8000
```

### Running with vLLM

First, export and merge your model:

```bash
llm-forge export --config config.yaml --model-path ./outputs/my-lora --format safetensors
```

Then serve with vLLM directly:

```bash
python -m vllm.entrypoints.openai.api_server \
  --model ./outputs/my-lora/merged \
  --host 0.0.0.0 \
  --port 8000
```

---

## Model Export

llm-forge supports exporting trained models to various formats for deployment.

### Export Command

```bash
llm-forge export --config config.yaml --model-path ./outputs/my-lora/merged --format safetensors
```

### Supported Export Formats

| Format | Extension | Use Case | Config Value |
|--------|-----------|----------|-------------|
| SafeTensors | `.safetensors` | HuggingFace ecosystem, fast loading | `safetensors` |
| GGUF | `.gguf` | llama.cpp, Ollama, local inference | `gguf` |
| ONNX | `.onnx` | Cross-platform deployment, edge inference | `onnx` |
| AWQ | -- | Activation-aware weight quantization | `awq` |
| GPTQ | -- | Post-training quantization | `gptq` |

### GGUF Export with Quantization

For local deployment with llama.cpp or Ollama:

```yaml
serving:
  export_format: "gguf"
  gguf_quantization: "Q4_K_M"     # Recommended balance of size/quality
```

Common GGUF quantization levels:

| Level | Description | Size vs FP16 | Quality |
|-------|-------------|-------------|---------|
| `Q2_K` | 2-bit extreme compression | ~15% | Lower |
| `Q4_K_M` | 4-bit medium (recommended) | ~28% | Good |
| `Q5_K_S` | 5-bit small | ~35% | Better |
| `Q6_K` | 6-bit | ~42% | High |
| `Q8_0` | 8-bit | ~55% | Very high |

### Merging LoRA Before Export

For LoRA/QLoRA models, the adapter must be merged into the base model before export:

```yaml
serving:
  merge_adapter: true
  export_format: "safetensors"
```

This creates a `merged/` directory containing the full model weights.

---

## Pushing to HuggingFace Hub

Share your fine-tuned model on HuggingFace:

```bash
llm-forge push --model-path ./outputs/my-lora/merged --repo-id your-username/my-model
```

### Prerequisites

1. Install the HuggingFace Hub CLI:

```bash
pip install huggingface_hub
```

2. Authenticate:

```bash
huggingface-cli login
```

3. Create a repository on HuggingFace (or let the push command create it automatically).

---

## Docker Deployment

llm-forge includes a Docker setup for containerized deployments.

### Docker Compose

```yaml
# docker-compose.yml
services:
  llm-forge-gradio:
    build: .
    ports:
      - "7860:7860"
    volumes:
      - ./models:/app/models
      - ./configs:/app/configs
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    command: >
      llm-forge serve
        --config /app/configs/config.yaml
        --model-path /app/models/my-model
        --backend gradio

  llm-forge-api:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./models:/app/models
      - ./configs:/app/configs
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    command: >
      llm-forge serve
        --config /app/configs/config.yaml
        --model-path /app/models/my-model
        --backend fastapi
```

### Running

```bash
docker compose up llm-forge-gradio    # Start Gradio UI
docker compose up llm-forge-api       # Start REST API
```

---

## Deployment Checklist

Before deploying a model to production:

1. **Merge LoRA adapters** -- Ensure `merge_adapter: true` is set so the model is self-contained
2. **Run evaluation benchmarks** -- Verify the model meets quality standards using `llm-forge eval`
3. **Test the serving endpoint** -- Manually test both streaming and non-streaming responses
4. **Check memory usage** -- Ensure the target deployment GPU has sufficient VRAM
5. **Set appropriate generation defaults** -- Configure sensible temperature, top-p, and max tokens
6. **Enable health checks** -- Use the `/health` endpoint for load balancer health checks
7. **Consider quantization** -- Export to GGUF or use AWQ/GPTQ for reduced memory usage in inference
8. **Security** -- The FastAPI server enables CORS with `allow_origins=["*"]` by default; restrict this in production

---

## Next Steps

- [Training Guide](training_guide.md) -- fine-tune your model
- [Evaluation Guide](evaluation_guide.md) -- benchmark before deploying
- [Configuration Reference](configuration.md) -- serving config fields
- [Distributed Training](distributed_training.md) -- train on multiple GPUs
