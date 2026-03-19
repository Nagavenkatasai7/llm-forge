# Deployment Guide

Export, serve, and deploy your fine-tuned models using llm-forge's export pipeline and serving backends.

---

## Table of Contents

1. [Deployment Overview](#deployment-overview)
2. [Model Export Formats](#model-export-formats)
3. [GGUF Export and Quantization](#gguf-export-and-quantization)
4. [Ollama Integration](#ollama-integration)
5. [Gradio Chat Interface](#gradio-chat-interface)
6. [FastAPI REST Server](#fastapi-rest-server)
7. [vLLM for High-Throughput Serving](#vllm-for-high-throughput-serving)
8. [HuggingFace Hub Upload](#huggingface-hub-upload)
9. [Docker Deployment](#docker-deployment)
10. [Production Considerations](#production-considerations)
11. [Deployment Checklist](#deployment-checklist)

---

## Deployment Overview

llm-forge supports a full deployment pipeline from trained model to production serving:

```
Training Output     Export Formats      Serving Options
-----------------   -----------------   -------------------
LoRA adapter   ---> safetensors    ---> FastAPI (REST API)
                    GGUF           ---> Ollama (local chat)
Merged model   ---> ONNX           ---> Gradio (web UI)
                    AWQ            ---> vLLM (high-throughput)
                    GPTQ           ---> HuggingFace Hub
```

---

## Model Export Formats

### Export Command

```bash
llm-forge export --config config.yaml --model-path ./outputs/my-lora/merged --format safetensors
```

### Supported Formats

| Format | Extension | Use Case | Config Value | Required Package |
|--------|-----------|----------|-------------|-----------------|
| SafeTensors | `.safetensors` | HuggingFace ecosystem, fast loading | `safetensors` | `safetensors` |
| GGUF | `.gguf` | llama.cpp, Ollama, local inference | `gguf` | llama.cpp tools |
| ONNX | `.onnx` | Cross-platform, edge inference | `onnx` | `optimum[exporters]` |
| AWQ | -- | Activation-aware weight quantization | `awq` | `autoawq` |
| GPTQ | -- | Post-training quantization | `gptq` | `auto-gptq` |

### Export via YAML Config

```yaml
serving:
  export_format: "gguf"
  gguf_quantization: "Q4_K_M"
  merge_adapter: true
```

### Export via Python API

```python
from llm_forge.serving.export import ModelExporter

# SafeTensors export
ModelExporter.export_safetensors(
    model="./outputs/my-lora/merged",
    output_path="./exports/safetensors/",
)

# GGUF export
ModelExporter.export_gguf(
    model_path="./outputs/my-lora/merged",
    output_path="./exports/model.gguf",
    quantization="q4_k_m",
)

# ONNX export
ModelExporter.export_onnx(
    model="./outputs/my-lora/merged",
    output_path="./exports/onnx/",
)

# AWQ export
ModelExporter.export_awq(
    model="./outputs/my-lora/merged",
    output_path="./exports/awq/",
)
```

### Merge + Export in One Step

For LoRA/QLoRA models, merge the adapter into the base model and export in a single operation:

```python
ModelExporter.merge_lora_and_export(
    base_model="meta-llama/Llama-3.2-1B-Instruct",
    adapter_path="./outputs/my-lora/",
    output_path="./exports/model.gguf",
    format="gguf",
    gguf_quantization="q4_k_m",
)
```

This handles the full pipeline: load base in float16, apply adapter, merge, strip generation markers, convert, and quantize.

---

## GGUF Export and Quantization

GGUF is the recommended format for local inference with llama.cpp and Ollama. It supports a wide range of quantization options that trade model size for quality.

### Prerequisites

Install llama.cpp build tools:

```bash
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
cmake -B build
cmake --build build
```

llm-forge will automatically find `convert_hf_to_gguf.py` and `llama-quantize` in:
- `~/llama.cpp/` (default)
- The `LLAMA_CPP_DIR` environment variable
- `/opt/llama.cpp/`
- System PATH

### Quantization Options

All 16 supported GGUF quantization types:

| Type | Bits | Description | Size vs FP16 | Quality | Recommended For |
|------|------|-------------|-------------|---------|-----------------|
| `f32` | 32 | Full precision | 200% | Maximum | Debugging only |
| `f16` | 16 | Half precision | 100% | Very high | Reference baseline |
| `q8_0` | 8 | 8-bit uniform | ~55% | Very high | When size is not a concern |
| `q6_k` | 6 | 6-bit k-quant | ~42% | High | Quality-first deployment |
| `q5_k_m` | 5 | 5-bit k-quant medium | ~37% | Good-High | Good balance |
| `q5_k_s` | 5 | 5-bit k-quant small | ~35% | Good | Slightly smaller than q5_k_m |
| `q5_0` | 5 | 5-bit uniform | ~35% | Good | Legacy, prefer k-quants |
| `q5_1` | 5 | 5-bit uniform v2 | ~37% | Good | Legacy, prefer k-quants |
| **`q4_k_m`** | **4** | **4-bit k-quant medium** | **~28%** | **Good** | **Recommended default** |
| `q4_k_s` | 4 | 4-bit k-quant small | ~26% | Good | Slightly smaller |
| `q4_0` | 4 | 4-bit uniform | ~25% | Acceptable | Legacy, prefer k-quants |
| `q4_1` | 4 | 4-bit uniform v2 | ~27% | Acceptable | Legacy, prefer k-quants |
| `q3_k_l` | 3 | 3-bit k-quant large | ~22% | Moderate | Memory-constrained |
| `q3_k_m` | 3 | 3-bit k-quant medium | ~20% | Moderate | Memory-constrained |
| `q3_k_s` | 3 | 3-bit k-quant small | ~18% | Lower | Very memory-constrained |
| `q2_k` | 2 | 2-bit k-quant | ~15% | Lower | Extreme compression only |

### Quantization Recommendation

For a 1B parameter model (e.g., Llama 3.2 1B):

| Quantization | GGUF Size | RAM Required | Recommended? |
|-------------|-----------|-------------|-------------|
| Q4_K_M | ~763 MB | ~1.5 GB | Yes -- best balance |
| Q5_K_M | ~900 MB | ~1.8 GB | Yes -- slightly better quality |
| Q8_0 | ~1.3 GB | ~2.5 GB | For quality-critical use |
| Q2_K | ~450 MB | ~1 GB | Only if RAM is very limited |

### Export Command

```bash
llm-forge export --config config.yaml \
  --model-path ./outputs/my-lora/merged \
  --format gguf \
  --quantization Q4_K_M
```

### Chat Template Handling

During GGUF export, llm-forge automatically strips `{% generation %}` markers from the chat template. These markers are used by TRL during training for loss masking but are not valid Jinja2 -- they would cause errors in llama.cpp and Ollama.

The cleanup is applied to both `chat_template.jinja` and `tokenizer_config.json`.

---

## Ollama Integration

Ollama provides local model serving with a simple chat interface. llm-forge generates Ollama-compatible Modelfiles and handles the full deployment workflow.

### End-to-End Workflow

```bash
# 1. Export model to GGUF
llm-forge export --config config.yaml \
  --model-path ./outputs/my-lora/merged \
  --format gguf --quantization Q4_K_M

# 2. Generate Modelfile (done automatically if generate_modelfile: true in config)
# 3. Create Ollama model
cd ./outputs/my-lora/merged
ollama create my-model -f Modelfile

# 4. Run the model
ollama run my-model
```

### Modelfile Generation

llm-forge generates a Modelfile with the correct Llama 3 chat template:

```yaml
serving:
  generate_modelfile: true
  export_format: "gguf"
  gguf_quantization: "Q4_K_M"
  inference_temperature: 0.1
  inference_top_k: 40
  inference_repeat_penalty: 1.1
  inference_num_predict: 256
  inference_num_ctx: 2048
```

### Generated Modelfile Structure

```
FROM ./model-Q4_K_M.gguf

SYSTEM "You are a helpful AI assistant."

TEMPLATE """<|start_header_id|>system<|end_header_id|>

Cutting Knowledge Date: December 2023

{{ if .System }}{{ .System }}
{{- end }}<|eot_id|>
{{- range $i, $_ := .Messages }}
{{- $last := eq (len (slice $.Messages $i)) 1 }}
{{- if eq .Role "user" }}<|start_header_id|>user<|end_header_id|>

{{ .Content }}<|eot_id|>{{ if $last }}<|start_header_id|>assistant<|end_header_id|>

{{ end }}
{{- else if eq .Role "assistant" }}<|start_header_id|>assistant<|end_header_id|>

{{ .Content }}{{ if not $last }}<|eot_id|>{{ end }}
{{- end }}
{{- end }}"""

PARAMETER temperature 0.1
PARAMETER top_p 0.9
PARAMETER top_k 40
PARAMETER repeat_penalty 1.1
PARAMETER repeat_last_n 128
PARAMETER stop "<|eot_id|>"
PARAMETER stop "<|end_of_text|>"
PARAMETER stop "<|start_header_id|>"
PARAMETER num_predict 256
PARAMETER num_ctx 2048
```

### Key Modelfile Design Decisions

These are based on debugging 11 root causes across finance-specialist v1-v6:

**`range .Messages` (not `.Prompt`/`.Response`)**: The legacy `.Prompt`/`.Response` pattern is single-turn only. It flattens the entire conversation history into two strings, destroying turn boundaries. The `range .Messages` pattern iterates over each message individually, preserving multi-turn context.

**No `<|begin_of_text|>` token**: Ollama automatically adds the BOS token. Including it in the template causes a duplicate BOS, which some models handle poorly.

**Three stop tokens**: `<|eot_id|>` (end of turn), `<|end_of_text|>` (end of text), and `<|start_header_id|>` (prevents generation of new turn headers).

**`num_ctx` matches training**: The KV-cache context window must match or exceed the training `max_seq_length`. A mismatch causes truncation of multi-turn conversations.

### Programmatic Modelfile Generation

```python
from llm_forge.serving.export import generate_modelfile

generate_modelfile(
    gguf_path="./exports/model-Q4_K_M.gguf",
    output_dir="./exports/",
    system_prompt="You are a finance specialist AI assistant.",
    temperature=0.1,
    top_p=0.9,
    top_k=40,
    repeat_penalty=1.1,
    num_predict=256,
    num_ctx=2048,
)
```

### Inference Parameter Tuning

| Parameter | Default | Range | Guidance |
|-----------|---------|-------|----------|
| `temperature` | 0.1 | 0.0-2.0 | Lower (0.1-0.3) for factual tasks; higher (0.7-1.0) for creative tasks |
| `top_p` | 0.9 | 0.0-1.0 | 0.9 works well for most cases; lower for more focused output |
| `top_k` | 40 | 1-100 | 40 is a good default; lower for more conservative output |
| `repeat_penalty` | 1.1 | 1.0-1.5 | 1.1 is gentle; 1.3+ causes topic avoidance in small models |
| `num_predict` | 256 | 1-4096 | Match to expected response length; too high wastes compute |
| `num_ctx` | 2048 | 512-8192 | Must match training max_seq_length |

**Important**: For small models (1B), keep `repeat_penalty` at 1.1. Values of 1.3+ were found to cause the model to avoid repeating ANY words, leading to stilted, topic-switching responses (root cause #8 from finance-specialist v5).

---

## Gradio Chat Interface

The Gradio backend provides an interactive web-based chat UI with streaming token generation.

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
  merge_adapter: true
```

### Features

- **Streaming generation**: Tokens appear in real-time
- **Adjustable parameters**: Sidebar controls for temperature (0.0-2.0), top-p (0.0-1.0), top-k (0-100), and max tokens (16-4096)
- **System prompt**: Configurable per conversation
- **RAG integration**: Toggle retrieval-augmented generation when a knowledge base is configured
- **Model info panel**: Displays model architecture, parameter count, device, and dtype
- **Multi-turn chat**: Full conversation history support
- **Stop button**: Cancel generation mid-stream

### Desktop Mode

Wrap the Gradio interface in a native desktop window using pywebview:

```bash
llm-forge ui --desktop
```

Requires `pip install llm-forge[desktop]` (installs pywebview 6.1).

### RAG-Enabled Serving

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

### Python API

```python
from llm_forge.serving.gradio_app import GradioApp

app = GradioApp(
    model_path="./outputs/my-lora/merged",
    config=config,
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

#### `GET /health`

Health check endpoint for load balancer integration.

```bash
curl http://localhost:8000/health
```

Response: `{"status": "healthy", "model_loaded": true}`

#### `GET /model/info`

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

#### `POST /generate`

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

#### `POST /generate` (Streaming)

Stream tokens via Server-Sent Events:

```bash
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Hello!", "max_tokens": 100, "stream": true}'
```

#### `POST /chat`

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

#### `POST /chat` (Streaming)

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

### Using with OpenAI Python SDK

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

## vLLM for High-Throughput Serving

For production deployments requiring high throughput and low latency, llm-forge integrates with vLLM, which provides continuous batching and PagedAttention for optimal GPU utilization.

### Configuration

```yaml
serving:
  backend: "vllm"
  host: "0.0.0.0"
  port: 8000
```

### Running with vLLM

```bash
# First, export and merge your model
llm-forge export --config config.yaml \
  --model-path ./outputs/my-lora \
  --format safetensors

# Option 1: Use llm-forge's vLLM integration
llm-forge serve --config config.yaml \
  --model-path ./outputs/my-lora/merged \
  --backend vllm

# Option 2: Use vLLM directly
python -m vllm.entrypoints.openai.api_server \
  --model ./outputs/my-lora/merged \
  --host 0.0.0.0 \
  --port 8000
```

### vLLM Features

| Feature | Benefit |
|---------|---------|
| Continuous batching | Serves multiple requests simultaneously without waiting |
| PagedAttention | Efficient KV-cache memory management, serves more concurrent requests |
| Speculative decoding | Faster token generation with draft models |
| Tensor parallelism | Distribute large models across multiple GPUs |
| OpenAI-compatible API | Drop-in replacement for OpenAI endpoints |

### vLLM Python API

```python
from llm_forge.serving.vllm_server import VLLMServer

server = VLLMServer(
    model_path="./outputs/my-lora/merged",
    config=config,
)

server.start(host="0.0.0.0", port=8000)
```

### When to Use vLLM vs FastAPI

| Scenario | Recommended Backend |
|----------|-------------------|
| Development / testing | FastAPI |
| Single-user interactive chat | Gradio |
| Multi-user production API | vLLM |
| High-throughput batch inference | vLLM |
| CPU-only deployment | FastAPI |
| Edge / mobile deployment | Export to GGUF + llama.cpp |

---

## HuggingFace Hub Upload

Share your fine-tuned model on HuggingFace Hub with an auto-generated model card.

### CLI

```bash
llm-forge push --model-path ./outputs/my-lora/merged --repo-id your-username/my-model
```

### Prerequisites

```bash
pip install huggingface_hub
huggingface-cli login
```

### Python API

```python
from llm_forge.serving.export import ModelExporter

url = ModelExporter.push_to_hub(
    model_path="./outputs/my-lora/merged",
    repo_id="your-username/my-model",
    private=False,
    commit_message="Upload finance specialist v7 model",
)

print(f"Model uploaded: {url}")
# https://huggingface.co/your-username/my-model
```

### Auto-Generated Model Card

When uploading, llm-forge automatically generates a `README.md` (model card) if one does not exist. It includes:

- Model type, hidden size, and layer count (from `config.json`)
- LoRA training details (from `adapter_config.json`, if present)
- Usage code snippet with `AutoModelForCausalLM` and `AutoTokenizer`
- File listing
- Tags for HuggingFace Hub discovery (`llm-forge`, `text-generation`)

The auto-generated card is cleaned up after upload so it does not pollute the local directory.

---

## Docker Deployment

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
# Start Gradio UI
docker compose up llm-forge-gradio

# Start REST API
docker compose up llm-forge-api

# Start both
docker compose up
```

### CPU-Only Docker

For deployments without GPU access, use the CPU configuration:

```yaml
services:
  llm-forge-cpu:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./models:/app/models
      - ./configs:/app/configs
    environment:
      - CUDA_VISIBLE_DEVICES=""
    command: >
      llm-forge serve
        --config /app/configs/config.yaml
        --model-path /app/models/my-model
        --backend fastapi
```

Note: CPU inference will be significantly slower. Consider using GGUF quantized models with llama.cpp for better CPU performance.

---

## Production Considerations

### Rate Limiting

The FastAPI server does not include built-in rate limiting. For production, add rate limiting via:

- A reverse proxy (nginx, Caddy, Traefik)
- FastAPI middleware (e.g., `slowapi`)
- API gateway (Kong, AWS API Gateway)

### Authentication

The FastAPI server has no built-in authentication. For production:

```python
# Example: Add API key authentication via middleware
from fastapi import Security, HTTPException
from fastapi.security import APIKeyHeader

api_key_header = APIKeyHeader(name="X-API-Key")

async def verify_api_key(api_key: str = Security(api_key_header)):
    if api_key != os.environ["LLM_FORGE_API_KEY"]:
        raise HTTPException(status_code=403, detail="Invalid API key")
```

### CORS

The FastAPI server enables CORS with `allow_origins=["*"]` by default. Restrict this in production:

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://your-domain.com"],
    allow_methods=["POST", "GET"],
    allow_headers=["Content-Type", "Authorization"],
)
```

### HTTPS/TLS

Always use HTTPS in production. Options:

- Terminate TLS at a reverse proxy (nginx with Let's Encrypt)
- Use a managed load balancer (AWS ALB, GCP Cloud Load Balancing)
- Run uvicorn with SSL certificates directly (not recommended for production)

### Health Checks

Use the `/health` endpoint for:

- Kubernetes liveness and readiness probes
- Load balancer health checks
- Monitoring systems (Datadog, Prometheus)

```yaml
# Kubernetes example
livenessProbe:
  httpGet:
    path: /health
    port: 8000
  initialDelaySeconds: 60
  periodSeconds: 30
readinessProbe:
  httpGet:
    path: /health
    port: 8000
  initialDelaySeconds: 30
  periodSeconds: 10
```

### Memory Management

| Model Size | Format | GPU VRAM (Inference) | CPU RAM |
|-----------|--------|---------------------|---------|
| 1B | FP16 | ~2.5 GB | ~4 GB |
| 1B | Q4_K_M GGUF | ~1 GB | ~1.5 GB |
| 3B | FP16 | ~7 GB | ~12 GB |
| 3B | Q4_K_M GGUF | ~2.5 GB | ~3.5 GB |
| 7B | FP16 | ~16 GB | ~28 GB |
| 7B | Q4_K_M GGUF | ~5 GB | ~7 GB |

### Monitoring

Key metrics to monitor in production:

| Metric | Normal Range | Alert Threshold |
|--------|-------------|-----------------|
| Request latency (p50) | 100-500ms | > 2s |
| Request latency (p99) | 500-2000ms | > 10s |
| GPU memory utilization | 60-80% | > 95% |
| GPU compute utilization | 40-90% | Sustained 100% |
| Request error rate | < 0.1% | > 1% |
| Token throughput | Varies by model | < 50% of peak |

---

## Deployment Checklist

Before deploying a model to production:

- [ ] **Merge LoRA adapters**: Ensure `merge_adapter: true` is set so the model is self-contained
- [ ] **Run evaluation benchmarks**: Verify the model meets quality standards using `llm-forge eval`
- [ ] **Check regression report**: Run `BenchmarkRunner.check_regression()` and confirm PASS
- [ ] **Run knowledge retention probes**: Verify retention rate > 90%
- [ ] **Test serving endpoint manually**: Verify both streaming and non-streaming responses
- [ ] **Test multi-turn conversation**: Ensure topic switching and meta-questions work
- [ ] **Check memory usage**: Confirm the target GPU has sufficient VRAM
- [ ] **Set appropriate generation defaults**: Configure temperature, top-p, max tokens for your use case
- [ ] **Quantize for deployment**: Export to GGUF Q4_K_M for local/edge, use FP16 for GPU serving
- [ ] **Enable health checks**: Use `/health` endpoint for load balancer integration
- [ ] **Restrict CORS**: Change from `allow_origins=["*"]` to your specific domains
- [ ] **Add authentication**: Implement API key or OAuth for production endpoints
- [ ] **Set up monitoring**: Track latency, error rate, GPU utilization
- [ ] **Configure rate limiting**: Add rate limits via reverse proxy or middleware
- [ ] **Test under load**: Verify the server handles concurrent requests without OOM

---

## Next Steps

- [Training Guide](training_guide.md) -- fine-tune your model
- [Evaluation Guide](evaluation_guide.md) -- benchmark before deploying
- [Configuration Reference](configuration.md) -- serving config fields
- [Distributed Training](distributed_training.md) -- train on multiple GPUs
