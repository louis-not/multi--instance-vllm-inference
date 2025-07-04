# LLM Inference Engine

A high-performance FastAPI server for LLM inference with LoRA adapter support, built on vLLM 0.9.1.

## Features

- **Fast Batch Inference**: Efficient batch processing using vLLM
- **LoRA Adapter Support**: Dynamic LoRA adapter loading and switching
- **Configurable Parameters**: Environment variables and command-line options
- **Production Ready**: Health checks, error handling, and logging

## Quick Start

### Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Start server with default settings
./start_server.sh
```

### Usage Examples

```bash
# Start with custom model
./start_server.sh --model meta-llama/Llama-2-7b-hf --port 8080

# Start with custom LoRA path
./start_server.sh --lora-path /path/to/my/lora/adapters

# Start with environment variables
MODEL_NAME=microsoft/DialoGPT-medium ./start_server.sh
```

## Configuration

### Command Line Options

```bash
./start_server.sh [OPTIONS]

Options:
  -m, --model MODEL           Model name/path
  -h, --host HOST             Host to bind to (default: 0.0.0.0)
  -p, --port PORT             Port to bind to (default: 8000)
  -l, --lora-path PATH        Path to LoRA adapters directory
  -r, --max-lora-rank RANK    Maximum LoRA rank (default: 64)
  -d, --dtype DTYPE           Data type (default: float16)
  -w, --workers WORKERS       Number of workers (default: 1)
  --help                      Show help message
```

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_NAME` | `Qwen/Qwen2.5-0.5B-Instruct-AWQ` | HuggingFace model name or local path |
| `SERVER_HOST` | `0.0.0.0` | Host to bind the server to |
| `SERVER_PORT` | `8000` | Port to bind the server to |
| `LORA_ADAPTERS_PATH` | `./lora_adapters` | Directory containing LoRA adapters |
| `MAX_LORA_RANK` | `64` | Maximum LoRA rank |
| `MODEL_DTYPE` | `float16` | Model data type |
| `UVICORN_WORKERS` | `1` | Number of Uvicorn workers |

## API Endpoints

### POST /inference

Perform batch inference with optional LoRA adapter.

**Request:**
```json
{
  "list_messages": [
    "What is the capital of France?",
    "Explain machine learning in simple terms."
  ],
  "lora_adapter": "my-custom-adapter",
  "max_tokens": 512,
  "temperature": 0.7,
  "top_p": 0.9
}
```

**Response:**
```json
{
  "responses": [
    "The capital of France is Paris.",
    "Machine learning is a method of teaching computers to learn patterns from data..."
  ],
  "lora_adapter_used": "my-custom-adapter"
}
```

### GET /health

Check server health and readiness.

**Response:**
```json
{
  "status": "healthy",
  "engine_ready": true
}
```

## LoRA Adapter Setup

1. **Create LoRA directory:**
```bash
mkdir -p ./lora_adapters
```

2. **Add your LoRA adapters:**
```bash
# Example: Download a LoRA adapter
cd lora_adapters
git clone https://huggingface.co/your-username/your-lora-adapter
```

3. **Use in requests:**
```bash
curl -X POST "http://localhost:8000/inference" \
  -H "Content-Type: application/json" \
  -d '{
    "list_messages": ["Hello, how are you?"],
    "lora_adapter": "your-lora-adapter"
  }'
```

## Examples

### Basic Usage

```bash
# Start server
./start_server.sh

# Send inference request
curl -X POST "http://localhost:8000/inference" \
  -H "Content-Type: application/json" \
  -d '{
    "list_messages": [
      "What is machine learning?",
      "How does deep learning work?"
    ],
    "max_tokens": 100,
    "temperature": 0.7
  }'
```

### With LoRA Adapter

```bash
# Start server with custom LoRA path
./start_server.sh --lora-path /path/to/lora/adapters

# Send request with LoRA adapter
curl -X POST "http://localhost:8000/inference" \
  -H "Content-Type: application/json" \
  -d '{
    "list_messages": ["Generate SQL query for user data"],
    "lora_adapter": "sql-lora-adapter",
    "max_tokens": 200
  }'
```

### Production Deployment

```bash
# Start with multiple workers for production
./start_server.sh \
  --model meta-llama/Llama-2-7b-hf \
  --host 0.0.0.0 \
  --port 8000 \
  --workers 4 \
  --dtype float16
```

## Monitoring

- **Health Check**: `GET /health`
- **API Documentation**: `http://localhost:8000/docs`
- **Server Logs**: Monitor uvicorn logs for request/error tracking

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**: Reduce `max_lora_rank` or use smaller model
2. **LoRA Not Found**: Check `LORA_ADAPTERS_PATH` and adapter directory structure
3. **Model Loading Error**: Verify model name and HuggingFace access

### Performance Tips

- Use `dtype="float16"` for memory efficiency
- Adjust `max_lora_rank` based on your adapters
- Monitor GPU memory usage with `nvidia-smi`
- Use multiple workers for concurrent requests

## Requirements

- Python 3.8+
- CUDA-compatible GPU (recommended)
- vLLM 0.9.1
- FastAPI
- Uvicorn