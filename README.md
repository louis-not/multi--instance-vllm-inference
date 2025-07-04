# LLM Inference Engine

A high-performance FastAPI server for LLM inference with multi-instance support and load balancing, built on vLLM 0.9.1.

## Features

- **Unified Deployment**: Single script for both single and multi-instance modes
- **Multi-Instance Deployment**: Scale horizontally with multiple model instances
- **Intelligent Load Balancing**: Automatic message distribution across instances
- **Fast Batch Inference**: Efficient batch processing using vLLM
- **Automatic LoRA Fallback**: Dynamic LoRA adapter loading with automatic compatibility fallback
- **Configurable Parameters**: Environment variables and command-line options
- **Production Ready**: Health checks, error handling, monitoring, and logging

## Deployment Options

### Unified Deployment Script (Recommended)

The `deploy.sh` script provides a unified interface for both single and multi-instance deployments:

```bash
# Install dependencies
pip install -r requirements.txt

# Single instance (default)
./deploy.sh
./deploy.sh --mode single --port 8000

# Multi-instance with load balancer
./deploy.sh --mode multi --num-instances 4

# Management commands
./deploy.sh --status
./deploy.sh --stop
```


## Quick Start Examples

### Usage Examples
```bash
# Basic single instance start (default)
./deploy.sh

# Single instance with custom configuration
./deploy.sh --mode single --model meta-llama/Llama-2-7b-hf --port 8080

# Multi-instance deployment
./deploy.sh --mode multi --num-instances 3 --lb-port 8080

# Custom multi-instance configuration
./deploy.sh --mode multi -n 4 -p 9000 -b 9080 --model meta-llama/Llama-2-7b-hf

# Environment variable configuration
MODEL_NAME=microsoft/DialoGPT-medium ./deploy.sh --mode single
NUM_INSTANCES=2 MODEL_NAME=Qwen/Qwen2.5-0.5B ./deploy.sh --mode multi
```


## Configuration

### Script Options

```bash
./deploy.sh [OPTIONS]

Deployment Modes:
  --mode single               Single instance deployment (default)
  --mode multi                Multi-instance deployment with load balancer
  --stop                      Stop all running instances
  --status                    Show status of running instances

Common Options:
  -m, --model MODEL           Model name/path (default: Qwen/Qwen2.5-0.5B-Instruct-AWQ)
  -h, --host HOST             Host to bind to (default: 0.0.0.0)
  -d, --dtype DTYPE           Data type (default: float16)
  -l, --lora-path PATH        Path to LoRA adapters directory (default: ./lora_adapters)
  -r, --max-lora-rank RANK    Maximum LoRA rank (default: 64)
  --help                      Show help message

Single Instance Options:
  -p, --port PORT             Port to bind to (default: 8000)
  -w, --workers WORKERS       Number of workers (default: 1)

Multi-Instance Options:
  -p, --base-port PORT        Base port for instances (default: 8000)
  -b, --lb-port PORT          Load balancer port (default: 8080)
  -n, --num-instances NUM     Number of model instances (default: 2)
```


### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_NAME` | `Qwen/Qwen2.5-0.5B-Instruct-AWQ` | HuggingFace model name or local path |
| `SERVER_HOST` | `0.0.0.0` | Host to bind the server to |
| `SERVER_PORT` | `8000` | Port to bind the server to (single) / Base port (multi) |
| `LOAD_BALANCER_PORT` | `8080` | Load balancer port (multi-instance only) |
| `NUM_INSTANCES` | `2` | Number of model instances (multi-instance only) |
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

### Multi-Instance Load Balanced Inference

```bash
# Start 4 instances with load balancer
./start_multi_instance.sh --num-instances 4 --lb-port 8080

# Send requests to load balancer (automatically distributes across instances)
curl -X POST "http://localhost:8080/inference" \
  -H "Content-Type: application/json" \
  -d '{
    "list_messages": [
      "What is machine learning?",
      "Explain neural networks",
      "How does deep learning work?",
      "What are transformers?"
    ],
    "max_tokens": 100
  }'

# Check load balancer status
curl "http://localhost:8080/health"

# Get detailed instance statistics
curl "http://localhost:8080/stats"
```

### With LoRA Adapter (Automatic Fallback)

```bash
# LoRA support with automatic fallback for older GPUs
./deploy.sh --mode single --lora-path /path/to/lora/adapters

# LoRA request (automatically falls back if unsupported)
curl -X POST "http://localhost:8000/inference" \
  -H "Content-Type: application/json" \
  -d '{
    "list_messages": ["Generate SQL query for user data"],
    "lora_adapter": "sql-lora-adapter",
    "max_tokens": 200
  }'
```

**Note**: The application automatically detects GPU compatibility and falls back to non-LoRA mode if needed.

### Production Deployment

```bash
# Single instance with optimized settings
./deploy.sh --mode single \
  --model meta-llama/Llama-2-7b-hf \
  --host 0.0.0.0 \
  --port 8000 \
  --workers 4 \
  --dtype float16

# Multi-instance production deployment
./deploy.sh --mode multi \
  --model meta-llama/Llama-2-7b-hf \
  --num-instances 8 \
  --base-port 9000 \
  --lb-port 8080 \
  --dtype float16
```

## Monitoring

### Single Instance
- **Health Check**: `GET /health`
- **API Documentation**: `http://localhost:8000/docs`
- **Server Logs**: Monitor uvicorn logs for request/error tracking

### Multi-Instance
- **Load Balancer Health**: `GET /health` (on load balancer port)
- **Instance Statistics**: `GET /stats` (detailed instance information)
- **Individual Instance Health**: `GET /health` (on each instance port)
- **Management Commands**: 
  - `./deploy.sh --status` (overall status)
  - `./deploy.sh --stop` (stop all)
- **Log Files**: 
  - Load balancer: `logs/load_balancer.log`
  - Instances: `logs/instance_0.log`, `logs/instance_1.log`, etc.

## Load Balancer Features

- **Intelligent Distribution**: Automatically splits `list_messages` evenly across healthy instances
- **Health Monitoring**: Continuous health checks with automatic failover
- **Parallel Processing**: Concurrent inference across all available instances
- **Statistics Endpoint**: Real-time monitoring of instance status and performance
- **Graceful Fallback**: Handles instance failures transparently

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**: 
   - Single instance: Reduce `gpu_memory_utilization` or use smaller model
   - Multi-instance: Reduce `--num-instances` or increase GPU memory
2. **LoRA Compilation Error**: The application automatically detects and falls back to non-LoRA mode on older GPUs (Compute Capability < 8.0)
3. **Instance Startup Failure**: Check logs in `logs/` directory for detailed error information
4. **Load Balancer Connection Error**: Verify all instances are healthy before starting load balancer

### Memory Requirements

- **Minimum for single instance**: ~2 GiB GPU memory
- **Recommended for multi-instance**: 6+ GiB GPU memory
- **Per instance memory usage**: ~1.4-2.0 GiB depending on model and settings

### Performance Tips

- Use `dtype="float16"` for memory efficiency
- Set appropriate `gpu_memory_utilization` (0.7 for single, 0.4 for multi-instance)
- Monitor GPU memory usage with `nvidia-smi`
- Use multi-instance deployment for high-throughput scenarios
- Adjust `max_model_len` to reduce memory usage if needed

## System Requirements

- **Python**: 3.8+
- **GPU**: CUDA-compatible GPU (minimum 2 GiB, recommended 6+ GiB for multi-instance)
- **Dependencies**: vLLM 0.9.1, FastAPI, Uvicorn, aiohttp
- **OS**: Linux (tested), Windows/macOS (should work with conda environment)

## Architecture

The LLM Inference Engine uses a single, unified FastAPI application (`app/app.py`) that:

- **Automatic LoRA Detection**: Tries to initialize with LoRA support, automatically falls back to compatibility mode
- **Memory Optimization**: Configured for both single and multi-instance deployments
- **GPU Compatibility**: Works on older GPUs (Compute Capability < 8.0) with automatic fallback
- **Unified Deployment**: Single `deploy.sh` script handles all deployment scenarios