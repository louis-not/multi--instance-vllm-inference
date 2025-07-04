#!/bin/bash

# LLM Inference Engine Server Startup Script
# This script starts the FastAPI server with configurable parameters

set -e

# Default values
DEFAULT_MODEL="Qwen/Qwen2.5-0.5B-Instruct-AWQ"
DEFAULT_HOST="0.0.0.0"
DEFAULT_PORT="8000"
DEFAULT_LORA_PATH="./lora_adapters"
DEFAULT_MAX_LORA_RANK="64"
DEFAULT_DTYPE="float16"
DEFAULT_WORKERS="1"

# Function to display usage
show_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -m, --model MODEL           Model name/path (default: $DEFAULT_MODEL)"
    echo "  -h, --host HOST             Host to bind to (default: $DEFAULT_HOST)"
    echo "  -p, --port PORT             Port to bind to (default: $DEFAULT_PORT)"
    echo "  -l, --lora-path PATH        Path to LoRA adapters directory (default: $DEFAULT_LORA_PATH)"
    echo "  -r, --max-lora-rank RANK    Maximum LoRA rank (default: $DEFAULT_MAX_LORA_RANK)"
    echo "  -d, --dtype DTYPE           Data type (default: $DEFAULT_DTYPE)"
    echo "  -w, --workers WORKERS       Number of workers (default: $DEFAULT_WORKERS)"
    echo "  --help                      Show this help message"
    echo ""
    echo "Environment Variables:"
    echo "  MODEL_NAME                  Model name/path"
    echo "  SERVER_HOST                 Host to bind to"
    echo "  SERVER_PORT                 Port to bind to"
    echo "  LORA_ADAPTERS_PATH          Path to LoRA adapters directory"
    echo "  MAX_LORA_RANK               Maximum LoRA rank"
    echo "  MODEL_DTYPE                 Data type"
    echo "  UVICORN_WORKERS             Number of workers"
    echo ""
    echo "Examples:"
    echo "  $0 --model meta-llama/Llama-2-7b-hf --port 8080"
    echo "  $0 -m microsoft/DialoGPT-medium -l /path/to/lora/adapters"
    echo "  MODEL_NAME=meta-llama/Llama-2-7b-hf $0"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -m|--model)
            MODEL_NAME="$2"
            shift 2
            ;;
        -h|--host)
            SERVER_HOST="$2"
            shift 2
            ;;
        -p|--port)
            SERVER_PORT="$2"
            shift 2
            ;;
        -l|--lora-path)
            LORA_ADAPTERS_PATH="$2"
            shift 2
            ;;
        -r|--max-lora-rank)
            MAX_LORA_RANK="$2"
            shift 2
            ;;
        -d|--dtype)
            MODEL_DTYPE="$2"
            shift 2
            ;;
        -w|--workers)
            UVICORN_WORKERS="$2"
            shift 2
            ;;
        --help)
            show_usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            show_usage
            exit 1
            ;;
    esac
done

# Set defaults from environment variables or use defaults
export MODEL_NAME="${MODEL_NAME:-$DEFAULT_MODEL}"
export SERVER_HOST="${SERVER_HOST:-$DEFAULT_HOST}"
export SERVER_PORT="${SERVER_PORT:-$DEFAULT_PORT}"
export LORA_ADAPTERS_PATH="${LORA_ADAPTERS_PATH:-$DEFAULT_LORA_PATH}"
export MAX_LORA_RANK="${MAX_LORA_RANK:-$DEFAULT_MAX_LORA_RANK}"
export MODEL_DTYPE="${MODEL_DTYPE:-$DEFAULT_DTYPE}"
export UVICORN_WORKERS="${UVICORN_WORKERS:-$DEFAULT_WORKERS}"

# Validate required parameters
if [[ -z "$MODEL_NAME" ]]; then
    echo "Error: MODEL_NAME is required"
    show_usage
    exit 1
fi

# Create LoRA adapters directory if it doesn't exist
if [[ ! -d "$LORA_ADAPTERS_PATH" ]]; then
    echo "Creating LoRA adapters directory: $LORA_ADAPTERS_PATH"
    mkdir -p "$LORA_ADAPTERS_PATH"
fi

# Display configuration
echo "========================================="
echo "LLM Inference Engine Server Configuration"
echo "========================================="
echo "Model: $MODEL_NAME"
echo "Host: $SERVER_HOST"
echo "Port: $SERVER_PORT"
echo "LoRA Adapters Path: $LORA_ADAPTERS_PATH"
echo "Max LoRA Rank: $MAX_LORA_RANK"
echo "Data Type: $MODEL_DTYPE"
echo "Workers: $UVICORN_WORKERS"
echo "========================================="

# Check if Python dependencies are installed
echo "Checking Python dependencies..."
python3 -c "import vllm, fastapi, uvicorn" 2>/dev/null || {
    echo "Error: Required Python packages not installed."
    echo "Please run: pip install -r requirements.txt"
    exit 1
}

# Start the server
echo "Starting LLM Inference Engine Server..."
echo "Server will be available at: http://$SERVER_HOST:$SERVER_PORT"
echo "Health check: http://$SERVER_HOST:$SERVER_PORT/health"
echo "API docs: http://$SERVER_HOST:$SERVER_PORT/docs"
echo ""

# Change to app directory
cd "$(dirname "$0")/app"

# Start the server with uvicorn
exec uvicorn app:app \
    --host "$SERVER_HOST" \
    --port "$SERVER_PORT" \
    --workers "$UVICORN_WORKERS" \
    --log-level info