#!/bin/bash

# Multi-Instance LLM Inference Engine Deployment Script
# This script starts multiple vLLM instances and a load balancer

set -e

# Default values
DEFAULT_MODEL="Qwen/Qwen2.5-0.5B-Instruct-AWQ"
DEFAULT_HOST="0.0.0.0"
DEFAULT_BASE_PORT="8000"
DEFAULT_LB_PORT="8080"
DEFAULT_LORA_PATH="./lora_adapters"
DEFAULT_MAX_LORA_RANK="64"
DEFAULT_DTYPE="float16"
DEFAULT_NUM_INSTANCES="2"

# Function to display usage
show_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -m, --model MODEL           Model name/path (default: $DEFAULT_MODEL)"
    echo "  -h, --host HOST             Host to bind to (default: $DEFAULT_HOST)"
    echo "  -p, --base-port PORT        Base port for instances (default: $DEFAULT_BASE_PORT)"
    echo "  -b, --lb-port PORT          Load balancer port (default: $DEFAULT_LB_PORT)"
    echo "  -n, --num-instances NUM     Number of model instances (default: $DEFAULT_NUM_INSTANCES)"
    echo "  -l, --lora-path PATH        Path to LoRA adapters directory (default: $DEFAULT_LORA_PATH)"
    echo "  -r, --max-lora-rank RANK    Maximum LoRA rank (default: $DEFAULT_MAX_LORA_RANK)"
    echo "  -d, --dtype DTYPE           Data type (default: $DEFAULT_DTYPE)"
    echo "  --stop                      Stop all running instances"
    echo "  --status                    Show status of running instances"
    echo "  --help                      Show this help message"
    echo ""
    echo "Environment Variables:"
    echo "  MODEL_NAME                  Model name/path"
    echo "  SERVER_HOST                 Host to bind to"
    echo "  SERVER_PORT                 Base port for instances"
    echo "  LOAD_BALANCER_PORT          Load balancer port"
    echo "  NUM_INSTANCES               Number of model instances"
    echo "  LORA_ADAPTERS_PATH          Path to LoRA adapters directory"
    echo "  MAX_LORA_RANK               Maximum LoRA rank"
    echo "  MODEL_DTYPE                 Data type"
    echo ""
    echo "Examples:"
    echo "  $0 --num-instances 4 --model meta-llama/Llama-2-7b-hf"
    echo "  $0 -n 3 -p 9000 -b 9080"
    echo "  $0 --stop"
    echo "  NUM_INSTANCES=4 $0"
}

# Function to stop all instances
stop_instances() {
    echo "Stopping all LLM inference instances..."
    
    # Stop load balancer
    pkill -f "load_balancer.py" || true
    
    # Stop model instances
    pkill -f "app:app" || true
    
    echo "All instances stopped"
}

# Function to show status
show_status() {
    echo "LLM Inference Engine Status:"
    echo "============================="
    
    # Check load balancer
    if pgrep -f "load_balancer.py" > /dev/null; then
        echo "✓ Load balancer is running"
        lb_pid=$(pgrep -f "load_balancer.py")
        echo "  PID: $lb_pid"
    else
        echo "✗ Load balancer is not running"
    fi
    
    # Check model instances
    instance_pids=$(pgrep -f "app:app" || true)
    if [ -n "$instance_pids" ]; then
        echo "✓ Model instances are running:"
        echo "$instance_pids" | while read pid; do
            echo "  PID: $pid"
        done
    else
        echo "✗ No model instances are running"
    fi
    
    # Check ports
    echo ""
    echo "Port Status:"
    netstat -tlnp 2>/dev/null | grep -E ":(8[0-9]{3}|9[0-9]{3})" | head -10 || echo "No ports found in range 8000-9999"
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
        -p|--base-port)
            SERVER_PORT="$2"
            shift 2
            ;;
        -b|--lb-port)
            LOAD_BALANCER_PORT="$2"
            shift 2
            ;;
        -n|--num-instances)
            NUM_INSTANCES="$2"
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
        --stop)
            stop_instances
            exit 0
            ;;
        --status)
            show_status
            exit 0
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
export SERVER_PORT="${SERVER_PORT:-$DEFAULT_BASE_PORT}"
export LOAD_BALANCER_PORT="${LOAD_BALANCER_PORT:-$DEFAULT_LB_PORT}"
export NUM_INSTANCES="${NUM_INSTANCES:-$DEFAULT_NUM_INSTANCES}"
export LORA_ADAPTERS_PATH="${LORA_ADAPTERS_PATH:-$DEFAULT_LORA_PATH}"
export MAX_LORA_RANK="${MAX_LORA_RANK:-$DEFAULT_MAX_LORA_RANK}"
export MODEL_DTYPE="${MODEL_DTYPE:-$DEFAULT_DTYPE}"

# Validate parameters
if [[ ! "$NUM_INSTANCES" =~ ^[0-9]+$ ]] || [[ "$NUM_INSTANCES" -lt 1 ]]; then
    echo "Error: NUM_INSTANCES must be a positive integer"
    exit 1
fi

if [[ ! "$SERVER_PORT" =~ ^[0-9]+$ ]]; then
    echo "Error: SERVER_PORT must be a valid port number"
    exit 1
fi

# Create LoRA adapters directory if it doesn't exist
if [[ ! -d "$LORA_ADAPTERS_PATH" ]]; then
    echo "Creating LoRA adapters directory: $LORA_ADAPTERS_PATH"
    mkdir -p "$LORA_ADAPTERS_PATH"
fi

# Check if Python dependencies are installed
echo "Checking Python dependencies..."
source ~/miniconda3/etc/profile.d/conda.sh
conda activate llm-inference
python3 -c "import vllm, fastapi, uvicorn, aiohttp" 2>/dev/null || {
    echo "Error: Required Python packages not installed."
    echo "Please run: conda activate llm-inference && pip install -r requirements.txt"
    exit 1
}

# Stop any existing instances
echo "Stopping any existing instances..."
stop_instances
sleep 2

# Display configuration
echo "========================================="
echo "Multi-Instance LLM Inference Engine"
echo "========================================="
echo "Model: $MODEL_NAME"
echo "Host: $SERVER_HOST"
echo "Base Port: $SERVER_PORT"
echo "Load Balancer Port: $LOAD_BALANCER_PORT"
echo "Number of Instances: $NUM_INSTANCES"
echo "LoRA Adapters Path: $LORA_ADAPTERS_PATH"
echo "Max LoRA Rank: $MAX_LORA_RANK"
echo "Data Type: $MODEL_DTYPE"
echo "========================================="

# Change to app directory
cd "$(dirname "$0")/app"

# Start model instances
echo "Starting $NUM_INSTANCES model instances..."
for ((i=0; i<$NUM_INSTANCES; i++)); do
    port=$((SERVER_PORT + i))
    
    echo "Starting instance $i on port $port..."
    
    # Set environment variables for this instance
    export INSTANCE_ID=$i
    export INSTANCE_PORT=$port
    
    # Start instance in background
    nohup bash -c "source ~/miniconda3/etc/profile.d/conda.sh && conda activate llm-inference && uvicorn simple_app:app \
        --host '$SERVER_HOST' \
        --port '$port' \
        --log-level info \
        --access-log" \
        > "../logs/instance_${i}.log" 2>&1 &
    
    instance_pid=$!
    echo "  Instance $i started with PID: $instance_pid"
    
    # Save PID for later management
    echo $instance_pid > "../logs/instance_${i}.pid"
done

# Wait for instances to start
echo "Waiting for model instances to initialize..."
sleep 5

# Verify instances are running
echo "Verifying instance health..."
for ((i=0; i<$NUM_INSTANCES; i++)); do
    port=$((SERVER_PORT + i))
    
    # Check if instance is responding
    max_retries=30
    retry_count=0
    
    while [[ $retry_count -lt $max_retries ]]; do
        if curl -sf "http://$SERVER_HOST:$port/health" > /dev/null 2>&1; then
            echo "  ✓ Instance $i (port $port) is healthy"
            break
        else
            ((retry_count++))
            if [[ $retry_count -eq $max_retries ]]; then
                echo "  ✗ Instance $i (port $port) failed to start properly"
                echo "    Check logs at ../logs/instance_${i}.log"
            else
                sleep 2
            fi
        fi
    done
done

# Start load balancer
echo "Starting load balancer on port $LOAD_BALANCER_PORT..."
nohup bash -c "source ~/miniconda3/etc/profile.d/conda.sh && conda activate llm-inference && python3 load_balancer.py" \
    > "../logs/load_balancer.log" 2>&1 &
lb_pid=$!
echo "Load balancer started with PID: $lb_pid"
echo $lb_pid > "../logs/load_balancer.pid"

# Wait for load balancer to start
echo "Waiting for load balancer to initialize..."
sleep 3

# Verify load balancer
max_retries=15
retry_count=0

while [[ $retry_count -lt $max_retries ]]; do
    if curl -sf "http://$SERVER_HOST:$LOAD_BALANCER_PORT/health" > /dev/null 2>&1; then
        echo "✓ Load balancer is healthy"
        break
    else
        ((retry_count++))
        if [[ $retry_count -eq $max_retries ]]; then
            echo "✗ Load balancer failed to start properly"
            echo "  Check logs at ../logs/load_balancer.log"
        else
            sleep 2
        fi
    fi
done

echo ""
echo "========================================="
echo "Multi-Instance Deployment Complete!"
echo "========================================="
echo "Load Balancer: http://$SERVER_HOST:$LOAD_BALANCER_PORT"
echo "API Documentation: http://$SERVER_HOST:$LOAD_BALANCER_PORT/docs"
echo "Health Check: http://$SERVER_HOST:$LOAD_BALANCER_PORT/health"
echo "Instance Stats: http://$SERVER_HOST:$LOAD_BALANCER_PORT/stats"
echo ""
echo "Model Instances:"
for ((i=0; i<$NUM_INSTANCES; i++)); do
    port=$((SERVER_PORT + i))
    echo "  Instance $i: http://$SERVER_HOST:$port"
done
echo ""
echo "Log Files:"
echo "  Load Balancer: logs/load_balancer.log"
for ((i=0; i<$NUM_INSTANCES; i++)); do
    echo "  Instance $i: logs/instance_${i}.log"
done
echo ""
echo "Management Commands:"
echo "  Status: $0 --status"
echo "  Stop: $0 --stop"
echo "========================================="