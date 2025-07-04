#!/bin/bash

# Unified LLM Inference Engine Deployment Script
# Supports single instance, multi-instance, and stop functionality

set -e

# Default values
DEFAULT_MODEL="Qwen/Qwen2.5-0.5B-Instruct-AWQ"
DEFAULT_HOST="0.0.0.0"
DEFAULT_PORT="8000"
DEFAULT_LB_PORT="8080"
DEFAULT_LORA_PATH="./lora_adapters"
DEFAULT_MAX_LORA_RANK="64"
DEFAULT_DTYPE="float16"
DEFAULT_WORKERS="1"
DEFAULT_NUM_INSTANCES="2"
DEFAULT_MODE="single"

# Function to display usage
show_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Deployment Modes:"
    echo "  --mode single               Single instance deployment (default)"
    echo "  --mode multi                Multi-instance deployment with load balancer"
    echo "  --stop                      Stop all running instances"
    echo "  --status                    Show status of running instances"
    echo ""
    echo "Common Options:"
    echo "  -m, --model MODEL           Model name/path (default: $DEFAULT_MODEL)"
    echo "  -h, --host HOST             Host to bind to (default: $DEFAULT_HOST)"
    echo "  -d, --dtype DTYPE           Data type (default: $DEFAULT_DTYPE)"
    echo "  -l, --lora-path PATH        Path to LoRA adapters directory (default: $DEFAULT_LORA_PATH)"
    echo "  -r, --max-lora-rank RANK    Maximum LoRA rank (default: $DEFAULT_MAX_LORA_RANK)"
    echo "  --help                      Show this help message"
    echo ""
    echo "Single Instance Options:"
    echo "  -p, --port PORT             Port to bind to (default: $DEFAULT_PORT)"
    echo "  -w, --workers WORKERS       Number of workers (default: $DEFAULT_WORKERS)"
    echo ""
    echo "Multi-Instance Options:"
    echo "  -p, --base-port PORT        Base port for instances (default: $DEFAULT_PORT)"
    echo "  -b, --lb-port PORT          Load balancer port (default: $DEFAULT_LB_PORT)"
    echo "  -n, --num-instances NUM     Number of model instances (default: $DEFAULT_NUM_INSTANCES)"
    echo ""
    echo "Environment Variables:"
    echo "  MODEL_NAME                  Model name/path"
    echo "  SERVER_HOST                 Host to bind to"
    echo "  SERVER_PORT                 Port (single) / Base port (multi)"
    echo "  LOAD_BALANCER_PORT          Load balancer port (multi-instance only)"
    echo "  NUM_INSTANCES               Number of model instances (multi-instance only)"
    echo "  LORA_ADAPTERS_PATH          Path to LoRA adapters directory"
    echo "  MAX_LORA_RANK               Maximum LoRA rank"
    echo "  MODEL_DTYPE                 Data type"
    echo "  UVICORN_WORKERS             Number of Uvicorn workers (single instance only)"
    echo ""
    echo "Examples:"
    echo "  # Single instance deployment"
    echo "  $0 --mode single --port 8000"
    echo "  $0  # defaults to single mode"
    echo ""
    echo "  # Multi-instance deployment"
    echo "  $0 --mode multi --num-instances 4 --lb-port 8080"
    echo "  $0 --mode multi -n 3 -p 9000 -b 9080"
    echo ""
    echo "  # Management commands"
    echo "  $0 --stop"
    echo "  $0 --status"
    echo ""
    echo "  # Environment variable usage"
    echo "  MODEL_NAME=meta-llama/Llama-2-7b-hf $0 --mode single"
    echo "  NUM_INSTANCES=4 $0 --mode multi"
}

# Function to stop all instances
stop_instances() {
    echo "Stopping all LLM inference instances..."
    
    # Stop load balancer
    echo "  Stopping load balancer..."
    pkill -f "load_balancer.py" 2>/dev/null || true
    
    # Stop model instances using multiple patterns
    echo "  Stopping model instances..."
    pkill -f "uvicorn app:app" 2>/dev/null || true
    pkill -f "uvicorn.*app:app" 2>/dev/null || true
    
    # Wait a moment for graceful shutdown
    sleep 2
    
    # Force kill any remaining uvicorn processes
    echo "  Force stopping any remaining uvicorn processes..."
    pkill -9 -f "uvicorn.*app" 2>/dev/null || true
    
    # Also kill by process name if needed
    pkill -f "python.*uvicorn" 2>/dev/null || true
    pkill -9 -f "python.*uvicorn" 2>/dev/null || true
    
    # Clean up PID files
    if [[ -d "logs" ]]; then
        rm -f logs/*.pid
    fi
    
    # Wait a moment for GPU memory to be freed
    echo "  Waiting for GPU memory to be freed..."
    sleep 3
    
    # Verify all processes are stopped
    remaining_processes=$(pgrep -f "uvicorn.*app" 2>/dev/null || true)
    if [[ -n "$remaining_processes" ]]; then
        echo "  Warning: Some processes may still be running:"
        ps -p $remaining_processes -o pid,cmd 2>/dev/null || true
        echo "  You may need to manually kill them with: kill -9 $remaining_processes"
    else
        echo "✓ All instances stopped successfully"
    fi
}

# Function to show status
show_status() {
    echo "LLM Inference Engine Status:"
    echo "=============================="
    
    # Check load balancer
    if pgrep -f "load_balancer.py" > /dev/null; then
        echo "✓ Load balancer is running"
        lb_pid=$(pgrep -f "load_balancer.py")
        echo "  PID: $lb_pid"
        
        # Try to get load balancer port from logs
        if [[ -f "logs/load_balancer.log" ]]; then
            lb_port=$(grep -o "Uvicorn running on.*:\([0-9]*\)" logs/load_balancer.log | tail -1 | grep -o "[0-9]*$" || echo "unknown")
            echo "  Port: $lb_port"
        fi
    else
        echo "✗ Load balancer is not running"
    fi
    
    # Check model instances
    instance_pids=$(pgrep -f "app:app\|simple_app:app" || true)
    if [[ -n "$instance_pids" ]]; then
        echo "✓ Model instances are running:"
        echo "$instance_pids" | while read pid; do
            # Try to extract port from process command line
            port=$(ps -p $pid -o args= | grep -o "\--port [0-9]*" | grep -o "[0-9]*" || echo "unknown")
            echo "  PID: $pid, Port: $port"
        done
    else
        echo "✗ No model instances are running"
    fi
    
    # Check ports
    echo ""
    echo "Active Ports (8000-9999 range):"
    netstat -tlnp 2>/dev/null | grep -E ":(8[0-9]{3}|9[0-9]{3})" | head -10 || echo "  No ports found in range 8000-9999"
    
    # Show log files if they exist
    if [[ -d "logs" ]]; then
        echo ""
        echo "Available Log Files:"
        ls -la logs/*.log 2>/dev/null | sed 's/^/  /' || echo "  No log files found"
    fi
}

# Function to check dependencies
check_dependencies() {
    echo "Checking Python dependencies..."
    
    # Activate conda environment
    source ~/miniconda3/etc/profile.d/conda.sh
    conda activate llm-inference
    
    python3 -c "import vllm, fastapi, uvicorn" 2>/dev/null || {
        echo "Error: Required Python packages not installed."
        echo "Please run: conda activate llm-inference && pip install -r requirements.txt"
        exit 1
    }
    
    # Check for aiohttp for multi-instance mode
    if [[ "$DEPLOY_MODE" == "multi" ]]; then
        python3 -c "import aiohttp" 2>/dev/null || {
            echo "Error: aiohttp package required for multi-instance mode."
            echo "Please run: conda activate llm-inference && pip install aiohttp"
            exit 1
        }
    fi
}

# Function to deploy single instance
deploy_single() {
    echo "========================================="
    echo "Single Instance LLM Inference Engine"
    echo "========================================="
    echo "Model: $MODEL_NAME"
    echo "Host: $SERVER_HOST"
    echo "Port: $SERVER_PORT"
    echo "LoRA Adapters Path: $LORA_ADAPTERS_PATH"
    echo "Max LoRA Rank: $MAX_LORA_RANK"
    echo "Data Type: $MODEL_DTYPE"
    echo "Workers: $UVICORN_WORKERS"
    echo "========================================="
    
    # Create logs directory
    mkdir -p logs
    
    # Change to app directory
    cd "$(dirname "$0")/app"
    
    echo "Starting single instance server..."
    
    # Start server
    nohup bash -c "source ~/miniconda3/etc/profile.d/conda.sh && conda activate llm-inference && uvicorn app:app \
        --host '$SERVER_HOST' \
        --port '$SERVER_PORT' \
        --workers '$UVICORN_WORKERS' \
        --log-level info \
        --access-log" \
        > "../logs/server.log" 2>&1 &
    
    server_pid=$!
    echo "Server started with PID: $server_pid"
    echo $server_pid > "../logs/server.pid"
    
    # Wait for server to start
    echo "Waiting for server to initialize..."
    sleep 5
    
    # Verify server is running
    max_retries=30
    retry_count=0
    
    while [[ $retry_count -lt $max_retries ]]; do
        if curl -sf "http://$SERVER_HOST:$SERVER_PORT/health" > /dev/null 2>&1; then
            echo "✓ Server is healthy"
            break
        else
            ((retry_count++))
            if [[ $retry_count -eq $max_retries ]]; then
                echo "✗ Server failed to start properly"
                echo "  Check logs at logs/server.log"
                exit 1
            else
                sleep 2
            fi
        fi
    done
    
    echo ""
    echo "========================================="
    echo "Single Instance Deployment Complete!"
    echo "========================================="
    echo "Server: http://$SERVER_HOST:$SERVER_PORT"
    echo "API Documentation: http://$SERVER_HOST:$SERVER_PORT/docs"
    echo "Health Check: http://$SERVER_HOST:$SERVER_PORT/health"
    echo ""
    echo "Log File: logs/server.log"
    echo ""
    echo "Management Commands:"
    echo "  Status: $0 --status"
    echo "  Stop: $0 --stop"
    echo "========================================="
}

# Function to deploy multi-instance
deploy_multi() {
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
    
    # Create logs directory
    mkdir -p logs
    
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
        nohup bash -c "source ~/miniconda3/etc/profile.d/conda.sh && conda activate llm-inference && uvicorn app:app \
            --host '$SERVER_HOST' \
            --port '$port' \
            --log-level info \
            --access-log" \
            > "../logs/instance_${i}.log" 2>&1 &
        
        instance_pid=$!
        echo "  Instance $i started with PID: $instance_pid"
        echo $instance_pid > "../logs/instance_${i}.pid"
    done
    
    # Wait for instances to start
    echo "Waiting for model instances to initialize..."
    sleep 5
    
    # Verify instances are running
    echo "Verifying instance health..."
    for ((i=0; i<$NUM_INSTANCES; i++)); do
        port=$((SERVER_PORT + i))
        
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
}

# Parse command line arguments
DEPLOY_MODE="$DEFAULT_MODE"

while [[ $# -gt 0 ]]; do
    case $1 in
        --mode)
            DEPLOY_MODE="$2"
            shift 2
            ;;
        -m|--model)
            MODEL_NAME="$2"
            shift 2
            ;;
        -h|--host)
            SERVER_HOST="$2"
            shift 2
            ;;
        -p|--port|--base-port)
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
        -w|--workers)
            UVICORN_WORKERS="$2"
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

# Validate deployment mode
if [[ "$DEPLOY_MODE" != "single" && "$DEPLOY_MODE" != "multi" ]]; then
    echo "Error: Invalid deployment mode '$DEPLOY_MODE'. Use 'single' or 'multi'."
    exit 1
fi

# Set defaults from environment variables or use defaults
export MODEL_NAME="${MODEL_NAME:-$DEFAULT_MODEL}"
export SERVER_HOST="${SERVER_HOST:-$DEFAULT_HOST}"
export SERVER_PORT="${SERVER_PORT:-$DEFAULT_PORT}"
export LOAD_BALANCER_PORT="${LOAD_BALANCER_PORT:-$DEFAULT_LB_PORT}"
export NUM_INSTANCES="${NUM_INSTANCES:-$DEFAULT_NUM_INSTANCES}"
export LORA_ADAPTERS_PATH="${LORA_ADAPTERS_PATH:-$DEFAULT_LORA_PATH}"
export MAX_LORA_RANK="${MAX_LORA_RANK:-$DEFAULT_MAX_LORA_RANK}"
export MODEL_DTYPE="${MODEL_DTYPE:-$DEFAULT_DTYPE}"
export UVICORN_WORKERS="${UVICORN_WORKERS:-$DEFAULT_WORKERS}"

# Validate parameters
if [[ "$DEPLOY_MODE" == "multi" ]]; then
    if [[ ! "$NUM_INSTANCES" =~ ^[0-9]+$ ]] || [[ "$NUM_INSTANCES" -lt 1 ]]; then
        echo "Error: NUM_INSTANCES must be a positive integer"
        exit 1
    fi
fi

if [[ ! "$SERVER_PORT" =~ ^[0-9]+$ ]]; then
    echo "Error: SERVER_PORT must be a valid port number"
    exit 1
fi

if [[ ! "$UVICORN_WORKERS" =~ ^[0-9]+$ ]] || [[ "$UVICORN_WORKERS" -lt 1 ]]; then
    echo "Error: UVICORN_WORKERS must be a positive integer"
    exit 1
fi

# Create LoRA adapters directory if it doesn't exist
if [[ ! -d "$LORA_ADAPTERS_PATH" ]]; then
    echo "Creating LoRA adapters directory: $LORA_ADAPTERS_PATH"
    mkdir -p "$LORA_ADAPTERS_PATH"
fi

# Check dependencies
check_dependencies

# Stop any existing instances
echo "Stopping any existing instances..."
stop_instances
sleep 2

# Deploy based on mode
case $DEPLOY_MODE in
    single)
        deploy_single
        ;;
    multi)
        deploy_multi
        ;;
esac