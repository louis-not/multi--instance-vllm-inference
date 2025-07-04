from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import asyncio
import logging
import os
from contextlib import asynccontextmanager
from multi_instance_manager import MultiInstanceManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global manager instance
manager: Optional[MultiInstanceManager] = None

class InferenceRequest(BaseModel):
    list_messages: List[str]
    lora_adapter: Optional[str] = None
    max_tokens: Optional[int] = 512
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.9

class InferenceResponse(BaseModel):
    responses: List[str]
    lora_adapter_used: Optional[str] = None
    instances_used: Optional[int] = None
    total_messages: Optional[int] = None
    failed_batches: Optional[int] = None

class InstanceStats(BaseModel):
    total_instances: int
    healthy_instances: int
    unhealthy_instances: int
    instances: List[Dict[str, Any]]

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    global manager
    
    # Get number of instances from environment
    num_instances = int(os.getenv("NUM_INSTANCES", "2"))
    
    logger.info(f"Starting load balancer with {num_instances} instances")
    
    # Initialize manager
    manager = MultiInstanceManager(num_instances=num_instances)
    
    # Start manager context
    async with manager:
        logger.info("Load balancer initialized successfully")
        yield
    
    logger.info("Load balancer shutdown complete")

app = FastAPI(
    title="LLM Inference Load Balancer",
    description="Load balancer for multiple vLLM instances",
    version="1.0.0",
    lifespan=lifespan
)

@app.post("/inference", response_model=InferenceResponse)
async def distributed_inference(request: InferenceRequest, background_tasks: BackgroundTasks):
    """Distribute inference across multiple model instances"""
    if not manager:
        raise HTTPException(status_code=500, detail="Load balancer not initialized")
    
    if not request.list_messages:
        raise HTTPException(status_code=400, detail="list_messages cannot be empty")
    
    try:
        logger.info(f"Processing {len(request.list_messages)} messages with LoRA: {request.lora_adapter}")
        
        # Distribute inference
        result = await manager.distribute_inference(
            messages=request.list_messages,
            lora_adapter=request.lora_adapter,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p
        )
        
        logger.info(f"Distributed inference completed: {result.get('instances_used', 0)} instances used")
        
        return InferenceResponse(**result)
        
    except Exception as e:
        logger.error(f"Distributed inference failed: {e}")
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    if not manager:
        return {"status": "initializing", "load_balancer_ready": False}
    
    try:
        # Update health status
        await manager.update_instance_health()
        
        # Get healthy instances
        healthy_instances = manager.get_healthy_instances()
        
        return {
            "status": "healthy" if healthy_instances else "unhealthy",
            "load_balancer_ready": True,
            "total_instances": len(manager.instances),
            "healthy_instances": len(healthy_instances),
            "unhealthy_instances": len(manager.instances) - len(healthy_instances)
        }
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "error",
            "load_balancer_ready": False,
            "error": str(e)
        }

@app.get("/stats", response_model=InstanceStats)
async def get_instance_stats():
    """Get detailed statistics about model instances"""
    if not manager:
        raise HTTPException(status_code=500, detail="Load balancer not initialized")
    
    try:
        # Update health status first
        await manager.update_instance_health()
        
        stats = manager.get_instance_stats()
        return InstanceStats(**stats)
        
    except Exception as e:
        logger.error(f"Failed to get stats: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get stats: {str(e)}")

@app.post("/health/refresh")
async def refresh_health():
    """Manually refresh health status of all instances"""
    if not manager:
        raise HTTPException(status_code=500, detail="Load balancer not initialized")
    
    try:
        await manager.update_instance_health()
        healthy_instances = manager.get_healthy_instances()
        
        return {
            "message": "Health status refreshed",
            "healthy_instances": len(healthy_instances),
            "total_instances": len(manager.instances)
        }
        
    except Exception as e:
        logger.error(f"Health refresh failed: {e}")
        raise HTTPException(status_code=500, detail=f"Health refresh failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    
    host = os.getenv("LOAD_BALANCER_HOST", "0.0.0.0")
    port = int(os.getenv("LOAD_BALANCER_PORT", "8080"))
    
    uvicorn.run(app, host=host, port=port, log_level="info")