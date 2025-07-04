from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from vllm import LLM, SamplingParams
from contextlib import asynccontextmanager
import os

llm_engine = None

class InferenceRequest(BaseModel):
    list_messages: List[str]
    lora_adapter: Optional[str] = None
    max_tokens: Optional[int] = 512
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.9

class InferenceResponse(BaseModel):
    responses: List[str]
    lora_adapter_used: Optional[str] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global llm_engine
    
    model_name = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-0.5B-Instruct-AWQ")
    dtype = os.getenv("MODEL_DTYPE", "float16")
    
    print(f"Initializing LLM with model: {model_name}")
    print(f"Data type: {dtype}")
    print("LoRA support: DISABLED for compatibility")
    
    # Initialize without LoRA to avoid compilation issues
    llm_engine = LLM(
        model=model_name,
        enable_lora=False,
        dtype=dtype,
        gpu_memory_utilization=0.4,  # Reduce memory usage for multiple instances
        max_model_len=2048,  # Correct parameter name
        enforce_eager=True  # Disable CUDA graphs to save memory
    )
    yield
    llm_engine = None

app = FastAPI(title="LLM Inference Engine", version="1.0.0", lifespan=lifespan)

@app.post("/inference", response_model=InferenceResponse)
async def batch_inference(request: InferenceRequest):
    if not llm_engine:
        raise HTTPException(status_code=500, detail="LLM engine not initialized")
    
    try:
        sampling_params = SamplingParams(
            temperature=request.temperature,
            top_p=request.top_p,
            max_tokens=request.max_tokens
        )
        
        if request.lora_adapter:
            print(f"Warning: LoRA adapter '{request.lora_adapter}' requested but LoRA is disabled for compatibility")
        
        outputs = llm_engine.generate(
            request.list_messages,
            sampling_params
        )
        
        responses = [output.outputs[0].text.strip() for output in outputs]
        
        return InferenceResponse(
            responses=responses,
            lora_adapter_used=None  # LoRA disabled
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")

@app.get("/health")
async def health_check():
    return {"status": "healthy", "engine_ready": llm_engine is not None, "lora_enabled": False}

if __name__ == "__main__":
    import uvicorn
    
    host = os.getenv("SERVER_HOST", "0.0.0.0")
    port = int(os.getenv("SERVER_PORT", "8000"))
    
    uvicorn.run(app, host=host, port=port)