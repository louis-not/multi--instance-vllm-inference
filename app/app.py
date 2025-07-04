from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
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
    max_lora_rank = int(os.getenv("MAX_LORA_RANK", "64"))
    dtype = os.getenv("MODEL_DTYPE", "float16")
    
    print(f"Initializing LLM with model: {model_name}")
    print(f"Max LoRA rank: {max_lora_rank}")
    print(f"Data type: {dtype}")
    
    # Disable LoRA for older GPUs due to Triton compilation issues
    try:
        llm_engine = LLM(
            model=model_name,
            enable_lora=True,
            max_lora_rank=max_lora_rank,
            dtype=dtype
        )
    except Exception as e:
        print(f"LoRA initialization failed: {e}")
        print("Falling back to model without LoRA support...")
        llm_engine = LLM(
            model=model_name,
            enable_lora=False,
            dtype=dtype
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
        
        lora_request = None
        if request.lora_adapter and hasattr(llm_engine, 'llm_engine') and llm_engine.llm_engine.lora_config:
            lora_adapters_path = os.getenv("LORA_ADAPTERS_PATH", "./lora_adapters")
            lora_request = LoRARequest(
                request.lora_adapter,
                1,
                f"{lora_adapters_path}/{request.lora_adapter}"
            )
        elif request.lora_adapter:
            print(f"Warning: LoRA adapter '{request.lora_adapter}' requested but LoRA is not enabled")
        
        outputs = llm_engine.generate(
            request.list_messages,
            sampling_params,
            lora_request=lora_request
        )
        
        responses = [output.outputs[0].text.strip() for output in outputs]
        
        return InferenceResponse(
            responses=responses,
            lora_adapter_used=request.lora_adapter
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")

@app.get("/health")
async def health_check():
    return {"status": "healthy", "engine_ready": llm_engine is not None}

if __name__ == "__main__":
    import uvicorn
    
    host = os.getenv("SERVER_HOST", "0.0.0.0")
    port = int(os.getenv("SERVER_PORT", "8000"))
    
    uvicorn.run(app, host=host, port=port)