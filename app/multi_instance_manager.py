import asyncio
import aiohttp
import os
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import logging
from concurrent.futures import ThreadPoolExecutor
import json
import random

@dataclass
class ModelInstance:
    host: str
    port: int
    model_name: str
    is_healthy: bool = True
    load_factor: float = 1.0
    
    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"
    
    @property
    def health_url(self) -> str:
        return f"{self.base_url}/health"
    
    @property
    def inference_url(self) -> str:
        return f"{self.base_url}/inference"

class MultiInstanceManager:
    def __init__(self, num_instances: int = 2):
        self.num_instances = num_instances
        self.instances: List[ModelInstance] = []
        self.base_port = int(os.getenv("SERVER_PORT", "8000"))
        self.host = os.getenv("SERVER_HOST", "127.0.0.1")
        self.model_name = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-0.5B-Instruct-AWQ")
        self.logger = logging.getLogger(__name__)
        self.session: Optional[aiohttp.ClientSession] = None
        
        # Initialize instances
        self._initialize_instances()
    
    def _initialize_instances(self):
        """Initialize model instances with different ports"""
        for i in range(self.num_instances):
            port = self.base_port + i
            instance = ModelInstance(
                host=self.host,
                port=port,
                model_name=self.model_name,
                load_factor=1.0
            )
            self.instances.append(instance)
            self.logger.info(f"Initialized instance {i}: {instance.base_url}")
    
    async def __aenter__(self):
        """Async context manager entry"""
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=300)
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()
    
    async def check_instance_health(self, instance: ModelInstance) -> bool:
        """Check if a model instance is healthy"""
        try:
            async with self.session.get(instance.health_url) as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get("engine_ready", False)
                return False
        except Exception as e:
            self.logger.warning(f"Health check failed for {instance.base_url}: {e}")
            return False
    
    async def update_instance_health(self):
        """Update health status for all instances"""
        health_tasks = [
            self.check_instance_health(instance) 
            for instance in self.instances
        ]
        
        health_results = await asyncio.gather(*health_tasks, return_exceptions=True)
        
        for instance, health in zip(self.instances, health_results):
            if isinstance(health, Exception):
                instance.is_healthy = False
                self.logger.error(f"Health check error for {instance.base_url}: {health}")
            else:
                instance.is_healthy = health
                self.logger.debug(f"Instance {instance.base_url} health: {health}")
    
    def get_healthy_instances(self) -> List[ModelInstance]:
        """Get list of healthy instances"""
        return [instance for instance in self.instances if instance.is_healthy]
    
    def split_messages(self, messages: List[str]) -> List[List[str]]:
        """Split messages evenly among healthy instances"""
        healthy_instances = self.get_healthy_instances()
        
        if not healthy_instances:
            raise RuntimeError("No healthy instances available")
        
        num_healthy = len(healthy_instances)
        
        # Calculate messages per instance
        messages_per_instance = len(messages) // num_healthy
        remainder = len(messages) % num_healthy
        
        # Split messages
        split_messages = []
        start_idx = 0
        
        for i in range(num_healthy):
            # Add one extra message to first 'remainder' instances
            batch_size = messages_per_instance + (1 if i < remainder else 0)
            end_idx = start_idx + batch_size
            
            batch = messages[start_idx:end_idx]
            split_messages.append(batch)
            start_idx = end_idx
        
        return split_messages
    
    async def send_batch_to_instance(
        self, 
        instance: ModelInstance, 
        messages: List[str], 
        lora_adapter: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Send a batch of messages to a specific instance"""
        if not messages:
            return {"responses": [], "lora_adapter_used": lora_adapter}
        
        payload = {
            "list_messages": messages,
            "lora_adapter": lora_adapter,
            **kwargs
        }
        
        try:
            async with self.session.post(
                instance.inference_url,
                json=payload,
                headers={"Content-Type": "application/json"}
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    self.logger.debug(f"Success from {instance.base_url}: {len(messages)} messages")
                    return result
                else:
                    error_text = await response.text()
                    raise RuntimeError(f"HTTP {response.status}: {error_text}")
        
        except Exception as e:
            self.logger.error(f"Request failed to {instance.base_url}: {e}")
            # Mark instance as unhealthy
            instance.is_healthy = False
            raise
    
    async def distribute_inference(
        self, 
        messages: List[str], 
        lora_adapter: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Distribute inference across multiple instances"""
        if not messages:
            return {"responses": [], "lora_adapter_used": lora_adapter}
        
        # Update health status
        await self.update_instance_health()
        
        # Get healthy instances
        healthy_instances = self.get_healthy_instances()
        
        if not healthy_instances:
            raise RuntimeError("No healthy instances available for inference")
        
        # Split messages among instances
        message_batches = self.split_messages(messages)
        
        self.logger.info(f"Distributing {len(messages)} messages across {len(healthy_instances)} instances")
        
        # Create tasks for parallel processing
        tasks = []
        for instance, batch in zip(healthy_instances, message_batches):
            if batch:  # Only create task if batch is not empty
                task = self.send_batch_to_instance(
                    instance, batch, lora_adapter, **kwargs
                )
                tasks.append(task)
        
        # Execute all tasks in parallel
        try:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Combine results
            all_responses = []
            failed_batches = []
            
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    self.logger.error(f"Batch {i} failed: {result}")
                    failed_batches.append(i)
                else:
                    all_responses.extend(result.get("responses", []))
            
            if failed_batches:
                self.logger.warning(f"Failed batches: {failed_batches}")
                if len(failed_batches) == len(results):
                    raise RuntimeError("All inference batches failed")
            
            return {
                "responses": all_responses,
                "lora_adapter_used": lora_adapter,
                "instances_used": len(healthy_instances),
                "total_messages": len(messages),
                "failed_batches": len(failed_batches)
            }
            
        except Exception as e:
            self.logger.error(f"Distributed inference failed: {e}")
            raise
    
    def get_instance_stats(self) -> Dict[str, Any]:
        """Get statistics about model instances"""
        healthy_count = sum(1 for instance in self.instances if instance.is_healthy)
        
        stats = {
            "total_instances": len(self.instances),
            "healthy_instances": healthy_count,
            "unhealthy_instances": len(self.instances) - healthy_count,
            "instances": [
                {
                    "url": instance.base_url,
                    "model": instance.model_name,
                    "healthy": instance.is_healthy,
                    "load_factor": instance.load_factor
                }
                for instance in self.instances
            ]
        }
        
        return stats