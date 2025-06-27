import json
from dataclasses import dataclass
from typing import Dict, List, Optional, Any

from redis.asyncio import Redis

from app.logger import logger
from app.settings import settings


@dataclass
class AnisotropicCache:
    """Cache layer for anisotropic module data using Redis"""
    
    redis: Redis
    TTL_SECONDS = settings.ISOTROPIC_TTL_SECONDS  # Reuse same TTL setting
    
    def _key_prefix(self, session_id: str) -> str:
        """Generate key prefix for session"""
        return f"storage:{session_id}:anisotropic"
    
    async def set_model_params(self, session_id: str, model_type: str,
                               kappa: Optional[float] = None,
                               alpha: Optional[float] = None) -> None:
        """Store model configuration"""
        key = f"{self._key_prefix(session_id)}:params"
        
        params_data = {
            "model_type": model_type,
            "kappa": kappa,
            "alpha": alpha,
        }
        
        async with self.redis as redis:
            await redis.hset(key, mapping={
                "params": json.dumps(params_data)
            })
            await redis.expire(key, self.TTL_SECONDS)
        
        logger.info(f"Stored anisotropic model params for session {session_id}")
    
    async def get_model_params(self, session_id: str) -> Dict[str, Any]:
        """Retrieve model configuration"""
        key = f"{self._key_prefix(session_id)}:params"
        
        async with self.redis as redis:
            config_str = await redis.hget(key, "params")
        
        if config_str is None:
            raise ValueError(f"No model params found for session {session_id}")
        
        return json.loads(config_str.decode('utf-8'))
    
    async def set_fitted_parameters(self, session_id: str, parameters: Dict[str, float]) -> None:
        """Store fitted model parameters"""
        key = f"{self._key_prefix(session_id)}:params"
        
        async with self.redis as redis:
            await redis.hset(key, mapping={
                "fitted_parameters": json.dumps(parameters)
            })
            await redis.expire(key, self.TTL_SECONDS)
        
        logger.info(f"Stored fitted parameters for session {session_id}")
    
    async def get_fitted_parameters(self, session_id: str) -> Dict[str, float]:
        """Retrieve fitted model parameters"""
        key = f"{self._key_prefix(session_id)}:params"
        
        async with self.redis as redis:
            params_str = await redis.hget(key, "fitted_parameters")
        
        if params_str is None:
            raise ValueError(f"No fitted parameters found for session {session_id}")
        
        return json.loads(params_str.decode('utf-8'))
    
    async def set_optimization_result(self, session_id: str, result: Dict[str, Any]) -> None:
        """Store optimization result metadata"""
        key = f"{self._key_prefix(session_id)}:params"
        
        async with self.redis as redis:
            await redis.hset(key, mapping={
                "optimization_result": json.dumps(result)
            })
            await redis.expire(key, self.TTL_SECONDS)
    
    async def get_optimization_result(self, session_id: str) -> Dict[str, Any]:
        """Retrieve optimization result metadata"""
        key = f"{self._key_prefix(session_id)}:params"
        
        async with self.redis as redis:
            result_str = await redis.hget(key, "optimization_result")
        
        if result_str is None:
            raise ValueError(f"No optimization result found for session {session_id}")
        
        return json.loads(result_str.decode('utf-8'))
    
    async def set_file_data(self, session_id: str, filename: str, byte_data: bytes) -> None:
        """Store file data"""
        key_data = f"{self._key_prefix(session_id)}:anisotropic_data"
        
        async with self.redis as redis:
            await redis.hset(key_data, filename, byte_data)
            await redis.expire(key_data, self.TTL_SECONDS)
        
        logger.info(f"Stored file data: {filename} for session {session_id}")
    
    async def get_files_data(self, session_id: str) -> Dict[bytes, bytes]:
        """Retrieve all file data for session"""
        key_data = f"{self._key_prefix(session_id)}:anisotropic_data"
        
        async with self.redis as redis:
            return await redis.hgetall(key_data)
    
    async def del_file_data(self, session_id: str, filename: str) -> None:
        """Delete specific file data"""
        key_data = f"{self._key_prefix(session_id)}:anisotropic_data"
        
        async with self.redis as redis:
            await redis.hdel(key_data, filename)
        
        logger.info(f"Deleted file data: {filename} for session {session_id}")
    
    async def del_all(self, session_id: str) -> None:
        """Delete all data for session"""
        key_data = f"{self._key_prefix(session_id)}:anisotropic_data"
        key_config = f"{self._key_prefix(session_id)}:params"

        
        async with self.redis as redis:
            val1 = await redis.delete(key_data)
            val2 = await redis.delete(key_config)

            
            logger.info(f"Deleted all anisotropic data for session {session_id}: "
                       f"data={val1}, params={val2}")
    
    async def session_exists(self, session_id: str) -> bool:
        """Check if session has any anisotropic data"""
        key_data = f"{self._key_prefix(session_id)}:anisotropic_data"
        key_config = f"{self._key_prefix(session_id)}:params"
        
        async with self.redis as redis:
            data_exists = await redis.exists(key_data)
            config_exists = await redis.exists(key_config)
        
        return bool(data_exists or config_exists)
    
