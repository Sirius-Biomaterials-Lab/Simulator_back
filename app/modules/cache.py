import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Any

from redis.asyncio import Redis

from app.logger import logger
from app.settings import settings


@dataclass
class ModuleCache(ABC):

    redis: Redis
    TTL_SECONDS = settings.ISOTROPIC_TTL_SECONDS  # Reuse same TTL setting

    @abstractmethod
    def _key_prefix(self, session_id: str) -> str:
        """Generate key prefix for session"""
        raise NotImplementedError()

    async def set_model_params(self, session_id: str, params_data: dict) -> None:
        """Store model configuration"""
        key = f"{self._key_prefix(session_id)}:parameters"

        async with self.redis as redis:
            await redis.hset(key, mapping={
                "parameters": json.dumps(params_data)
            })
            await redis.expire(key, self.TTL_SECONDS)

        logger.info(f"Stored anisotropic model params for session {session_id}")

    async def get_model_params(self, session_id: str) -> Dict[str, Any]:
        """Retrieve model configuration"""
        key = f"{self._key_prefix(session_id)}:parameters"

        async with self.redis as redis:
            params_str = await redis.hget(key, "parameters")

        if params_str is None:
            raise ValueError(f"No model params found for session {session_id}")

        return json.loads(params_str.decode('utf-8'))

    async def set_optimization_params(self, session_id: str, parameters: Dict[str, Any]) -> None:
        """Store optimization model parameters"""
        key = f"{self._key_prefix(session_id)}:parameters"
        async with self.redis as redis:
            await redis.hset(key, mapping={'optimization_parameters': json.dumps(parameters)})
            await self.redis.expire(key, self.TTL_SECONDS)

    async def get_optimization_params(self, session_id: str) -> Dict[str, Any]:
        """Retrieve optimization model parameters"""
        key = f"{self._key_prefix(session_id)}:parameters"

        async with self.redis as redis:
            params_str = await redis.hget(key, "optimization_parameters")

        if params_str is None:
            raise ValueError(f"No fitted parameters found for session {session_id}")

        return json.loads(params_str.decode('utf-8'))

    async def set_file_data(self, session_id: str, filename: str, byte_data: bytes) -> None:
        key_data = f"{self._key_prefix(session_id)}:data"
        async with self.redis as redis:
            await redis.hset(key_data, filename, byte_data)
            await redis.expire(key_data, self.TTL_SECONDS)
        logger.info(f"Stored file data: {filename} for session {session_id}")

    async def get_files_data(self, session_id: str) -> dict:
        key_data = f"{self._key_prefix(session_id)}:data"
        async with self.redis as redis:
            return await redis.hgetall(key_data)

    async def del_file_data(self, session_id: str, filename: str) -> None:
        """Delete specific file data"""
        key_data = f"{self._key_prefix(session_id)}:data"

        async with self.redis as redis:
            await redis.hdel(key_data, filename)

        logger.info(f"Deleted file data: {filename} for session {session_id}")

    async def del_all(self, session_id: str) -> None:
        """Delete all data for session"""
        key_data = f"{self._key_prefix(session_id)}:data"
        key_config = f"{self._key_prefix(session_id)}:parameters"

        async with self.redis as redis:
            val1 = await redis.delete(key_data)
            val2 = await redis.delete(key_config)

            logger.info(f"Deleted all data for session {session_id}: "
                        f"data={val1}, params={val2}")

    async def session_exists(self, session_id: str) -> bool:
        """Check if session has any anisotropic data"""
        key_data = f"{self._key_prefix(session_id)}:data"
        key_config = f"{self._key_prefix(session_id)}:parameters"

        async with self.redis as redis:
            data_exists = await redis.exists(key_data)
            config_exists = await redis.exists(key_config)

        return bool(data_exists or config_exists)
