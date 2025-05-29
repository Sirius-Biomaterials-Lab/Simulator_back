import json
from dataclasses import dataclass

from redis.asyncio import Redis

from app.logger import logger
from app.settings import settings


@dataclass
class IsotropicCache:
    redis: Redis
    TTL_SECONDS = settings.ISOTROPIC_TTL_SECONDS

    def _key_prefix(self, session_id: str) -> str:
        return f"storage:{session_id}"

    async def set_model_params(self, session_id: str, hyperelastic_model: str, error_function: str) -> None:
        key = f"{self._key_prefix(session_id)}:isotropic_meta"

        async with self.redis as redis:
            await redis.hset(key, mapping={
                "hyperelastic_model": hyperelastic_model,
                "error_function": error_function
            })
            await self.redis.expire(key, self.TTL_SECONDS)

    async def set_optimization_params(self, session_id: str, params: list) -> None:
        key = f"{self._key_prefix(session_id)}:isotropic_meta"
        async with self.redis as redis:
            await redis.hset(key, mapping={'optimization_params': json.dumps(params)})
            await self.redis.expire(key, self.TTL_SECONDS)

    async def set_file_data(self, session_id: str, filename: str, byte_data: bytes) -> None:
        key_data = f"{self._key_prefix(session_id)}:isotropic_data"
        logger.info('Added: {}'.format(key_data, filename))
        async with self.redis as redis:
            await redis.hset(key_data, filename, byte_data)
            await redis.expire(key_data, self.TTL_SECONDS)

    async def get_model_params(self, session_id) -> tuple:
        key = f"{self._key_prefix(session_id)}:isotropic_meta"
        async with self.redis as redis:
            hyperelastic_model = await redis.hget(key, "hyperelastic_model")
            error_function = await redis.hget(key, "error_function")
        return hyperelastic_model.decode('utf-8'), error_function.decode('utf-8')

    async def get_files_data(self, session_id: str) -> dict:
        key_data = f"{self._key_prefix(session_id)}:isotropic_data"
        async with self.redis as redis:
            return await redis.hgetall(key_data)

    async def get_optimization_params(self, session_id: str) -> None:
        key = f"{self._key_prefix(session_id)}:isotropic_meta"
        async with self.redis as redis:
            optimization_params = await redis.hget(key, "optimization_params")
        return json.loads(optimization_params)

    async def del_file_data(self, session_id: str, filename: str) -> None:
        key_data = f"{self._key_prefix(session_id)}:isotropic_data"
        async with self.redis as redis:
            await redis.hdel(key_data, filename)

    async def del_all(self, session_id: str) -> None:
        key_data = f"{self._key_prefix(session_id)}:isotropic_data"
        key_info = f"{self._key_prefix(session_id)}:isotropic_meta"
        async with self.redis as redis:
            await redis.hgetall(key_data)
            await redis.hgetall(key_info)
