import json
from dataclasses import dataclass
from typing import Dict, Any

from redis.asyncio import Redis

from app.logger import logger
from app.modules.cache import ModuleCache
from app.settings import settings


@dataclass
class AnisotropicCache(ModuleCache):
    """Cache layer for anisotropic module data using Redis"""

    redis: Redis
    TTL_SECONDS = settings.ISOTROPIC_TTL_SECONDS  # Reuse same TTL setting

    def _key_prefix(self, session_id: str) -> str:
        return f"storage:{session_id}:anisotropic"

