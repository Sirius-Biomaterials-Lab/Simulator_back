from fastapi import Depends

from app.modules.isotropic.isotropic_cache import IsotropicCache
from app.modules.isotropic.server import Service
from app.modules.isotropic.solver import IsotropicSolver
from app.storage.redis.accessor import get_redis_connection


async def get_isotropic_cache_repository(redis_connection=Depends(get_redis_connection)) -> IsotropicCache:
    return IsotropicCache(redis_connection)


async def get_service(isotropic_redis=Depends(get_isotropic_cache_repository)) -> Service:
    return Service(isotropic_cache=isotropic_redis, solver=IsotropicSolver())
