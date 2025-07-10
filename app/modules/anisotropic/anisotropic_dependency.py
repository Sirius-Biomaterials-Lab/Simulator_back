from fastapi import Depends

from app.modules.anisotropic.anisotropic_cache import AnisotropicCache
from app.modules.anisotropic.service import AnisotropicService
from app.modules.anisotropic.solver import AnisotropicSolver
from app.storage.redis.accessor import get_redis_connection


async def get_anisotropic_cache_repository(redis_connection=Depends(get_redis_connection)) -> AnisotropicCache:
    """Get anisotropic cache repository instance"""
    return AnisotropicCache(redis_connection)


async def get_anisotropic_service(anisotropic_cache=Depends(get_anisotropic_cache_repository)) -> AnisotropicService:
    """Get anisotropic service instance"""
    return AnisotropicService(module_cache=anisotropic_cache, solver=AnisotropicSolver())
