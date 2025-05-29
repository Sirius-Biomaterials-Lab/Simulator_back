from redis.asyncio import Redis

from app.settings import Settings

settings = Settings()


def get_redis_connection() -> Redis:
    return Redis(host=settings.CACHE_HOST, port=settings.CACHE_PORT, db=settings.CACHE_DB)
