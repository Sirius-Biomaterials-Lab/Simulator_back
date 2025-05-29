import httpx


async def get_async_client() -> httpx.AsyncClient:
    return httpx.AsyncClient()
