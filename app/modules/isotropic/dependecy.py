from app.modules.isotropic.server import Service
from app.modules.isotropic.tmp_storage import storage


async def get_service() -> Service:
    return Service(storage=storage)
