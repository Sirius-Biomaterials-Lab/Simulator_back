from dataclasses import dataclass

from app.modules.cache import ModuleCache


@dataclass
class IsotropicCache(ModuleCache):

    def _key_prefix(self, session_id: str) -> str:
        return f"storage:{session_id}:isotropic"
