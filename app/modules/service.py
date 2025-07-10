from abc import ABC, abstractmethod
from dataclasses import dataclass

from fastapi import UploadFile

from app.logger import logger
from app.modules.cache import ModuleCache
from app.modules.solver import ModuleSolver


@dataclass
class ModuleService(ABC):

    module_cache: ModuleCache
    solver: ModuleSolver

    @abstractmethod
    async def predict(self, session_id: str, file: UploadFile):
        raise NotImplementedError()

    @abstractmethod
    async def fit(self, session_id: str):
        raise NotImplementedError()

    @staticmethod
    @abstractmethod
    def _validate_and_process_file(filename: str, content: bytes):
        raise NotImplementedError()

    async def set_data(self, session_id: str, file: UploadFile) -> None:
        """Store uploaded file data"""
        logger.info(f"Setting data for session {session_id}, file: {file.filename}")

        content = await file.read()
        validated_content = self._validate_and_process_file(file.filename, content)
        await file.seek(0)

        await self.module_cache.set_file_data(session_id, file.filename, validated_content)

    async def delete_data(self, session_id: str, filename: str) -> None:
        """Delete specific file data"""
        await self.module_cache.del_file_data(session_id, filename)

    async def delete_all_data(self, session_id: str) -> None:
        """Delete all data for session"""
        await self.module_cache.del_all(session_id)
        logger.info(f'Deleted all anisotropic data for session {session_id}')
