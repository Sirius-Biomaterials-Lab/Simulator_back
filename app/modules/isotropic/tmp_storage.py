"""
позже либо S3 либо БД
"""

from io import BytesIO
from typing import Dict, Optional

import pandas as pd
from fastapi import UploadFile

from app.logger import logger


class Storage:
    _instance = None
    data: pd.DataFrame
    storage: Dict[str, bytes]
    hyperlastic_model_name: str
    error_function_name: str

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Storage, cls).__new__(cls)
            cls._instance.data = pd.DataFrame()
            cls._instance.storage = {}
            hyperlastic_model_name: str
            error_function_name: str
        return cls._instance

    def set_model_and_error_name(self, hyperlastic_model_name: str, error_function_name: str):
        self.error_function_name = error_function_name
        self.hyperlastic_model_name = hyperlastic_model_name

    async def set_data(self, file: UploadFile):
        await file.seek(0)
        self.storage[file.filename] = BytesIO(await file.read())
        logger.info(f"{file.filename} added in storage")

    async def del_data(self, filename: str):
        if filename in self.storage:
            del self.storage[filename]

    def get_data(self) -> Optional[pd.DataFrame]:
        data_frames = []
        for filename, byte_content in self.storage.items():

            if filename.endswith('.csv'):
                df = pd.read_csv(byte_content)
                data_frames.append(df)

            elif filename.endswith(('.xls', '.xlsx')):
                df = pd.read_excel(byte_content)
                data_frames.append(df)

            byte_content.seek(0)

        if data_frames:
            return pd.concat(data_frames, ignore_index=True)
        return None


storage = Storage()
