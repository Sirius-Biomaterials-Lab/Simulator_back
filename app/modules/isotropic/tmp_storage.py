"""
позже либо S3 либо БД
"""

from io import BytesIO
from typing import Dict, Optional

import pandas as pd
from fastapi import UploadFile, HTTPException

from app.exception import DataNotCorrect
from app.logger import logger


class Storage:
    _instance = None
    data: pd.DataFrame
    storage: Dict[str, bytes]
    hyperlastic_model_name: str
    error_function_name: str

    def __new__(cls):
        if cls._instance is None:
            cls._instance.data = pd.DataFrame()
            cls._instance.storage = {}
            hyperlastic_model_name: str
            error_function_name: str
            cls._instance = super(Storage, cls).__new__(cls)
        return cls._instance

    def set_model_and_error_name(self, hyperlastic_model_name: str, error_function_name: str):
        self.error_function_name = error_function_name
        self.hyperlastic_model_name = hyperlastic_model_name

    async def set_data(self, file: UploadFile):
        await file.seek(0)

        content = await file.read()
        buffer = BytesIO(content)
        await self.check_file(file, buffer)

        self.storage[file.filename] = buffer
        logger.info(f"{file.filename} added in storage")

    async def del_data(self, filename: str):
        if filename in self.storage:
            del self.storage[filename]

    def clear_data(self):
        self.storage = {}
        del self.hyperlastic_model_name
        del self.error_function_name

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

    async def check_file(self, file: UploadFile, buffer: BytesIO):
        # Пробуем прочитать файл
        try:
            if file.filename.endswith('.csv'):
                data = pd.read_csv(buffer)
            elif file.filename.endswith(('.xls', '.xlsx')):
                data = pd.read_excel(buffer)
            else:
                raise HTTPException(status_code=400, detail="Неподдерживаемый формат файла")

            # Валидация содержимого
            required_columns = ['lambda_x', 'lambda_y', 'stress_x_mpa', 'stress_y_mpa']
            missing_columns = [col for col in required_columns if col not in data.columns]

            if missing_columns:
                raise DataNotCorrect(f"Отсутствуют необходимые колонки в данных: {missing_columns}")

            if data[required_columns].empty:
                raise DataNotCorrect("Один или несколько массивов данных пусты")

            if not all(data[required_columns].apply(
                    lambda s: pd.to_numeric(s, errors='coerce').notnull().all())):
                raise DataNotCorrect("Колонки содержат нечисловые значения")

            buffer.seek(0)
            await file.seek(0)

        except DataNotCorrect as ve:
            logger.warning(f"Ошибка в данных: {ve}")
            raise HTTPException(status_code=400, detail=str(ve))


storage = Storage()
