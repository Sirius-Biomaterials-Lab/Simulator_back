from dataclasses import dataclass
from io import BytesIO

import numpy as np
import pandas as pd
from fastapi import UploadFile

from app.exception import DataNotCorrect, DataNotFound
from app.logger import logger
from app.modules.exception import UnsupportedFormatFile
from app.modules.isotropic.energy import EnergyInfo
from app.modules.isotropic.solver import IsotropicModelType
from app.modules.isotropic.solver.shema import IsotropicFitResponse
from app.modules.service import ModuleService


@dataclass
class IsotropicService(ModuleService):

    async def fit(self, session_id: str) -> IsotropicFitResponse:

        data_frames = []
        filename_to_bytes = await self.module_cache.get_files_data(session_id)
        # logger.info(f"filename_to_bytes: {filename_to_bytes}")
        for filename, bytes_data in reversed(filename_to_bytes.items()):

            if filename.decode('utf-8').endswith('.csv'):
                data = pd.read_csv(BytesIO(bytes_data))
            elif filename.decode('utf-8').endswith(('.xls', '.xlsx')):
                data = pd.read_excel(BytesIO(bytes_data))
            else:
                raise DataNotCorrect
            data_frames.append(data)
        if not data_frames:
            raise DataNotCorrect
        all_data = pd.concat(data_frames, ignore_index=True)

        params: dict = await self.module_cache.get_model_params(session_id)
        logger.info(f"{params=}")
        isotropic_model = IsotropicModelType(params['hyperelastic_model'])
        self.solver.setup_solver(isotropic_model)

        # optimization_params: np.ndarray[float] = self.solver.fit_model(all_data, error_function_callable)
        optimization_result = self.solver.fit(all_data)

        optimization_info = {
            'success': optimization_result.success,
            'fitted_params': optimization_result.parameters,
            'iterations': optimization_result.iterations,
            'message': optimization_result.message,
        }
        await self.module_cache.set_optimization_params(session_id,
                                                        optimization_info)
        return self.solver.graph_fit()

    async def set_model_parameters(self, session_id: str, hyperlastic_model_name: str):
        params = {"hyperelastic_model": hyperlastic_model_name}
        await self.module_cache.set_model_params(
            session_id, params)

    async def predict(self, session_id: str, file: UploadFile):
        content = await file.read()
        content = self._validate_and_process_file(file.filename, content)
        await file.seek(0)

        if file.filename.endswith('.csv'):
            data = pd.read_csv(BytesIO(content))
        elif file.filename.endswith(('.xls', '.xlsx')):
            data = pd.read_excel(BytesIO(content))
        else:
            raise DataNotCorrect
        await file.seek(0)

        params: dict = await self.module_cache.get_model_params(session_id)
        isotropic_model = IsotropicModelType(params['hyperelastic_model'])
        self.solver.setup_solver(isotropic_model)

        fitted_params: dict = (await self.module_cache.get_optimization_params(session_id))['fitted_params']
        fitted_params = np.array(fitted_params, dtype=np.float32)
        return self.solver.predict(data, fitted_params)

    async def calculate_energy(self, session_id: str) -> str:
        try:
            params: dict = await self.module_cache.get_model_params(session_id)
            isotropic_model = params['hyperelastic_model']

            fitted_params: dict = (await self.module_cache.get_optimization_params(session_id))['fitted_params']
            fitted_params = np.array(fitted_params, dtype=np.float32)
            logger.info(isotropic_model, fitted_params)
            return EnergyInfo.energy_text(name=isotropic_model, params=fitted_params)
        except (AttributeError, TypeError):
            raise DataNotFound("Model or optimization parameters not found. Please fit the model first.")

    @staticmethod
    def _validate_and_process_file(filename: str, content: bytes) -> bytes:

        buffer = BytesIO(content)

        if filename.endswith('.csv'):
            data = pd.read_csv(buffer)
        elif filename.endswith(('.xls', '.xlsx')):
            data = pd.read_excel(buffer)
        else:
            raise UnsupportedFormatFile(detail="Неподдерживаемый формат файла")

        if "lambda_y" not in data.columns:  # свободное сужение
            data["lambda_y"] = data["lambda_x"] ** -0.5
        if "stress_y_mpa" not in data.columns:
            data["stress_y_mpa"] = 0.0

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

        # buffer — ваш исходный BytesIO
        buffer.seek(0)  # возвращаем курсор в начало
        buffer.truncate()  # обрезаем всё содержимое

        if filename.endswith('.csv'):
            data.to_csv(buffer, index=False, encoding='utf-8')
        else:
            with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                data.to_excel(writer, index=False)

        buffer.seek(0)
        return buffer.getvalue()
