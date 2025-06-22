from dataclasses import dataclass
from io import BytesIO

import numpy as np
import pandas as pd
from fastapi import UploadFile, HTTPException

from app.exception import DataNotCorrect, DataNotFound
from app.logger import logger
from app.modules.isotropic.energy import energy_text
from app.modules.isotropic.isotropic_cache import IsotropicCache
from app.modules.isotropic.solver import IsotropicSolver, HyperelasticModel
from app.modules.isotropic.solver.shema import IsotropicFitResponse


@dataclass
class Service:
    solver: IsotropicSolver
    isotropic_cache: IsotropicCache

    # _solver: IsotropicSolver

    def data_collection(self):
        pass

    async def set_data(self, session_id: str, file: UploadFile):
        content = await file.read()
        content = self._check_file(file.filename, content)
        await file.seek(0)
        await self.isotropic_cache.set_file_data(session_id, file.filename, content)

    async def delete_data(self, session_id: str, filename: str):
        await self.isotropic_cache.del_file_data(session_id=session_id, filename=filename)

    #
    async def delete_all_data(self, session_id: str):
        await self.isotropic_cache.del_all(session_id=session_id)
        logger.info('Delete all data')

    async def fit(self, session_id: str) -> IsotropicFitResponse:

        data_frames = []
        filename_to_bytes = await self.isotropic_cache.get_files_data(session_id)
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

        hyperelastic_model, error_function = await self.isotropic_cache.get_model_params(session_id)
        # error_function_callable = ErrorFunction.get_error_function(error_function)
        hyperelastic_model = HyperelasticModel(hyperelastic_model)

        self.solver.set_up_solver(hyperelastic_model)
        # optimization_params: np.ndarray[float] = self.solver.fit_model(all_data, error_function_callable)
        optimization_params: np.ndarray[float] = self.solver.fit_model(all_data)
        await self.isotropic_cache.set_optimization_params(session_id, optimization_params.tolist())
        return self.solver.graph_fit(optimization_params)

    async def calculate_energy(self, session_id: str) -> str:
        try:
            hyperelastic_model, _ = await self.isotropic_cache.get_model_params(session_id)
            optimization_params = await self.isotropic_cache.get_optimization_params(session_id)
            optimization_params = np.array(optimization_params, dtype=np.float32)
            logger.info(hyperelastic_model, optimization_params)
            return energy_text(name=hyperelastic_model, params=optimization_params)
        except (AttributeError, TypeError):
            raise DataNotFound("Model or optimization parameters not found. Please fit the model first.")

    async def set_model_and_error_name(self, session_id: str, hyperlastic_model_name: str, error_function_name: str):
        await self.isotropic_cache.set_model_params(
            session_id,
            hyperelastic_model=hyperlastic_model_name,
            error_function=error_function_name
        )

    async def predict(self, session_id: str, file: UploadFile):
        content = await file.read()
        content = self._check_file(file.filename, content)
        await file.seek(0)

        if file.filename.endswith('.csv'):
            data = pd.read_csv(BytesIO(content))
        elif file.filename.endswith(('.xls', '.xlsx')):
            data = pd.read_excel(BytesIO(content))
        else:
            raise DataNotCorrect
        await file.seek(0)

        hyperelastic_model, _ = await self.isotropic_cache.get_model_params(session_id)
        hyperelastic_model = HyperelasticModel(hyperelastic_model)
        self.solver.set_up_solver(hyperelastic_model)

        optimization_params = await self.isotropic_cache.get_optimization_params(session_id)
        optimization_params = np.array(optimization_params, dtype=np.float32)
        return self.solver.predict(data, optimization_params)

    @staticmethod
    def _check_file(filename: str, content: bytes) -> BytesIO:
        buffer = BytesIO(content)
        try:
            if filename.endswith('.csv'):
                data = pd.read_csv(buffer)
            elif filename.endswith(('.xls', '.xlsx')):
                data = pd.read_excel(buffer)
            else:
                raise HTTPException(status_code=400, detail="Неподдерживаемый формат файла")

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


        except DataNotCorrect as ve:
            logger.warning(f"Ошибка в данных: {ve}")
            raise HTTPException(status_code=400, detail=str(ve))
