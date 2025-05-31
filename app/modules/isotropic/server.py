from dataclasses import dataclass
from io import BytesIO

import numpy as np
import pandas as pd
from fastapi import UploadFile, HTTPException

from app.exception import DataNotCorrect
from app.logger import logger
from app.modules.isotropic.isotropic_cache import IsotropicCache
from app.modules.isotropic.solver import IsotropicSolver, ErrorFunction, HyperelasticModel
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
        self._check_file(file.filename, content)
        await file.seek(0)
        await self.isotropic_cache.set_file_data(session_id, file.filename, content)

    async def delete_data(self, session_id: str, filename: str):
        await self.isotropic_cache.del_file_data(session_id=session_id, filename=filename)

    #
    async def delete_all_data(self, session_id: str):
        logger.info('aaaaaaaaaaaaaaaaaaaaaaa')
        await self.isotropic_cache.del_all(session_id=session_id)

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
        error_function_callable = ErrorFunction.get_error_function(error_function)
        hyperelastic_model = HyperelasticModel(hyperelastic_model)

        self.solver.set_up_solver(hyperelastic_model)
        optimization_params: np.ndarray[float] = self.solver.fit_model(all_data, error_function_callable)
        await self.isotropic_cache.set_optimization_params(session_id, optimization_params.tolist())
        return self.solver.graph_fit(optimization_params)

    async def set_model_and_error_name(self, session_id: str, hyperlastic_model_name: str, error_function_name: str):
        await self.isotropic_cache.set_model_params(
            session_id,
            hyperelastic_model=hyperlastic_model_name,
            error_function=error_function_name
        )

    async def predict(self, session_id: str, file: UploadFile):
        content = await file.read()
        self._check_file(file.filename, content)
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
    def _check_file(filename: str, content: bytes):
        buffer = BytesIO(content)
        try:
            if filename.endswith('.csv'):
                data = pd.read_csv(buffer)
            elif filename.endswith(('.xls', '.xlsx')):
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


        except DataNotCorrect as ve:
            logger.warning(f"Ошибка в данных: {ve}")
            raise HTTPException(status_code=400, detail=str(ve))
