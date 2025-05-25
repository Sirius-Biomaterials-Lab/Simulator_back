from dataclasses import dataclass
from io import BytesIO
from typing import Optional

import pandas as pd
from fastapi import UploadFile

from app.modules.isotropic.solver import ErrorFunction, HyperelasticModel
from app.modules.isotropic.solver.solver import IsotropicSolver
from app.modules.isotropic.tmp_storage import Storage


@dataclass
class Service:
    # solver: IsotropicSolver
    storage: Storage
    _solver: Optional[IsotropicSolver] = None

    def data_collection(self):
        pass

    async def set_data(self, file: UploadFile):
        await self.storage.set_data(file)

    async def del_data(self, filename: str):
        await self.storage.del_data(filename=filename)

    def fit(self):
        self._solver = self._setup_solver()
        self._solver.fit_model()
        return self._solver.graph_fit()

    def set_model_and_error_name(self, hyperlastic_model_name: str, error_function_name: str):
        self.storage.set_model_and_error_name(
            hyperlastic_model_name=hyperlastic_model_name,
            error_function_name=error_function_name
        )

    def _setup_solver(self):
        error_function_callable = ErrorFunction.get_error_function(self.storage.error_function_name)
        hyperelastic_model = HyperelasticModel(self.storage.hyperlastic_model_name)
        data = self.storage.get_data()
        return IsotropicSolver(
            data=data,
            hyperelastic_model=hyperelastic_model,
            error_function=error_function_callable,
            error_function_name=self.storage.error_function_name,
        )

    async def predict(self, file: UploadFile):
        await file.seek(0)

        content = await file.read()
        buffer = BytesIO(content)
        await self.storage.check_file(file, buffer)

        if file.filename.endswith('.csv'):
            data = pd.read_csv(buffer)

        elif file.filename.endswith(('.xls', '.xlsx')):
            data = pd.read_excel(buffer)
